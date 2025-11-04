module Depth_synthesis
using HerbSearch, HerbGrammar, HerbBenchmarks, HerbConstraints, HerbCore, HerbSpecification, HerbInterpret
using DataStructures
import HerbSearch: add_to_bank!, calculate_new_horizon, calc_measure,
                   get_bank, get_grammar, get_measure_limit, build_cost_tensor,
                   unique_sorted_costs, UniformIterator, UniformSolver, Axis,
                   UniformTreeEntry, CostBank, get_solver, enqueue_entry_costs!,
                   get_tree, GenericBUState
import HerbCore: UniformHole, AbstractUniformHole, Hole
import HerbSearch: AbstractCostBasedBottomUpIterator, CostBank, CostAccessAddress, indices_at_cost, _pathvec

export DepthCostBasedBottomUpIterator,
       apply_probe_update!,           # <- new: single call from your outer loop
       program_rule_count             # <- handy metric for “smaller is better” tie-break

mutable struct ResetVariables
    depth_exhausted::Bool
    reset_count::Int
    max_resets::Int
    seen_shapes::Set{UInt64}
    seen_amount::Int
end

InitialiseResetVariables() = ResetVariables(
    false,   # depth_exhausted
    0,        # reset_count
    100,
    Set{UInt64}(),
    0
)

# ---------- iterator definition (add JIT fields & toggle) ----------
@programiterator DepthCostBasedBottomUpIterator(
    bank=CostBank(),
    max_cost::Float64=Inf,
    # costs are "log probs" (i.e., -log2 p) laid out as (rules × depth)
    logp_by_depth::Matrix{Float64},        # current, mutable costs
    base_logp_by_depth::Matrix{Float64}=copy(logp_by_depth),   # frozen base costs
    fit::Matrix{Float64}=zeros(size(logp_by_depth)),            # same shape: (rules × depth)
    jit_enabled::Bool=false,
    reset_variables=InitialiseResetVariables(),         
) <: AbstractCostBasedBottomUpIterator

get_reset_variables(iter::DepthCostBasedBottomUpIterator) = iter.reset_variables

# BOUNDED MAX-HEAP OF BEST PARENTS BY DEPTH-AWARE LB
struct ParentCand
    lb::Float64
    hole::UniformHole
    fp::UInt64
end
Base.isless(a::ParentCand, b::ParentCand) = a.lb > b.lb

function build_depth_aware_axes(
    iter::DepthCostBasedBottomUpIterator,
    grammar::AbstractGrammar,
    hole::UniformHole;
    path::Tuple{Vararg{Int}}=(),
    depth::Int=1,
)
    # Costs are stored as (rules × depth)
    @inline _depth_cols(iter)::Int = size(iter.logp_by_depth, 2)

    @inline _cost(iter, ridx::Int, depth::Int)::Float64 = iter.logp_by_depth[ridx, depth]
    axes = Axis[]
    Dlim = get_max_depth(iter)               # from HerbSearch, already stored on iter
    d = min(depth, _depth_cols(iter))        # still keep read guard

    if isempty(hole.children)
        # Leaf hole: only terminals anyway
        term_mask  = grammar.isterminal .& hole.domain
        term_inds  = findall(term_mask)
        isempty(term_inds) && return axes   # dead end leaf
        term_costs = Float64[_cost(iter, ridx, d) for ridx in term_inds]
        push!(axes, Axis(path, term_inds, term_costs))
        return axes
    end

    if depth >= Dlim
        # At max depth: FORBID operators that would add children.
        # Keep only terminal rules from this domain.
        term_mask = grammar.isterminal .& hole.domain
        term_inds = findall(term_mask)
        if isempty(term_inds)
            # No valid terminals → no expansions from this hole (dead end)
            return axes  # empty; caller will see no options
        end
        term_costs = Float64[_cost(iter, ridx, d) for ridx in term_inds]
        push!(axes, Axis(path, term_inds, term_costs))
        return axes
    end

    # Below max depth: allow operator choices (nonterminals) and recurse into children
    op_mask  = (.~grammar.isterminal) .& hole.domain
    op_inds  = findall(op_mask)
    isempty(op_inds) && return axes   # no ops → nothing buildable here
    op_costs = Float64[_cost(iter, ridx, d) for ridx in op_inds]
    push!(axes, Axis(path, op_inds, op_costs))

    @inbounds for (j, ch) in pairs(hole.children)
        child_path = (path..., j)
        append!(axes, build_depth_aware_axes(iter, grammar, ch; path=child_path, depth=depth+1))
    end
    return axes
end


function HerbSearch.add_to_bank!(iter::DepthCostBasedBottomUpIterator, uh::UniformHole)::Int
    grammar = get_grammar(iter.solver)
    bank    = get_bank(iter)
    axes = build_depth_aware_axes(iter, grammar, uh)
    T    = build_cost_tensor(axes)
    sorted_costs = unique_sorted_costs(T)
    rtype = HerbGrammar.return_type(grammar, uh)
    uh_id = (bank.next_id[] += 1) - 1
    usolver = UniformSolver(grammar, uh, with_statistics=get_solver(iter).statistics)
    uiter   = UniformIterator(usolver, iter)
    bank.uh_index[uh_id] = UniformTreeEntry(uh, axes, T, sorted_costs, rtype, true, uiter)
    return uh_id
end

# ---------- helper: rule index + children (UniformHole / StateHole) ----------
_ruleindex(u::UniformHole) = begin
    j = findfirst(u.domain)
    @assert j !== nothing "UniformHole has empty/invalid domain"
    j
end
_children(u::UniformHole) = u.children

_ruleindex(u::HerbConstraints.StateHole) = begin
    j = findfirst(u.domain)
    @assert j !== nothing "StateHole has empty/invalid domain"
    j
end
_children(u::HerbConstraints.StateHole) = u.children

# Optional metric you used in the old loop
function program_rule_count(node::Union{UniformHole,HerbConstraints.StateHole})
    total = 1
    @inbounds for c in _children(node)
        total += program_rule_count(c)
    end
    return total
end

# function HerbSearch.enqueue_entry_costs!(
#     iter::DepthCostBasedBottomUpIterator,
#     uh_id::Int, 
#     n::Int  # how many lowest-cost trees to enqueue
# )
#     bank  = get_bank(iter)
#     ent   = bank.uh_index[uh_id]
#     limit = get_measure_limit(iter)
#     costs = ent.sorted_costs

#     # keep indices with cost ≤ limit
#     idxs = [i for (i, c) in pairs(costs) if c ≤ limit]
#     isempty(idxs) && return nothing

#     # choose the n smallest by cost
#     if n < length(idxs)
#         perm = partialsortperm(idxs, 1:n; by = i -> costs[i])
#         idxs = idxs[perm]
#     else
#         sort!(idxs; by = i -> costs[i])
#     end

#     @inbounds for i in idxs
#         c = costs[i]
#         enqueue!(bank.pq, (uh_id, i), c)
#     end
#     return nothing
# end

function HerbSearch.populate_bank!(iter::DepthCostBasedBottomUpIterator)
    grammar = get_grammar(iter.solver)
    bank    = get_bank(iter)

    new_ids = Int[]
    for t in unique(grammar.types)
        term_mask = grammar.isterminal .& grammar.domains[t]
        if any(term_mask)
            uh = UniformHole(term_mask, [])
            uh_id = add_to_bank!(iter, uh)
            push!(new_ids, uh_id)
        end
    end

    # Incrementally enqueue only the new seed entries
    for uh_id in new_ids
        enqueue_entry_costs!(iter, uh_id)
    end

    # Collect all addresses within the initial window [last_horizon, new_horizon)
    out   = CostAccessAddress[]
    limit = get_measure_limit(iter)
    for (key, cost) in bank.pq
        (uh_id, _idx) = key
        ent  = bank.uh_index[uh_id]
        idxs = indices_at_cost(iter, ent, cost)
        @inbounds for i in 1:length(idxs)
            push!(out, CostAccessAddress(uh_id, cost, i))
        end
    end
    sort!(out; by = a -> a.cost)
    return out
end

@inline _Dcols(iter)::Int = size(iter.logp_by_depth, 2)

# Min atom cost over a domain at depth d (clamped to last column if d > D)
function _min_domain_cost_at_depth(iter::DepthCostBasedBottomUpIterator,
                                   domain::BitVector,
                                   d::Int)
    dclamp = min(d, _Dcols(iter))
    idxs = findall(domain)
    @inbounds return minimum(iter.logp_by_depth[i, dclamp] for i in idxs)
end

# Depth-aware lower bound for a UniformHole (operators at depth d, children at d+1)
function _lb_uniformhole!(iter::DepthCostBasedBottomUpIterator,
                          grammar::AbstractGrammar,
                          node::UniformHole,
                          d::Int)::Float64
    if isempty(node.children)
        term_mask = grammar.isterminal .& node.domain
        return any(term_mask) ?
            _min_domain_cost_at_depth(iter, term_mask, d) :
            _min_domain_cost_at_depth(iter, node.domain, d)
    else
        op_mask = (.~grammar.isterminal) .& node.domain
        op_cost = any(op_mask) ?
            _min_domain_cost_at_depth(iter, op_mask, d) :
            _min_domain_cost_at_depth(iter, node.domain, d)
        @inbounds for ch in node.children
            op_cost += _lb_uniformhole!(iter, grammar, ch, d+1)
        end
        return op_cost
    end
end

_lb_uniformhole(iter::DepthCostBasedBottomUpIterator,
                grammar::AbstractGrammar,
                hole::UniformHole)::Float64 = _lb_uniformhole!(iter, grammar, hole, 1)

@inline function _mix(h::UInt64, x::UInt64)
    h ⊻= x
    return h * 0x00000100000001B3 % UInt64  # FNV-1a style mix
end

function shape_fingerprint(u::UniformHole)::UInt64
    # Hash the domain bitvector and recursively the children
    h = 0xcbf29ce484222325 % UInt64
    # domain bits
    @inbounds for b in u.domain
        h = _mix(h, UInt64(b))
    end
    # children
    @inbounds for c in u.children
        h = _mix(h, shape_fingerprint(c))
    end
    return h
end

function tree_depth(node)::Int
    isempty(_children(node)) && return 1
    m = 0
    @inbounds for c in _children(node)
        d = tree_depth(c); d > m && (m = d)
    end
    return 1 + m
end

function HerbSearch.combine(iter::DepthCostBasedBottomUpIterator, state)
    bank       = get_bank(iter)
    reset_variables       = get_reset_variables(iter)
    grammar    = get_grammar(iter.solver)
    size_limit  = get_max_size(iter)
    depth_limit = get_max_depth(iter)
    enqueued_this_call = Set{UInt64}()  # prevents duplicate heap inserts this pass

    ######DEBUGGING######
    println("UH INDEX : ", length(bank.uh_index))
    count_uh_programs(bank) = begin
        total = 0
        for ent in values(bank.uh_index)
            total += isempty(ent.axes) ? 0 : prod(length(ax.options) for ax in ent.axes)
        end
        println("Total programs in UH-index: ", total); total
    end
    count_uh_programs(bank)
    ######################

    
    # tune or promote to a field later
    top_n = 30

    newly_flagged_ids = Set{Int}()
    for (id, ent) in bank.uh_index
        ent.new_shape && push!(newly_flagged_ids, id)
    end

    bytype = Dict{Symbol, Vector{Tuple{Int,UniformTreeEntry}}}()
    for (id, ent) in bank.uh_index
        push!(get!(bytype, ent.rtype, Tuple{Int,UniformTreeEntry}[]), (id, ent))
    end

    terminals = grammar.isterminal
    nonterm   = .~terminals
    shapes    = UniformHole.(partition(Hole(nonterm), grammar), ([],))

    total_child_tuples  = 0
    pruned_by_depthsize = 0
    heap = BinaryMaxHeap{ParentCand}()

    for shape in shapes
        rule_idx = findfirst(shape.domain)
        rule_idx === nothing && continue
        child_types = Tuple(grammar.childtypes[rule_idx])

        candidates = Vector{Vector{Tuple{Int,UniformTreeEntry}}}(undef, length(child_types))
        feasible = true
        @inbounds for i in 1:length(child_types)
            lst = get(bytype, child_types[i], nothing)
            if lst === nothing || isempty(lst)
                feasible = false; break
            end
            candidates[i] = lst
        end
        feasible || continue

        for tuple_children in Iterators.product(candidates...)
            any_new = any( (id ∈ newly_flagged_ids) for (id, _e) in tuple_children )
            any_new || continue
            total_child_tuples += 1
            parent_hole = UniformHole(
                shape.domain,
                UniformHole[e.hole for (_id, e) in tuple_children]
            )
            finger = shape_fingerprint(parent_hole)
            if finger ∈ reset_variables.seen_shapes
                # We built this parent shape in a previous epoch; skip it
                reset_variables.seen_amount += 1
                # continue
            end
            # Skip if we already pushed this shape into the heap in THIS combine call
            # if finger ∈ enqueued_this_call
            #     println("")
            # end
            if length(parent_hole) > size_limit || program_rule_count(parent_hole) > tree_depth(parent_hole) * 2 || tree_depth(parent_hole) > 9 || program_rule_count(parent_hole) > 12
                pruned_by_depthsize += 1
                continue
            end

            lb = _lb_uniformhole(iter, grammar, parent_hole)  # depth-aware from logp_by_depth
            push!(heap, ParentCand(lb, parent_hole, finger))
            push!(enqueued_this_call, finger)  # local mark only
            if length(heap) > top_n
                pop!(heap)  # evict worst (highest) lb
            end
        end
    end

    # clear flags
    for id in newly_flagged_ids
        bank.uh_index[id].new_shape = false
    end

    # materialize only the kept parents
    added_ids = Int[]
    while !isempty(heap)
        cand = pop!(heap)
        if cand.fp ∈ reset_variables.seen_shapes
            # continue
        end
        uh_id = add_to_bank!(iter, cand.hole)
        push!(added_ids, uh_id)
        push!(reset_variables.seen_shapes, cand.fp)
    end

    for uh_id in added_ids
        enqueue_entry_costs!(iter, uh_id)
    end
       
    reset_variables.depth_exhausted = (
        isempty(added_ids) &&       # nothing new could be added
        total_child_tuples > 0 &&   # there were tuples to try
        pruned_by_depthsize == total_child_tuples # all were blocked by depth/size
    )

    out = CostAccessAddress[]
    for uh_id in added_ids
        ent  = bank.uh_index[uh_id]
        for (i, c) in pairs(ent.sorted_costs)
            c ≤ get_measure_limit(iter) || continue
            idxs = indices_at_cost(iter, ent, c)
            @inbounds for k in 1:length(idxs)
                push!(out, CostAccessAddress(uh_id, c, k))
            end
        end
    end
    println("\n SEEEN AMOUNT : ", reset_variables.seen_amount)
    println("OUT SIZE : ", length(out))
    sort!(out; by = a -> a.cost)
    return out, state
end

# Update FIT along the program's tree using coverage cov = hits/N
function update_fit_from_program!(
    fit::AbstractMatrix{Float64},
    prog::Union{UniformHole,HerbConstraints.StateHole},
    hits::Int, N::Int, depth::Int=1
)
    cov = (N == 0) ? 0.0 : clamp(hits / N, 0.0, 1.0)
    _upd_fit!(fit, prog, cov, depth)
    return fit
end

function _upd_fit!(
    fit::AbstractMatrix{Float64},
    node::Union{UniformHole,HerbConstraints.StateHole},
    cov::Float64, depth::Int
)
    D = size(fit, 2)                 # note: (rules × depth)
    d = clamp(depth, 1, D)
    j = _ruleindex(node)
    fit[j, d] = max(fit[j, d], cov)  # keep the best coverage seen so far
    @inbounds for c in _children(node)
        _upd_fit!(fit, c, cov, depth + 1)
    end
end

# In-cost space version of PROBE update:
#   P'_d(r) ∝ P_base_d(r)^(1 - FIT_d(r))
#   cost = -log2 P, so:
#   base cost C0 = -log2 P_base
#   b(d,r) = (1 - FIT) * C0
#   normalize within same LHS at each depth via log-sum-exp
function probe_update_costs!(
    dest_costs::AbstractMatrix{Float64},     # (rules × depth) ← updated
    base_costs::AbstractMatrix{Float64},     # (rules × depth) frozen base
    grammar::AbstractGrammar,
    fit::AbstractMatrix{<:Real}              # (rules × depth)
)
    @assert size(dest_costs) == size(base_costs) == size(fit)
    R, D = size(base_costs)
    types = grammar.types
    # group rule indices by LHS type once
    by_type = Dict{Symbol, Vector{Int}}()
    for j in 1:R
        push!(get!(by_type, types[j], Int[]), j)
    end

    @inbounds for d in 1:D
        for (t, idxs) in by_type
            # b = (1 - FIT) * C0  (with clamp to avoid negative)
            b = similar(base_costs, length(idxs))
            for (k, j) in pairs(idxs)
                b[k] = max(0.0, 1 - clamp(fit[j, d], 0.0, 1.0)) * base_costs[j, d]
            end
            # log-sum-exp in base 2: Z = -m + log2(∑ 2^(-(x - m)))
            m = minimum(b)
            K = 0.0
            for x in b
                K += 2.0 ^ (-(x - m))
            end
            Z = -m + log2(K)
            # final updated costs: b + Z
            for (k, j) in pairs(idxs)
                dest_costs[j, d] = b[k] + Z
            end
        end
    end
    return dest_costs
end

# Convenience entry-point you call from your outer “evaluation loop”.
# Use your “improvement policy” (more hits OR same hits & smaller tree) before calling this.
function apply_probe_update!(
    iter::DepthCostBasedBottomUpIterator,
    prog::Union{UniformHole,HerbConstraints.StateHole},
    hits::Int, total::Int
)
    depth = 1
    iter.jit_enabled || return false
    println(freeze_state(prog))
    rt = HerbGrammar.return_type(get_grammar(iter.solver), freeze_state(prog))
    @assert rt == :Start || rt == :ControlFlow || rt == :Block || rt == :Action "apply_probe_update! needs a :Start-rooted program for depth-accurate updates (got $rt)"
    
    if rt == :ControlFlow
        depth = depth + 2
    elseif  rt == :Block
        depth = depth + 1
    elseif rt == :Action
        depth = depth + 2
    end
    # 1) update FIT on the path of this program
    update_fit_from_program!(iter.fit, prog, hits, total, depth)
    # 2) recompute current costs from base costs + FIT
    probe_update_costs!(iter.logp_by_depth, iter.base_logp_by_depth, get_grammar(iter.solver), iter.fit)
    return true
end

function _reset_bank_and_seed!(iter::DepthCostBasedBottomUpIterator)
    bank = get_bank(iter)
    empty!(bank.uh_index)
    empty!(bank.pq)
    bank.next_id[] = 1

    # Re-seed from terminals using your populate function that enqueues entries.
    # We don't return addrs here; the outer iterate() will re-call populate_bank!
    nothing
end

function Base.iterate(iter::DepthCostBasedBottomUpIterator)
    grammar = get_grammar(iter.solver)
    # Check whether all probabilities are negative, or all costs are positive.
#    @assert all(p <= 0 for p in grammar.log_probabilities) ||
#         all(c > 0 for c in grammar.log_probabilities)
#
    addrs = populate_bank!(iter)
    solver = get_solver(iter)
    start  = get_tree(solver)
    st = GenericBUState(addrs, nothing, nothing, start)
    return Base.iterate(iter, st)
end


function Base.iterate(iter::DepthCostBasedBottomUpIterator, state::GenericBUState)

    reset_variables = get_reset_variables(iter)
    if reset_variables.depth_exhausted && (reset_variables.reset_count < reset_variables.max_resets)
        reset_variables.reset_count += 1
        reset_variables.depth_exhausted = false  # consume the f
        _reset_bank_and_seed!(iter)
        return Base.iterate(iter)
    end
    if isempty(state.combinations)
        addrs, _ = combine(iter, state)
        if isempty(addrs)
            return nothing
        end
        state.combinations = addrs
    end

    addr = popfirst!(state.combinations)
    prog = retrieve(iter, addr)
    return prog, state
end


end # module
