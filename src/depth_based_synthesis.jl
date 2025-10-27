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

# ---------- iterator definition (add JIT fields & toggle) ----------
@programiterator DepthCostBasedBottomUpIterator(
    bank=CostBank(),
    max_cost::Float64=Inf,
    # costs are "log probs" (i.e., -log2 p) laid out as (rules × depth)
    logp_by_depth::Matrix{Float64},        # current, mutable costs
    base_logp_by_depth::Matrix{Float64}=copy(logp_by_depth),   # frozen base costs
    fit::Matrix{Float64}=zeros(size(logp_by_depth)),            # same shape: (rules × depth)
    jit_enabled::Bool=false                                                   # toggle
) <: AbstractCostBasedBottomUpIterator


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

# ---------- lower bound utility (unchanged from your new code) ----------
function min_entry_lb_at_depth(iter, ent::UniformTreeEntry, d0::Int)
    D = size(iter.logp_by_depth, 2)
    lb = 0.0
    @inbounds for ax in ent.axes
        d_abs = d0 + length(ax.path)
        d = d_abs ≤ D ? d_abs : D
        m = Inf
        for ridx in ax.options
            c = iter.logp_by_depth[ridx, d]
            if c < m; m = c; end
        end
        lb += m
    end
    return lb
end

# ---------- horizon ----------
function calculate_new_horizon(iter::AbstractCostBasedBottomUpIterator)::Float64
    bank    = get_bank(iter)
    grammar = get_grammar(iter.solver)
    limit   = get_measure_limit(iter)

    bytype = Dict{Symbol, Vector{UniformTreeEntry}}()
    for (_, ent) in bank.uh_index
        push!(get!(bytype, ent.rtype, UniformTreeEntry[]), ent)
    end

    terminals = grammar.isterminal
    nonterm   = .~terminals
    shapes    = UniformHole.(partition(Hole(nonterm), grammar), ([],))

    best = Inf
    D = size(iter.logp_by_depth, 2)
    d_op = 1
    d_child0 = d_op + 1

    for shape in shapes
        rule_idx = findfirst(shape.domain)
        rule_idx === nothing && continue
        child_types = Tuple(grammar.childtypes[rule_idx])

        candidate_lists = Vector{Vector{UniformTreeEntry}}(undef, length(child_types))
        feasible = true
        @inbounds for i in 1:length(child_types)
            lst = get(bytype, child_types[i], nothing)
            if lst === nothing || isempty(lst)
                feasible = false; break
            end
            candidate_lists[i] = lst
        end
        feasible || continue

        op_min = Inf
        @inbounds for ridx in findall(shape.domain)
            d = d_op ≤ D ? d_op : D
            c = iter.logp_by_depth[ridx, d]
            if c < op_min; op_min = c; end
        end
        isfinite(op_min) || continue

        for tuple_children in Iterators.product(candidate_lists...)
            any_new = any(e.new_shape for e in tuple_children)
            any_new || continue

            lb = op_min
            @inbounds for e in tuple_children
                lb += min_entry_lb_at_depth(iter, e, d_child0)
            end
            if lb < best && lb ≤ limit
                best = lb
            end
        end
    end
    return best
end

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
        term_inds  = findall(hole.domain)
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

    # Below max depth: allow operator choices and recurse into children
    op_inds  = findall(hole.domain)
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

# function populate_bank!(iter::DepthCostBasedBottomUpIterator)
#     grammar = get_grammar(iter.solver)
#     bank    = get_bank(iter)

#     new_ids = Int[]
#     for t in unique(grammar.types)
#         term_mask = grammar.isterminal .& grammar.domains[t]
#         if any(term_mask)
#             uh = UniformHole(term_mask, [])
#             uh_id = add_to_bank!(iter, uh)
#             push!(new_ids, uh_id)
#         end
#     end

#     # Incrementally enqueue only the new seed entries
#     for uh_id in new_ids
#         enqueue_entry_costs!(iter, uh_id)
#     end

#     # Establish the very first horizon boundary
#     bank.new_horizon = calculate_new_horizon(iter)
    
#     # Collect all addresses within the initial window [last_horizon, new_horizon)
#     out   = CostAccessAddress[]
#     limit = get_measure_limit(iter)
#     styp = get_starting_symbol(iter.solver)

#     for (key, cost) in bank.pq
#         if cost ≥ bank.last_horizon && cost < bank.new_horizon && cost ≤ limit
#             (uh_id, _idx) = key
#             ent  = bank.uh_index[uh_id]
#             ent.rtype == styp || continue

#             idxs = indices_at_cost(iter, ent, cost)
#             @inbounds for i in 1:length(idxs)
#                 push!(out, CostAccessAddress(uh_id, cost, i))
#             end
#         end
#     end
#     sort!(out; by = a -> a.cost)
#     return out
# end

# function combine(iter::DepthCostBasedBottomUpIterator, state)
#     bank    = get_bank(iter)
#     grammar = get_grammar(iter.solver)

#     size_limit = get_max_size(iter)
#     depth_limit = get_max_depth(iter)

#     newly_flagged_ids = Set{Int}()
#     for (id, ent) in bank.uh_index
#         if ent.new_shape
#             push!(newly_flagged_ids, id)
#         end
#     end

#     bytype = Dict{Symbol, Vector{Tuple{Int,UniformTreeEntry}}}()
#     for (id, ent) in bank.uh_index
#         push!(get!(bytype, ent.rtype, Tuple{Int,UniformTreeEntry}[]), (id, ent))
#     end

#     terminals = grammar.isterminal
#     nonterm   = .~terminals
#     shapes    = UniformHole.(partition(Hole(nonterm), grammar), ([],))

#     added_ids = Int[]

#     for shape in shapes
#         rule_idx = findfirst(shape.domain)
#         rule_idx === nothing && continue
#         child_types = Tuple(grammar.childtypes[rule_idx])

#         candidates = Vector{Vector{Tuple{Int,UniformTreeEntry}}}(undef, length(child_types))
#         feasible = true
#         @inbounds for i in 1:length(child_types)
#             lst = get(bytype, child_types[i], nothing)
#             if lst === nothing || isempty(lst)
#                 feasible = false; break
#             end
#             candidates[i] = lst
#         end
#         feasible || continue

#         for tuple_children in Iterators.product(candidates...)
#             any_new = any( (id ∈ newly_flagged_ids) for (id, _e) in tuple_children )
#             any_new || continue # At least one newly found shape must be present
#             parent_hole = UniformHole(shape.domain, UniformHole[e.hole for (_id, e) in tuple_children])
#             if length(parent_hole) > size_limit || 
#                depth(parent_hole) > depth_limit 
#                 continue
#             end
#             uh_id = add_to_bank!(iter, parent_hole) 
#             push!(added_ids, uh_id)
#         end
#     end

#     for id in newly_flagged_ids
#         bank.uh_index[id].new_shape = false
#     end

#     for uh_id in added_ids
#         enqueue_entry_costs!(iter, uh_id)
#     end

#     bank.last_horizon = bank.new_horizon
#     bank.new_horizon  = calculate_new_horizon(iter)

#     out   = CostAccessAddress[]
#     limit = get_measure_limit(iter)
#     styp = get_starting_symbol(iter.solver)

#     for (key, cost) in bank.pq
#         if cost ≥ bank.last_horizon && cost < bank.new_horizon && cost ≤ limit
#             (uh_id, _idx_in_sorted) = key
#             ent  = bank.uh_index[uh_id]
#             ent.rtype == styp || continue
#             idxs = indices_at_cost(iter, ent, cost)
#             @inbounds for i in 1:length(idxs)
#                 push!(out, CostAccessAddress(uh_id, cost, i))
#             end
#         end
#     end

#     sort!(out; by = a -> a.cost)
#     return out, state
# end
# --- Add this helper once ---
# get_starting_symbol
# _start_type(iter) = HerbGrammar.return_type(get_grammar(iter.solver), get_tree(get_solver(iter)))

# --- Replace your Base.iterate(iter, state) with this filtered version ---
# function Base.iterate(iter::AbstractCostBasedBottomUpIterator, state::GenericBUState)
#     # We already filter to styp inside populate_bank!/combine, so no need to
#     # re-check the type here.
#     while true
#         if isempty(state.combinations)
#             # keep pushing horizon until we get some Start addrs, or no progress
#             while true
#                 addrs, _ = combine(iter, state)
#                 if !isempty(addrs)
#                     state.combinations = addrs
#                     break
#                 end
#                 # no addrs this slice; check if we can still progress
#                 bank = get_bank(iter)
#                 # combine() has just updated last_horizon/new_horizon
#                 if bank.new_horizon == bank.last_horizon || !isfinite(bank.new_horizon)
#                     return nothing  # exhausted
#                 end
#                 # otherwise loop again to advance to the next window
#             end
#         end

#         # pop next address and reconstruct the concrete program
#         addr = popfirst!(state.combinations)
#         prog = retrieve(iter, addr)
#         if prog === nothing
#             continue  # skip infeasible reconstructions
#         end
#         return prog, state  # already Start-typed by construction
#     end
# end


# function HerbSearch.iterate(iter::DepthCostBasedBottomUpIterator)
#     grammar = get_grammar(iter.solver)
#     # Check whether all probabilities are negative, or all costs are positive.
#     # @assert all(p <= 0 for p in iter.logp_by_depth) ||
#     #      all(c > 0 for c in iter.logp_by_depth)
    
#     addrs = populate_bank!(iter)
#     solver = get_solver(iter)
#     start  = get_tree(solver)
#     st = GenericBUState(addrs, nothing, nothing, start)
#     return Base.iterate(iter, st)
# end


# ================================================================
#         PROBE-style JIT learning (ported & adapted)
# ================================================================

# Update FIT along the program's tree using coverage cov = hits/N
function update_fit_from_program!(
    fit::AbstractMatrix{Float64},
    prog::Union{UniformHole,HerbConstraints.StateHole},
    hits::Int, N::Int; depth::Int=1
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
    println("R : ", R, " D : ", D)
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
    iter.jit_enabled || return false
    println(freeze_state(prog))
    rt = HerbGrammar.return_type(get_grammar(iter.solver), freeze_state(prog))
    @assert rt == :Start "apply_probe_update! needs a :Start-rooted program for depth-accurate updates (got $rt)"

    # 1) update FIT on the path of this program
    update_fit_from_program!(iter.fit, prog, hits, total)
    # 2) recompute current costs from base costs + FIT
    probe_update_costs!(iter.logp_by_depth, iter.base_logp_by_depth, get_grammar(iter.solver), iter.fit)
    return true
end

end # module
