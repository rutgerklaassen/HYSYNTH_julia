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
export DepthCostBasedBottomUpIterator

# --- Pretty printing utilities ----------------------------------------------


@programiterator DepthCostBasedBottomUpIterator(
    bank=CostBank(),
    max_cost::Float64=Inf,
    logp_by_depth::Matrix{Float64}   # rows = rules, cols = depth (root=1)
) <: AbstractCostBasedBottomUpIterator

function min_entry_lb_at_depth(iter, ent::UniformTreeEntry, d0::Int)
    D = size(iter.logp_by_depth, 2)           # number of depth columns
    lb = 0.0
    @inbounds for ax in ent.axes
        # absolute depth for this axis
        d_abs = d0 + length(ax.path)
        d = d_abs ≤ D ? d_abs : D             # clamp to last column
        # min across allowed rules on this axis at depth d
        m = Inf
        for ridx in ax.options
            c = iter.logp_by_depth[ridx, d]
            if c < m; m = c; end
        end
        lb += m
    end
    return lb
end


function calculate_new_horizon(iter::AbstractCostBasedBottomUpIterator)::Float64
    bank    = get_bank(iter)
    grammar = get_grammar(iter.solver)
    limit   = get_measure_limit(iter)

    # group entries by return type
    bytype = Dict{Symbol, Vector{UniformTreeEntry}}()
    for (_, ent) in bank.uh_index
        push!(get!(bytype, ent.rtype, UniformTreeEntry[]), ent)
    end

    terminals = grammar.isterminal
    nonterm   = .~terminals
    shapes    = UniformHole.(partition(Hole(nonterm), grammar), ([],))

    best = Inf
    D = size(iter.logp_by_depth, 2)

    # depth of the operator we’re about to place (root expansion)
    d_op = 1
    d_child0 = d_op + 1

    for shape in shapes
        # Find any op to get its child types
        rule_idx = findfirst(shape.domain)
        rule_idx === nothing && continue
        child_types = Tuple(grammar.childtypes[rule_idx])

        # Gather candidate child entries per required type
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

        # DEPTH-AWARE operator lower bound at depth d_op
        op_min = Inf
        @inbounds for ridx in findall(shape.domain)
            # only consider operator rules (nonterm); shape.domain should already enforce that
            d = d_op ≤ D ? d_op : D
            c = iter.logp_by_depth[ridx, d]
            if c < op_min; op_min = c; end
        end
        isfinite(op_min) || continue

        # Combine children; require at least one new entry
        for tuple_children in Iterators.product(candidate_lists...)
            any_new = any(e.new_shape for e in tuple_children)
            any_new || continue

            # Children LB at their start depth (d_child0)
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
    axes = Axis[]

    if isempty(hole.children)
        term_inds  = findall(hole.domain)
        term_costs = Float64[iter.logp_by_depth[ridx, depth] for ridx in term_inds]
        push!(axes, Axis(path, term_inds, term_costs))
        return axes
    end

    op_inds  = findall(hole.domain)
    op_costs = Float64[iter.logp_by_depth[ridx, depth] for ridx in op_inds]
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
end