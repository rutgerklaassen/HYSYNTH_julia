module Utils
export read_response, print_tree, collect_counts_by_depth, counts_matrix, pretty_print_counts
import ..ParseKarel: ParseTree

function read_response(path::String)::String
    isfile(path) || error("Response file not found: $path")
    return read(path, String)
end

"""
    print_tree(tree; rule_counts=Dict{String,Int}(), counts_by_depth=Dict{Int,Dict{String,Int}}(), prefix="", is_last=true, depth=1)

Print the tree and simultaneously accumulate:
  - total counts per rule in `rule_counts`
  - per-depth counts in `counts_by_depth`
Returns `(rule_counts, counts_by_depth)`.
"""
function print_tree(tree::ParseTree;
                    rule_counts=Dict{String, Int}(),
                    counts_by_depth=Dict{Int, Dict{String, Int}}(),
                    prefix::String="",
                    is_last::Bool=true,
                    depth::Int=1)

    # total counts
    rule_counts[tree.rule] = get(rule_counts, tree.rule, 0) + 1

    # per-depth counts
    row = get!(counts_by_depth, depth, Dict{String,Int}())
    row[tree.rule] = get(row, tree.rule, 0) + 1

    # Drawing
    connector = is_last ? "└── " : "├── "
    println(prefix * connector * tree.rule * " (x$(rule_counts[tree.rule]))")

    new_prefix = prefix * (is_last ? "    " : "│   ")
    for (i, child) in enumerate(tree.children)
        print_tree(child;
                   rule_counts=rule_counts,
                   counts_by_depth=counts_by_depth,
                   prefix=new_prefix,
                   is_last=(i == length(tree.children)),
                   depth=depth+1)
    end

    return rule_counts, counts_by_depth
end

"""
    collect_counts_by_depth(tree) -> Dict{Int, Dict{String,Int}}

If you don't want printing, just collect per-depth counts.
"""
function collect_counts_by_depth(tree::ParseTree)
    counts_by_depth = Dict{Int, Dict{String, Int}}()
    _walk_depth(tree, 1, counts_by_depth)
    return counts_by_depth
end

function _walk_depth(node::ParseTree, depth::Int, acc::Dict{Int,Dict{String,Int}})
    row = get!(acc, depth, Dict{String,Int}())
    row[node.rule] = get(row, node.rule, 0) + 1
    for ch in node.children
        _walk_depth(ch, depth+1, acc)
    end
end

"""
    counts_matrix(counts_by_depth) -> (M, depths, rules)

Build a depth × rule matrix `M` where M[i,j] is the frequency of `rules[j]` at `depths[i]`.
"""
function counts_matrix(counts_by_depth::Dict{Int, Dict{String, Int}})
    depths = sort(collect(keys(counts_by_depth)))
    # collect unique rules across all depths
    rules = sort!(collect(Set(Iterators.flatten(keys.(values(counts_by_depth))))))
    M = zeros(Int, length(depths), length(rules))
    for (i, d) in enumerate(depths)
        row = counts_by_depth[d]
        for (j, r) in enumerate(rules)
            M[i, j] = get(row, r, 0)
        end
    end
    return M, depths, rules
end

"""
    pretty_print_counts(M, depths, rules)

Nicely print the Rule × Depth table like in your example (depths as rows, rules as columns).
"""
function pretty_print_counts(M::AbstractMatrix, depths::Vector{Int}, rules::Vector{String})
    # simple spacing
    colw = maximum(length, rules)
    header = rpad("", 4) * join(rpad.(rules, colw+2), "")
    println(header)
    for (i, d) in enumerate(depths)
        rowvals = [lpad(string(M[i,j]), colw) * "  " for j in eachindex(rules)]
        println(lpad(string(d)*".", 4), join(rowvals, ""))
    end
end

end # module
