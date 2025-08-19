module Utils
export read_response, print_tree, collect_counts_by_depth, counts_matrix, pretty_print_counts, make_rule_matrix
import ..ParseKarel: ParseTree
using HerbGrammar
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

function make_rule_matrix(grammar; nrows::Int = 6)
    # rule_types[i] is the LHS for rules[i]
    rules = [string(lhs, "->", rhs)
             for (lhs, rhs) in zip(grammar.types, grammar.rules)]
    rules[3] = "Block->(Action; Block)"
    println(rules)

    rule_index = Dict(r => i for (i, r) in enumerate(rules))
    M = zeros(Int, nrows, length(rules))

    return M, rules, rule_index
end


"""
    counts_matrix(counts_by_depth, grammar; nrows=6) -> (M, depths, rules)

Build a depth × rule matrix `M` where M[i,j] is the frequency of `rules[j]` at depth i.
Rows are fixed to depths 1..nrows.
Unknown rule strings in `counts_by_depth` (not present in `grammar`) are ignored.
"""
function counts_matrix(counts_by_depth::Dict{Int, Dict{String, Int}}, grammar; nrows::Int = 6)
    M, rules, rule_index = make_rule_matrix(grammar; nrows=nrows)
    println(counts_by_depth)
    println(rule_index)
    for (depth, d) in counts_by_depth
        1 <= depth <= nrows || continue
        println(depth, d)
        for (rule_str, cnt) in d
            j = get(rule_index, rule_str, 0)
            j == 0 && continue           # skip rules not in the grammar
            M[depth, j] += cnt
        end
    end

    depths = collect(1:nrows)
    return M, depths, rules
end
"""
    pretty_print_counts(M, depths, rules)

Nicely print the Rule × Depth table like in your example (depths as rows, rules as columns).
"""
function pretty_print_counts(M::AbstractMatrix, depths::Vector{Int})
    ncols = size(M, 2)
    # header with column numbers
    header = rpad("", 4) * join([lpad(string(j), 4) for j in 1:ncols], " ")
    println(header)
    # each row
    for (i, d) in enumerate(depths)
        rowvals = [lpad(string(M[i,j]), 4) for j in 1:ncols]
        println(lpad(string(d)*".", 4), " ", join(rowvals, " "))
    end
end

end # module
