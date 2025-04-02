module Utils
export read_response, print_tree
import ..Parse: ParseTree

function read_response(path::String)::String
    isfile(path) || error("Response file not found: $path")
    return read(path, String)
end

function print_tree(tree::ParseTree, rule_counts=Dict{String, Int}(), prefix="", is_last=true)
    # Count occurrences of rules
    rule_counts[tree.rule] = get(rule_counts, tree.rule, 0) + 1

    # Define tree branch symbols
    connector = is_last ? "└── " : "├── "

    # Print the current node with the appropriate prefix
    println(prefix * connector * tree.rule * " (x$(rule_counts[tree.rule]))")

    # Update prefix for children
    new_prefix = prefix * (is_last ? "    " : "│   ")

    # Recursively print children
    for (i, child) in enumerate(tree.children)
        print_tree(child, rule_counts, new_prefix, i == length(tree.children))
    end

    return rule_counts  # Return rule counts for further analysis
end

end # module