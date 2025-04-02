include("parse.jl")
include("grammar.jl")
include("synthesis.jl")
include("utils.jl")

using .Parse
using .Grammar
using .Synthesis
using .Utils

# Example LLM responses
llm_response_1 = "moveRight(); moveDown(); drop(); grab()"
llm_response_2 = "IF(atTop(), moveDown(), moveRight()); grab()"
llm_response_3 = "WHILE(notAtBottom(), moveDown()); drop()"

# Parse trees array
parsed_trees = Vector{Union{Nothing, ParseTree}}(undef, 3)
parsed_trees[1] = parse_llm_response(llm_response_1)
parsed_trees[2] = parse_llm_response(llm_response_2)
parsed_trees[3] = parse_llm_response(llm_response_3)

# Loop through each parsed tree
for tree in parsed_trees
    if tree !== nothing
        rule_counts = print_tree(tree)
        dict = construct_dict(rule_counts)
        pcsg = make_pcsg_from_dict(grammar_robots, dict)
        #print_pcsg_rules(pcsg)
        run_synthesis_tests(pcsg)
    else
        println("Failed to parse LLM response.")
    end
end
