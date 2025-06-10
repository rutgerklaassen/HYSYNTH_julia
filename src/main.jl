include("parse.jl")
include("grammar.jl")
include("synthesis.jl")
include("utils.jl")

using .Parse
using .Grammar
using .Synthesis
using .Utils
using HerbSearch, HerbCore, HerbGrammar, HerbSpecification

using HerbBenchmarks
using HerbBenchmarks.Robots_2020

# Example LLM responses
llm_response_1 = "moveRight(); moveDown(); drop(); grab()"
llm_response_2 = "IF(atTop(), moveDown(), moveRight()); grab()"
llm_response_3 = "WHILE(notAtBottom(), moveDown()); drop()"

# Parse trees array
parsed_trees = Vector{Union{Nothing, ParseTree}}(undef, 3)
parsed_trees[1] = parse_llm_response(llm_response_1)
parsed_trees[2] = parse_llm_response(llm_response_2)
parsed_trees[3] = parse_llm_response(llm_response_3)


grammar_robots = @csgrammar begin
    Start = Sequence
    Sequence = Operation
    Sequence = (Operation; Sequence)
    Operation = Transformation
    Operation = ControlStatement
    Transformation = moveRight() | moveDown() | moveLeft() | moveUp() | drop() | grab()
    ControlStatement = IF(Condition, Sequence, Sequence)
    ControlStatement = WHILE(Condition, Sequence)
    Condition = atTop() | atBottom() | atLeft() | atRight() |
                notAtTop() | notAtBottom() | notAtLeft() | notAtRight()
end

# Loop through each parsed tree
for tree in parsed_trees
    if tree !== nothing
        rule_counts = print_tree(tree)
        #println(rule_counts)
        dict = construct_dict(rule_counts)
        #println(dict)
        pcsg = make_pcsg_from_dict(grammar_robots, dict)
        
        #print_pcsg_rules(pcsg)

        #run_synthesis_tests(pcsg)
        #println(grammar_robots)
        #run_synthesis(pcsg)

        run_priority_robot_test()
        exit()
    else
        println("Failed to parse LLM response.")
    end
end

