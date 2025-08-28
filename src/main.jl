include("parseKarel.jl")
include("grammar.jl")
include("hysynth.jl")
include("utils.jl")
include("depth_based_synthesis.jl")

using .ParseKarel
using .Grammar
using .Hysynth
using .Depth_synthesis
using .Utils
using HerbSearch, HerbCore, HerbGrammar, HerbSpecification

using HerbBenchmarks
using HerbBenchmarks.Robots_2020

# Example LLM responses
llm_response_1 = "move turnleft move move"
llm_response_2 = "WHILE(markersPresent pickMarker move turnLeft putMarker)"
llm_response_3 = "WHILE(notAtBottom(), moveDown()); drop()"

# Parse trees array
parsed_trees = Vector{Union{Nothing, ParseTree}}(undef, 3)
parsed_trees[1] = parse_llm_response(llm_response_1)
# parsed_trees[2] = parse_llm_response(llm_response_2)
# parsed_trees[3] = parse_llm_response(llm_response_3)


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

grammar_karel =  @csgrammar begin
    Start = Block                                   #1

    Block = Action                                  #2
    Block = (Action; Block)                         #3
    Block = ControlFlow                             #4

    Action = move                                   #5
    Action = turnLeft                               #6
    Action = turnRight                              #7
    Action = pickMarker                             #8
    Action = putMarker                              #9

    ControlFlow = IF(Condition, Block)              #10
    ControlFlow = IFELSE(Condition, Block, Block)   #11
    ControlFlow = WHILE(Condition, Block)           #12
    ControlFlow = REPEAT(R=INT, Block)              #13
    INT = |(1:5)                                    #14-18

    Condition = frontIsClear                        #19
    Condition = leftIsClear                         #20
    Condition = rightIsClear                        #21
    Condition = markersPresent                      #22
    Condition = noMarkersPresent                    #23
    Condition = NOT(Condition)                      #24
end

# Loop through each parsed tree
for tree in parsed_trees
    if tree !== nothing
        rule_counts, counts_by_depth = Utils.print_tree(tree)
        println(rule_counts)
        println(counts_by_depth)

        # Build and fill the numeric matrix (6 rows; all grammar rules as columns)
        M, rules = Utils.counts_matrix(counts_by_depth, grammar_karel; nrows=6)

        # Print the whole matrix
        println("\nMatrix M (size = ", size(M), "):")
        println(M)
        C = Grammar.frequencies_to_costs(M, rules; alpha=1.0, eps=1e-3)
        println("Cost matrix size: ", size(C))

        # Pretty table
        Utils.pretty_print_counts(C)

        # dict = construct_dict(rule_counts)
        #println(dict)
        # pcsg = make_pcsg_from_dict(grammar_karel, dict)
        
        # print_pcsg_rules(pcsg)
        #run_synthesis_tests(pcsg)
        #println(grammar_robots)
        println("TEST")
        #run_hysynth_tests()
        # run_depth_synthesis_tests()
        run_depth_synthesis(C, grammar_karel)
        # run_priority_robot_test(pcsg)
        exit()
    else
        println("Failed to parse LLM response.")
    end
end

