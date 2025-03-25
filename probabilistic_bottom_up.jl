using HerbSearch
using Test  # Import Julia's testing module
using HerbConstraints
using Herb

#@programiterator mutable ProbabilisticBU(bank) <: BottomUpIterator 
# This creates a Bottom-Up Iterator using HerbSearch
abstract type MyBottomUp <: BottomUpIterator end;

function combine(iter::MyBottomUp, state)
    addresses = Vector{CombineAddress}()
    max_in_bank = maximum(keys(get_bank(iter)))
    grammar = get_grammar(iter.solver)
    terminals = grammar.isterminal
    nonterminals = .~terminals
    non_terminal_shapes = UniformHole.(partition(Hole(nonterminals), grammar), ([],))

    # if we have exceeded the maximum number of programs to generate
    if max_in_bank >= state[:max_combination_depth]
        return nothing, nothing
    end

    #check bound function
    function check_bound(combination)
        return 1 + sum((x[1] for x in combination)) > max_in_bank
    end

    # loop over groups of rules with the same arity and child types
    for shape in non_terminal_shapes
        nchildren = length(grammar.childtypes[findfirst(shape.domain)])

        # *Lazily* collect addresses, their combinations, and then filter them based on `check_bound`
        all_addresses = ((key, idx) for key in keys(get_bank(iter)) for idx in eachindex(get_bank(iter)[key]))
        all_combinations = Iterators.product(Iterators.repeated(all_addresses, nchildren)...)
        filtered_combinations = Iterators.filter(check_bound, all_combinations)

        # Construct the `CombineAddress`s from the filtered combinations
        addresses = map(address_pair -> CombineAddress(shape, AccessAddress.(address_pair)), filtered_combinations)
    end

    return addresses, state
end
# Define a dummy pCFG with weighted rules
dummy_pcfg = Dict(
    "Sequence" => Dict("Operation Sequence" => 0.7, "Operation" => 0.3),
    "Operation" => Dict("Transformation" => 0.6, "ControlStatement" => 0.4),
    "Transformation" => Dict("moveRight()" => 0.5, "moveDown()" => 0.3, "drop()" => 0.2)
)

# Mock Grammar
grammar = @csgrammar begin
    Int = 1 | 2
    Int = 3 + Int
end

# Create a mock solver
mock_solver = HerbConstraints.UniformSolver(grammar, nothing)

# Create an instance of MyBottomUp with a mock bank
test_iter = MyBottomUp(nothing, mock_solver)
test_iter.pcfg = dummy_pcfg  # Store pCFG in iter

@testset "MyBottomUp Combine Function" begin
    # Mock state to test with
    mock_state = Dict(:max_combination_depth => 10)

    # Call overloaded combine function
    sorted_combinations, _ = HerbSearch.combine(test_iter, mock_state)

    # Extract rule names for easy checking
    extracted_rules = [string(addr) for addr in sorted_combinations]

    # Expected order based on probabilities (most likely first)
    expected_order = [
        "Sequence -> Operation Sequence",  # 0.7
        "Sequence -> Operation",           # 0.3
        "Operation -> Transformation",     # 0.6
        "Operation -> ControlStatement",   # 0.4
        "Transformation -> moveRight()",   # 0.5
        "Transformation -> moveDown()",    # 0.3
        "Transformation -> drop()"         # 0.2
    ]

    # Test that rules are ordered by probability
    @test extracted_rules == expected_order
end
# function HerbSearch.combine(iter::ProbabilisticBU, state)
#     addresses = Vector{CombineAddress}()
#     grammar = get_grammar(iter.solver)
    
#     # Get rule probabilities from the pCFG
#     rule_weights = iter.state[:pcfg]  # Use state instead of a direct field

#     nonterminals = .~grammar.isterminal
#     non_terminal_shapes = UniformHole.(partition(Hole(nonterminals), grammar), ([],))

#     function check_bound(combination)
#         return 1 + sum((x[1] for x in combination)) > state[:max_combination_depth]
#     end

#     for shape in non_terminal_shapes
#         nchildren = length(grammar.childtypes[findfirst(shape.domain)])
#         all_combinations = Iterators.product(Iterators.repeated(get_all_addresses(iter), nchildren)...)

#         # Sort rule combinations by probability from the pCFG
#         sorted_combinations = sort(all_combinations, by=x -> -get(rule_weights, string(x), 0.0))

#         addresses = map(combination -> CombineAddress(shape, AccessAddress.(combination)), sorted_combinations)
#     end

#     return addresses, state
# end


# # Define a dummy pCFG with weighted rules
# dummy_pcfg = Dict(
#     "Sequence" => Dict("Operation Sequence" => 0.7, "Operation" => 0.3),
#     "Operation" => Dict("Transformation" => 0.6, "ControlStatement" => 0.4),
#     "Transformation" => Dict("moveRight()" => 0.5, "moveDown()" => 0.3, "drop()" => 0.2)
# )

# # Create an instance of ProbabilisticBU
# test_iter = ProbabilisticBU(nothing, mock_solver)  # Bank & solver are handled automatically
# test_iter.state = Dict(:pcfg => dummy_pcfg)  # Store pCFG in state

# @testset "ProbabilisticBU Combine Function" begin
#     # Mock state to test with
#     mock_state = Dict(:max_combination_depth => 10)

#     # Call overloaded combine function
#     sorted_combinations, _ = HerbSearch.combine(test_iter, mock_state)

#     # Extract rule names for easy checking
#     extracted_rules = [string(addr) for addr in sorted_combinations]

#     # Expected order based on probabilities (most likely first)
#     expected_order = [
#         "Sequence -> Operation Sequence",  # 0.7
#         "Sequence -> Operation",           # 0.3
#         "Operation -> Transformation",     # 0.6
#         "Operation -> ControlStatement",   # 0.4
#         "Transformation -> moveRight()",   # 0.5
#         "Transformation -> moveDown()",    # 0.3
#         "Transformation -> drop()"         # 0.2
#     ]

#     # Test that rules are ordered by probability
#     @test extracted_rules == expected_order
# end

# # Run the test
# @testset "ProbabilisticBU Creation" begin
#     @test test_iter isa ProbabilisticBU  # Check if it's the correct type
#     @test test_iter.state[:pcfg] == dummy_pcfg  # Check if the pCFG is stored correctly
#     @test hasfield(ProbabilisticBU, :bank)  # Ensure bank field exists
#     @test hasfield(ProbabilisticBU, :solver)  # Ensure solver field exists
# end
