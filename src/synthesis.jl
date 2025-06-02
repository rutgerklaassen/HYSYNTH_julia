module Synthesis
export run_synthesis_tests, run_synthesis

using HerbGrammar, HerbSearch, HerbCore, HerbConstraints, HerbSpecification, HerbBenchmarks, Test
import ..Grammar
import HerbSearch: get_bank

@programiterator mutable MyBU(bank) <: BottomUpIterator

function HerbSearch.init_combine_structure(iter::MyBU)
    Dict(:max_combination_depth => 10)
end


# function HerbSearch.combine(iter::MyBU, state)
#     @info "Calling combine" current_state=state

#     addresses = Vector{CombineAddress}()
#     max_in_bank = maximum(keys(get_bank(iter)))
#     grammar = HerbSearch.get_grammar(iter.solver)
#     terminals = grammar.isterminal
#     nonterminals = .~terminals
#     non_terminal_shapes = UniformHole.(partition(Hole(nonterminals), grammar), ([],))

#     if max_in_bank >= state[:max_combination_depth]
#         return nothing, nothing
#     end

#     function check_bound(combination)
#         return 1 + sum((x[1] for x in combination)) > max_in_bank
#     end

#     scored_combinations = []

#     for rule_idx in eachindex(grammar.rules)
#         rule = grammar.rules[rule_idx]
#         if grammar.isterminal[rule_idx]
#             continue
#         end
#         shape = UniformHole(BitVector(i == rule_idx for i in 1:length(grammar.rules)), [])
#         nchildren = length(grammar.childtypes[rule_idx])
    
#         all_addresses = ((key, idx) for key in keys(get_bank(iter)) for idx in eachindex(get_bank(iter)[key]))
#         all_combinations = Iterators.product(Iterators.repeated(all_addresses, nchildren)...)
#         filtered_combinations = Iterators.filter(check_bound, all_combinations)
    
#         for address_pair in filtered_combinations
#             combine_addr = CombineAddress(shape, AccessAddress.(address_pair))
#             score = grammar.log_probabilities[rule_idx]
    
#             @info "Scoring combination" rule=rule score=score
    
#             push!(scored_combinations, (score, combine_addr))
#         end

#         #print_bank(iter)
#         #input("Press Enter to continue...")

#     end

#     sorted = sort(scored_combinations; by = x -> -x[1])
#     addresses = last.(sorted) 
#     return addresses, state
# end


function HerbSearch.new_address(iter::MyBU, program_combination::AbstractAddress, program_type::Symbol)::AbstractAddress
    println("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
    println("type of program_combination: ", typeof(program_combination))
    total_cost = sum(x.addr[1] for x in program_combination.addrs) + iter.rule_costs[program_combination.op]

    if program_combination isa CombineAddress
        total_cost = sum(x.addr[1] for x in program_combination.addrs) + iter.rule_costs[program_combination.op]
        return AccessAddress((total_cost, program_type, 1))
    else
        error("new_address expected a CombineAddress but got ", typeof(program_combination))
    end
end


function HerbSearch.combine(iter::MyBU, state)
    println(state)
    addresses = Vector{CombineAddress}()
    max_in_bank = maximum(keys(get_bank(iter)))
    grammar = HerbSearch.get_grammar(iter.solver)
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


function run_synthesis_tests(grammar)
    @testset "Bottom Up Search" begin
        iter = MyBU(grammar, :Start, nothing; max_depth=5)
        create_bank!(iter)
        populate_bank!(iter)

        combinations, state = combine(iter, init_combine_structure(iter))
        @test !isempty(combinations)
        #println("Combinations from combine: ", combinations)
    end

    @testset "PCFG prioritization in combine" begin
        g = @csgrammar begin
            Int = Int + Int   # Rule 1
            Int = Int - Int   # Rule 2
            Int = Int * Int   # Rule 3
            Int = 1
            Int = 2
        end

        rule_counts = Dict(
            "Int->Int+Int" => 10,   # Should be second
            "Int->Int-Int" => 1,    # Should be last
            "Int->Int*Int" => 100,  # Should be chosen FIRST
            "Int->1" => 1,
            "Int->2" => 1
        )

        pcfg = Grammar.make_pcsg_from_dict(g, Grammar.construct_dict(rule_counts))
        iter = MyBU(pcfg, :Int, nothing; max_depth=3)

        create_bank!(iter)
        populate_bank!(iter)
        combinations, _ = combine(iter, init_combine_structure(iter))

        @test !isempty(combinations)

        top_n = min(10, length(combinations))
        rule_indices = [findfirst(ca.op.domain) for ca in combinations[1:top_n]]
        grammar = HerbSearch.get_grammar(iter.solver)
        rule_names = [sprint(show, grammar.rules[i]) for i in rule_indices]

        @info "Top rule names (should reflect PCFG priority)" rule_names=rule_names

        idx_mul   = findfirst(r -> occursin("*", r), rule_names)
        idx_plus  = findfirst(r -> occursin("+", r), rule_names)
        idx_minus = findfirst(r -> occursin("-", r), rule_names)

        if any(isnothing.([idx_mul, idx_plus, idx_minus]))
            @warn "Not all expected rules were found in the top combinations" missing=[idx_mul, idx_plus, idx_minus]
        else
            @test idx_mul < idx_plus < idx_minus
        end
    end
end


function my_synth(problem::Problem, iterator::ProgramIterator)
    for (i, program) ∈ enumerate(iterator)
        expr = rulenode2expr(program, HerbSearch.get_grammar(iterator.solver))
        println("[$i] Trying program: $expr")
    
        falsified = false
        for ex in problem.spec
            try
                output = Robots_2020.interpret(program, HerbSearch.get_grammar(iterator.solver), ex)

                if output != ex.out
                    falsified = true
                    break
                end
            catch exception
                falsified = true
                break
            end
        end

        if !falsified
            return rulenode2expr(program, HerbSearch.get_grammar(iterator.solver))
        end 
    end
    return nothing
end

function run_synthesis(grammar)
    pairs = get_all_problem_grammar_pairs(Robots_2020)

    solved_problemos = 0
    for (i, pair) in enumerate(pairs)
        println("Running problem number $i.")
        iterator = MyBU(grammar, :Start, nothing; max_depth=4)
        solution = my_synth(pair.problem, iterator)

        if !isnothing(solution)
            @show "Solution: ", solution
            solved_problemos += 1
            exit()
        end
    end
end # End run synthesis function

end # module
