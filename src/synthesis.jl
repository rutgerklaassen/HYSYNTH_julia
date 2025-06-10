module Synthesis
export run_synthesis_tests, run_synthesis, run_priority_robot_test

using HerbGrammar, HerbSearch, HerbCore, HerbConstraints, HerbSpecification, HerbBenchmarks, Test
using DataStructures: DefaultDict
using Printf
import ..Grammar
import HerbSearch: get_bank


@programiterator mutable MyBU(bank) <: BottomUpIterator

function HerbSearch.init_combine_structure(iter::MyBU)
    Dict(:max_cost_in_bank => 20)
end

const COST_PRECISION = 1e-6
round_cost(x::Real) = round(x; digits=4)  # enforce consistent dictionary keys

function HerbSearch.create_bank!(iter::MyBU)
    iter.bank = DefaultDict{Float64, Vector{AbstractRuleNode}}(() -> AbstractRuleNode[])
end

function HerbSearch.populate_bank!(iter::BottomUpIterator)::AbstractVector{AccessAddress}
    grammar = HerbSearch.get_grammar(iter.solver)
    # Create the terminal programs
    terminal_programs = UniformHole(grammar.isterminal, [])

    # Initialize bank entries for each terminal according to its log_prob
    for (i, is_terminal) in enumerate(grammar.isterminal)
        if is_terminal
            logprob = grammar.log_probabilities[i]
            println(logprob)
            index = round_cost(logprob)  # safe index
    
            # Create BitVector with single true
            bv = falses(length(grammar.isterminal))
            bv[i] = true
            program = UniformHole(bv, [])  # This is the correct UniformHole
    
            if !(haskey(get_bank(iter), index))
                get_bank(iter)[index] = Vector{AbstractRuleNode}()
            end
    
            push!(get_bank(iter)[index], program)
        end
    end
    

    # Add all terminals to the solver state
    new_state!(iter.solver, terminal_programs)

    # Return AccessAddress for all terminals in the bank (flattened)
    return [AccessAddress((idx, x)) for (idx, bucket) in get_bank(iter) for x in 1:length(bucket)]
end


function HerbSearch.new_address(iter::MyBU, program_combination::AbstractAddress)::AbstractAddress
    # println("type of program_combination: ", typeof(program_combination))
    # println("type of iter: ", typeof(iter))
    # println(program_combination.op.domain)
    grammar = HerbSearch.get_grammar(iter.solver)
    rule_cost = grammar.log_probabilities[findfirst(program_combination.op.domain)]
    child_costs = sum(x.addr[1] for x in program_combination.addrs)
    total_cost = round_cost(rule_cost + child_costs)
    return AccessAddress((total_cost, 1))  # The second element can be 1 if you don’t care about indexing
end

function HerbSearch.add_to_bank!(iter::MyBU, program::AbstractRuleNode, address::AccessAddress)::Bool
    cost = round_cost(address.addr[1])
    push!(get_bank(iter)[cost], program)
    return true  # You can insert observational equivalence checks later
end

function HerbSearch.combine(iter::MyBU, state)
    addresses = Vector{CombineAddress}()
    bank = get_bank(iter)
    max_cost_in_bank = isempty(bank) ? 0.0 : maximum(keys(bank))
    grammar = HerbSearch.get_grammar(iter.solver)
    terminals = grammar.isterminal
    nonterminals = .~terminals
    non_terminal_shapes = UniformHole.(partition(Hole(nonterminals), grammar), ([],))

    # if we have exceeded the maximum number of programs to generate
    if max_cost_in_bank >= state[:max_cost_in_bank]
        return nothing, nothing
    end

    #check bound function TODO: This definetly needs fixing but not right now
    # function check_bound(combination)
    #     return sum((x[1] for x in combination)) > max_cost_in_bank
    # end

    function estimate_cost(grammar, addr::CombineAddress)
        child_costs = sum(x.addr[1] for x in addr.addrs)
        rule_cost = grammar.log_probabilities[findfirst(addr.op.domain)]
        return round_cost(child_costs + rule_cost)
    end

    all_combinations = CombineAddress[]
    for shape in non_terminal_shapes
        nchildren = length(grammar.childtypes[findfirst(shape.domain)])

        # Create address pairs for child programs in bank
        all_addresses = ((key, idx) for key in keys(bank) for idx in eachindex(bank[key]))
        combinations = Iterators.product(Iterators.repeated(all_addresses, nchildren)...)

        # Wrap into CombineAddress structs
        combine_addrs = map(address_pair -> CombineAddress(shape, AccessAddress.(address_pair)), combinations)

        # Filter based on cost
        for addr in combine_addrs
            if estimate_cost(grammar, addr) <= state[:max_cost_in_bank]
                push!(all_combinations, addr)
            end
        end
    end

    # Global sort across all rule shapes
    sorted_combinations = sort(all_combinations; by = addr -> estimate_cost(grammar, addr))
    # println("\n📦 Top 100 CombineAddresses by estimated cost:")
    # for (i, addr) in enumerate(sorted_combinations[1:min(100, end)])
    #     est_cost = estimate_cost(grammar, addr)
    #     println("[$i] Cost = $est_cost │ Shape = $(addr.op.domain) │ Child costs = $([x.addr[1] for x in addr.addrs])")
    # end    
    return sorted_combinations, state
end


function run_synthesis_tests(grammar)
    # @testset "Bottom Up Search" begin
    #     iter = MyBU(grammar, :Start, nothing; max_depth=5)
    #     create_bank!(iter)
    #     populate_bank!(iter)

    #     combinations, state = combine(iter, init_combine_structure(iter))
    #     @test !isempty(combinations)
    #     #println("Combinations from combine: ", combinations)
    # end

    

    @testset "Cost correctness and bank population" begin
        g = @csgrammar begin
            Int = Int + Int   # Rule 1
            Int = 1
        end
    
        rule_counts = Dict(
            "Int->Int+Int" => 1,  # high cost
            "Int->1" => 100       # low cost
        )
    
        pcfg = Grammar.make_pcsg_from_dict(g, Grammar.construct_dict(rule_counts))
        iter = MyBU(pcfg, :Int, nothing; )
    
        # Test 1: Bank population
        create_bank!(iter)
        terms = populate_bank!(iter)
    
        bank = get_bank(iter)
        @test haskey(bank, round_cost(0.0))  # terminals should be in cost bucket 0.0
        @test length(bank[round_cost(0.0)]) > 0
    
        # Test 2: Manual cost check for CombineAddress
        grammar = HerbSearch.get_grammar(iter.solver)
        addr = CombineAddress(
            UniformHole(BitVector([true, false]), []),
            (AccessAddress((0.0, 1)), AccessAddress((0.0, 1)))
        )
        rule_cost = grammar.log_probabilities[1]
        expected_cost = round_cost(rule_cost + 0.0 + 0.0)
        computed_cost = round_cost(sum(x.addr[1] for x in addr.addrs) + rule_cost)
        @test computed_cost == expected_cost
    
        # Test 3: Check rule prioritization via ascending cost
        combinations, _ = combine(iter, init_combine_structure(iter))
        @test length(combinations) > 0
        costs = [sum(x.addr[1] for x in ca.addrs) + grammar.log_probabilities[findfirst(ca.op.domain)] for ca in combinations]
        @test issorted(costs)
    
        # Test 4: new_address agrees with manual cost
        auto_addr = new_address(iter, addr)
        @test auto_addr.addr[1] == expected_cost
    
        # Test 5: combine never exceeds max cost
        max_allowed_cost = init_combine_structure(iter)[:max_cost_in_bank]
        for ca in combinations
            total_cost = sum(x.addr[1] for x in ca.addrs) + grammar.log_probabilities[findfirst(ca.op.domain)]
            @test total_cost <= max_allowed_cost
        end
    
        # Test 6: only terminals in cost 0.0 bucket
        terminals_only = bank[round_cost(0.0)]
        for prog in terminals_only
            expr = rulenode2expr(prog, grammar)
            @test occursin("1", string(expr))  # or more generally, match known terminals
        end
    
        # Test 7: cost order of yielded programs
        costs_seen = Float64[]
        for prog in iter
            # Match against bank buckets to recover cost
            for (cost, progs) in bank
                if prog in progs
                    push!(costs_seen, cost)
                    break
                end
            end
        end
        @test issorted(costs_seen)
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
        if(i < 2)
            println("Running problem number $i.")
            iterator = MyBU(grammar, :Start, nothing; max_depth=10)
            solution = my_synth(pair.problem, iterator)

            if !isnothing(solution)
                @show "Solution: ", solution
                solved_problemos += 1
                #exit()
            end
        end
    end
end # End run synthesis function

function run_priority_robot_test()
    pairs = get_all_problem_grammar_pairs(Robots_2020)
    pair = pairs[1]  # Pick the first task
    base_grammar = pair.grammar

    # RULE STRINGS — you can adjust this if needed
    all_rules = [
        "Start->Sequence",
        "Sequence->Operation",
        "Sequence->(Operation;Sequence)",
        "Operation->Transformation",
        "Operation->ControlStatement",
        "Transformation->moveRight()",
        "Transformation->moveDown()",
        "Transformation->moveLeft()",
        "Transformation->moveUp()",
        "Transformation->drop()",
        "Transformation->grab()",
        "ControlStatement->IF(Condition,Sequence,Sequence)",
        "ControlStatement->WHILE(Condition,Sequence)",
        "Condition->atTop()",
        "Condition->atBottom()",
        "Condition->atLeft()",
        "Condition->atRight()",
        "Condition->notAtTop()",
        "Condition->notAtBottom()",
        "Condition->notAtLeft()",
        "Condition->notAtRight()"
    ]
    
    # GOOD: reward common low-level movements and correct structure
    good_counts = Dict(rule => (
        occursin("moveRight()", rule) || occursin("moveDown()", rule) || occursin("grab()", rule) || occursin("drop()", rule) ? 100 :
        occursin("Operation->Transformation", rule) ? 50 :
        1
    ) for rule in all_rules)
    # BAD: reward loops and irrelevant conditions
    bad_counts = Dict(rule => (
        occursin("moveLeft", rule) || occursin("WHILE", rule) || occursin("IF", rule) || occursin("notAtTop", rule) || occursin("notAtLeft", rule) ? 100 :
        occursin("Sequence", rule) || occursin("Operation", rule) ? 20 :
        1
    ) for rule in all_rules)

    println("\n🟢 Running with GOOD guidance...\n")
    good_pcfg = Grammar.make_pcsg_from_dict(base_grammar, Grammar.construct_dict(good_counts))
    #run_synthesis(good_pcfg)
    #@show good_pcfg.log_probabilities

    println("\n🔴 Running with BAD guidance...\n")
    bad_pcfg = Grammar.make_pcsg_from_dict(base_grammar, Grammar.construct_dict(bad_counts))
    println(bad_pcfg.log_probabilities)
    run_synthesis(bad_pcfg)
    #@show bad_pcfg.rules
    #@show bad_pcfg.log_probabilities

end

end # module

