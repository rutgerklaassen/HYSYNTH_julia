module Synthesis
export run_synthesis_tests, run_synthesis, run_priority_robot_test

using HerbGrammar, HerbSearch, HerbCore, HerbConstraints, HerbSpecification, HerbBenchmarks, Test
using DataStructures: DefaultDict
using Printf
import ..Grammar
import HerbSearch: get_bank


@programiterator MyBU(bank=DefaultDict{Int,DefaultDict}(() -> (DefaultDict{Symbol,AbstractVector{AbstractRuleNode}}(() -> AbstractRuleNode[]))),
    max_cost_in_bank=0,
    current_max_cost = 2,
    seen_programs = Set{UniformHole}()  #  field to track structurally unique programs
) <: BottomUpIterator

function HerbSearch.init_combine_structure(iter::MyBU)
    return Dict(
        :max_cost_in_bank => iter.max_cost_in_bank,
        :current_max_cost => iter.current_max_cost
    )
end

const COST_PRECISION = 1e-6
round_cost(x::Real) = trunc(Int,ceil(x))  # enforce consistent dictionary keys


function HerbSearch.populate_bank!(iter::BottomUpIterator)::AbstractVector{AccessAddress}
    grammar = HerbSearch.get_grammar(iter.solver)
    for (i, is_terminal) in enumerate(grammar.isterminal) #loop over all terminals
        if is_terminal
            logprob = grammar.log_probabilities[i] # get log prob per termminal
            index = round_cost(logprob) # Make it the index
            program_type = grammar.types[i] # Make sure to sort by type

            bv = falses(length(grammar.isterminal))
            bv[i] = true # Create bitvector with "true"for the correct terminal rule used
            program = UniformHole(bv, []) # Make a program from it

            push!(get_bank(iter)[index][program_type], program) # Push to bank by it's log_prob and type
        end
    end

    return [AccessAddress((index, t, x))
            for (index, bucket) in get_bank(iter)
            for (t, vec) in bucket
            for x in 1:length(vec)]
end


function HerbSearch.new_address(iter::MyBU, program_combination::CombineAddress, program_type::Symbol, idx)::AbstractAddress
    grammar = HerbSearch.get_grammar(iter.solver)
    rule_cost = grammar.log_probabilities[findfirst(program_combination.op.domain)]
    child_costs = sum(x.addr[1] for x in program_combination.addrs)
    total_cost = round_cost(rule_cost + child_costs)

    return AccessAddress((total_cost, program_type, idx))
end

function HerbSearch.add_to_bank!(iter::MyBU, program_combination::AbstractAddress, program::AbstractRuleNode, program_type::Symbol)::Bool
    bank = get_bank(iter)
    prog_cost = 1 + maximum([x.addr[1] for x in program_combination.addrs])
    n_in_bank = length(bank[prog_cost][program_type])
    address = new_address(iter, program_combination, program_type, n_in_bank + 1)

    push!(get_bank(iter)[address.addr[1]][address.addr[2]], program)

    return true
end



function HerbSearch.combine(iter::MyBU, state)
    # @show state.starting_node
    # @show get_type(get_grammar(solver), state.starting_node)
    bank = get_bank(iter)
    max_cost_in_bank = isempty(bank) ? 0.0 : maximum(keys(bank))
    max_total_cost = state[:max_cost_in_bank]
    current_limit = state[:current_max_cost]

    grammar = HerbSearch.get_grammar(iter.solver)
    terminals = grammar.isterminal
    nonterminals = .~terminals
    non_terminal_shapes = UniformHole.(partition(Hole(nonterminals), grammar), ([],))
    println(current_limit)
    println(max_total_cost)
    if current_limit > max_total_cost
        return nothing, nothing
    end

    function appropriately_typed(child_types)
        return combination -> child_types == [x[2] for x in combination]
    end

    function reconstruct_program(iter::BottomUpIterator, addr::CombineAddress)::UniformHole
        children = [retrieve(iter, a) for a in addr.addrs]
        return UniformHole(addr.op.domain, children)
    end

    function check_seen_programs(iter::BottomUpIterator, addr::CombineAddress)
        for program in iter.seen_programs
            # println("Comparing :", program, " with : ", addr)
            if HerbConstraints.pattern_match(program, reconstruct_program(iter, addr)) == HerbConstraints.PatternMatchSuccess()
                # println("FOUND ONE!")
                return true
            end
        end
        return false
    end
    # One-pass tuple collection with precomputed cost
    scored_combinations = Tuple{CombineAddress, Int}[]

    for shape in non_terminal_shapes
        nchildren = length(grammar.childtypes[findfirst(shape.domain)])

        all_addresses = (
            (key, typename, idx)
            for key in keys(bank) if key <= current_limit
            for typename in keys(bank[key])
            for idx in eachindex(bank[key][typename])
        )

        combinations = Iterators.product(Iterators.repeated(all_addresses, nchildren)...)
        bounded_and_typed_combinations = Iterators.filter(appropriately_typed(grammar.childtypes[findfirst(shape.domain)]), combinations)
        combine_addrs = map(address_pair -> CombineAddress(shape, AccessAddress.(address_pair)), bounded_and_typed_combinations)

        for addr in combine_addrs

            child_costs = sum(x.addr[1] for x in addr.addrs)
            rule_cost = grammar.log_probabilities[findfirst(addr.op.domain)]
            total_cost = round_cost(child_costs + rule_cost)

            if total_cost <= current_limit && total_cost <= max_total_cost
                prog = reconstruct_program(iter, addr)

                if check_seen_programs(iter, addr)
                    continue
                else
                    # println("ADDED")
                    push!(iter.seen_programs, prog)
                    check_seen_programs(iter, addr)
                    # println(iter.seen_programs)
                    push!(scored_combinations, (addr, total_cost))
                end
            end
        end
    end

    # Sort based on precomputed total_cost
    sorted_combinations = sort(scored_combinations; by = x -> x[2], rev=true)
    final = first.(sorted_combinations)  # Extract only CombineAddress objects

    if isempty(final)
        state[:current_max_cost] += 1
        return combine(iter, state)
    end
    
    state[:current_max_cost] += 1
    return final, state
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
            iterator = MyBU(grammar, :Start, max_cost_in_bank=15)
            solution = my_synth(pair.problem, iterator)

            if !isnothing(solution)
                @show "Solution: ", solution
                solved_problemos += 1
            end
        end
    end
end # End run synthesis function

function run_priority_robot_test(pcsg)
    pairs = get_all_problem_grammar_pairs(Robots_2020)
    pair = pairs[1]  # Pick the first task

    println(typeof(base_grammar))
    exit()
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
        occursin("moveRight", rule) || occursin("WHILE", rule) || occursin("IF", rule) || occursin("notAtTop", rule) || occursin("notAtLeft", rule) ? 100 :
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

