module Synthesis
export run_synthesis_tests, run_synthesis, run_priority_robot_test

using HerbGrammar, HerbSearch, HerbCore, HerbConstraints, HerbSpecification, HerbBenchmarks, Test
using DataStructures: DefaultDict
using Printf
using HerbBenchmarks.Karel_2018

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

function print_bank_with_costs(iter::BottomUpIterator)
    G = HerbSearch.get_grammar(iter.solver)
    bank = get_bank(iter)

    for cost in keys(bank)  # iterate in the bank's native order
        println("\n=== Cost bucket: $cost ===")
        for (t, vec) in bank[cost]
            for (i, prog) in enumerate(vec)
                expr = rulenode2expr(prog, G)
                println("[$cost][$t][$i] $expr")
            end
        end
        println("\n")
    end
end

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
    # print_bank_with_costs(iter)
    bank = get_bank(iter)
    limit = iter.current_max_cost            # e.g., 2 at the start
    addrs = AccessAddress[]
    for index in sort!(collect(keys(bank)))  # ascending cost for stable behavior
        bucket = bank[index]
        for (t, vec) in bucket
            for x in eachindex(vec)
                push!(addrs, AccessAddress((index, t, x)))
            end
        end
    end
    return addrs
    # return [AccessAddress((index, t, x))
    #         for (index, bucket) in get_bank(iter)
    #         for (t, vec) in bucket
    #         for x in 1:length(vec)]
end


function HerbSearch.new_address(iter::MyBU, program_combination::CombineAddress, program_type::Symbol, idx)::AbstractAddress
    grammar = HerbSearch.get_grammar(iter.solver)
    rule_cost = grammar.log_probabilities[findfirst(program_combination.op.domain)]
    child_costs = sum(x.addr[1] for x in program_combination.addrs)
    total_cost = round_cost(rule_cost + child_costs)

    return AccessAddress((total_cost, program_type, idx))
end

# function HerbSearch.add_to_bank!(iter::MyBU, program_combination::AbstractAddress, program::AbstractRuleNode, program_type::Symbol)::Bool
#     bank = get_bank(iter)
#     prog_cost = 1 + maximum([x.addr[1] for x in program_combination.addrs])
#     n_in_bank = length(bank[prog_cost][program_type])
#     address = new_address(iter, program_combination, program_type, n_in_bank + 1)

#     push!(get_bank(iter)[address.addr[1]][address.addr[2]], program)

#     return true
# end

function HerbSearch.add_to_bank!(iter::MyBU, program_combination::AbstractAddress, program::AbstractRuleNode, program_type::Symbol)::Bool
    grammar = HerbSearch.get_grammar(iter.solver)
    bank = get_bank(iter)
    rule_idx = findfirst(program_combination.op.domain) 
    rule_cost = grammar.log_probabilities[rule_idx]

    prog_cost = round_cost(rule_cost + sum(x.addr[1] for x in program_combination.addrs))
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
    non_terminal_shapes = [let v = falses(length(dom)); v[j] = true; UniformHole(v, []) end
                        for dom in partition(Hole(nonterminals), grammar)
                        for j in findall(dom)] 
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
        return UniformHole(copy(addr.op.domain), children)
    end

    function check_seen_programs(iter::BottomUpIterator, addr::CombineAddress)
        for program in iter.seen_programs
            if HerbConstraints.pattern_match(program, reconstruct_program(iter, addr)) == HerbConstraints.PatternMatchSuccess()
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
                    push!(iter.seen_programs, deepcopy(prog))
                    # check_seen_programs(iter, addr)
                    # println(iter.seen_programs)
                    push!(scored_combinations, (addr, total_cost))
                end
            end
        end
    end

    # Sort based on precomputed total_cost
    sorted_combinations = sort(scored_combinations; by = x -> x[2])
    final = first.(sorted_combinations)  # Extract only CombineAddress objects

    if isempty(final)
        state[:current_max_cost] += 1
        return combine(iter, state)
    end
    
    state[:current_max_cost] += 1
    return final, state
end

function run_synthesis_tests()
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
            Int = Int - Int
            Int = Int * Int
            Int = 1
            Int = 2
        end
    
        rule_counts = Dict(
            "Int->Int+Int" => 1,  # very high cost 3.8
            "Int->Int-Int" => 10,  # low cost 1.3
            "Int->Int*Int" => 1, #very high cost 3.8
            "Int->1" => 1,       # very high cost 3.8
            "Int->2" => 10,       # low cost 1.3
        )
        # Real[3.807354922057604, 1.3479233034203069, 3.807354922057604, 3.807354922057604, 1.3479233034203069]

        # Order of programs should be :
        expected_programs  = [
        "2", # 2
        "1", # 4
        "2-2",#6
        "2+2", # 8
        "2-1", # 8
        "1-2", # 8
        "2*2", # 8
        "2+1", # 10
        "1+2", # 10
        "2-2-2", # 10
        "1-1", # 10
        "2-2-2", # 10
        "2*1", # 10
        "1*2", # 10
        "2+2-2", #12
        "1+1", # 12
        "2-2+2", # 12
        "2-2+2", # 12
        "2-2-1", # 12
        "2-1-2", # 12
        "1-2-2", # 12
        "2-2-1", # 12
        "2+2-2", # 12
        "2-1-2", # 12
        "1-2-2", # 12 
        "2-2*2", # 12
        "2*2-2", # 12
        "2*2-2", # 12
        "1*1", # 12
        "2-2*2" # 12    
        ]

        pcfg = Grammar.make_pcsg_from_dict(g, Grammar.construct_dict(rule_counts))
        println(pcfg.log_probabilities)
        iter = MyBU(pcfg, :Int; max_cost_in_bank = 15 )
    
        programs = []
        let G = HerbSearch.get_grammar(iter.solver)
            for (i, prog) in enumerate(iter)
                if i == 1
                    @test replace(string(rulenode2expr(prog, G)), r"[\s\(\)]" => "") == "2"
                end
                if i == 2
                    @test replace(string(rulenode2expr(prog, G)), r"[\s\(\)]" => "") == "1"
                end
                if i == 3 
                    @test replace(string(rulenode2expr(prog, G)), r"[\s\(\)]" => "") == "2-2"
                end
                if i > 3 && i < 8
                    @test replace(string(rulenode2expr(prog, G)), r"[\s\(\)]" => "") in expected_programs[4:7]
                end
                if i > 7 && i < 15
                    @test replace(string(rulenode2expr(prog, G)), r"[\s\(\)]" => "") in expected_programs[8:14]
                end
                if i > 14 && i < 30
                    @test replace(string(rulenode2expr(prog, G)), r"[\s\(\)]" => "") in expected_programs[15:29]
                end
                # if in 7:13
                #     @test string(rulenode2expr(prog, G)) in expected_programs[7:13]
                # end
                push!(programs, string(rulenode2expr(prog, G)))
            end
        end
    end    
end


function my_synth(problem::Problem, iterator::ProgramIterator)
    for (i, program) ∈ enumerate(iterator)
        expr = rulenode2expr(program, HerbSearch.get_grammar(iterator.solver))
        println("[$i] Trying program: $expr")
        falsified = false
        for ex in problem.spec
            println("EX: ",ex)
            output = Robots_2020.interpret(program, HerbSearch.get_grammar(iterator.solver), ex)
            println("OUTPUT: ", output)
            exit()

            if output != ex.out
                falsified = true
                break
            end
            # catch exception
            #     print("EXCEPTION!!", exception)
            #     falsified = true
            #     break
            # end
        end

        if !falsified
            return rulenode2expr(program, HerbSearch.get_grammar(iterator.solver))
        end 
    end
    return nothing
end

# function run_synthesis(grammar)
#     problems = Karel_2018.get_all_problems()
#     solved_problemos = 0
#     println(problems[1])
#     exit()
#     for (i, pair) in enumerate(pairs)
#         if(i < 2)
#             println("Running problem number $i.")
#             iterator = MyBU(grammar, :Start, max_cost_in_bank=15)
#             solution = my_synth(pair.problem, iterator)

#             if !isnothing(solution)
#                 @show "Solution: ", solution
#                 solved_problemos += 1
#             end
#         end
#     end
# end # End run synthesis function
function run_synthesis(pcfg)  # pcsg built from Karel grammar

    problems = Karel_2018.get_all_problems()  # ~10 IOExamples per problem:contentReference[oaicite:7]{index=7}
    prob = problems[1]  # try one first

    # run with your probabilistic bottom-up iterator
    prog, hits = solve_one(prob, pcfg, 10)  # increase 15 if you need a bigger search radius
    if isnothing(prog)
        println("No program found. Best coverage: $hits / $(length(prob.spec))")
    else
        println("Solved with $hits / $(length(prob.spec))")
        println("Program: ", rulenode2expr(prog, pcfg))
    end
end

function count_matches(prog::AbstractRuleNode, prob::Problem, grammar)
    hits = 0
    for ex in prob.spec
        # println(typeof(prog))
        # frozen = HerbConstraints.freeze_state(prog)
        # println(typeof(frozen), typeof(grammar), typeof(ex))
        # println(grammar_karel)
        out = Karel_2018.interpret(prog, grammar_karel, ex, Dict{Int,AbstractRuleNode}())  # Karel interpreter
        hits += (out == ex.out) ? 1 : 0
    end
    return hits
end

function solve_one(prob::Problem, grammar, max_cost::Int)
    it = MyBU(grammar, :Start; max_cost_in_bank = max_cost)
    best_prog = nothing
    best_hits = 0
    nexp = length(prob.spec)

    for (i, prog) in enumerate(it)
        println("Program: ", rulenode2expr(prog, grammar))
        println(prog)
        hits = count_matches(prog, prob, grammar)
        if hits > best_hits
            best_hits = hits
            best_prog = deepcopy(prog)

            println("\n NEWWWWW RECORDDDDDd: $hits / $nexp (program #$i)\n")
            println("Best program for now: ", rulenode2expr(prog, grammar))
            println("actual holes: ", prog)
            if hits == nexp
                break  # full solution
            end
        end
    end
    println("BEST PROG BEFORE RETURN:", best_prog)
    return best_prog, best_hits
end

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

