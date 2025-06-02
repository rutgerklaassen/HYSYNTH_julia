module Synthesis
export run_synthesis_tests, run_synthesis

using HerbGrammar, HerbSearch, HerbCore, HerbConstraints, HerbSpecification, HerbBenchmarks, Test
using DataStructures: DefaultDict
import ..Grammar
import HerbSearch: get_bank

@programiterator mutable MyBU(bank) <: BottomUpIterator

function HerbSearch.init_combine_structure(iter::MyBU)
    Dict(:max_cost_in_bank => 10)
end

const COST_PRECISION = 1e-6
round_cost(x::Real) = round(x; digits=6)  # enforce consistent dictionary keys

function HerbSearch.create_bank!(iter::MyBU)
    iter.bank = DefaultDict{Float64, Vector{AbstractRuleNode}}(() -> AbstractRuleNode[])
end

function HerbSearch.populate_bank!(iter::MyBU)::Vector{AccessAddress}
    grammar = HerbSearch.get_grammar(iter.solver)
    terminal_program = UniformHole(grammar.isterminal, [])

    # Terminal programs go in cost-0.0 bucket
    cost_bucket = round_cost(0.0)
    get_bank(iter)[cost_bucket] = [terminal_program]
    new_state!(iter.solver, terminal_program)

    return [AccessAddress((cost_bucket, 1))]
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

    #check bound function TODO: Check if you can remove
    # function check_bound(combination)
    #     return sum((x[1] for x in combination)) < max_cost_in_bank
    # end

    function estimate_cost(grammar, addr::CombineAddress)
        child_costs = sum(x.addr[1] for x in addr.addrs)
        rule_cost = grammar.log_probabilities[findfirst(addr.op.domain)]
        return round_cost(child_costs + rule_cost)
    end

    # loop over groups of rules with the same arity and child types
    for shape in non_terminal_shapes
        nchildren = length(grammar.childtypes[findfirst(shape.domain)])

        # *Lazily* collect addresses, their combinations, and then filter them based on `check_bound`
        all_addresses = ((key, idx) for key in keys(get_bank(iter)) for idx in eachindex(get_bank(iter)[key]))
        all_combinations = Iterators.product(Iterators.repeated(all_addresses, nchildren)...)
        #filtered_combinations = Iterators.filter(check_bound, all_combinations)

        # Construct the `CombineAddress`s from the filtered combinations and sort it
        #combine_addrs = map(address_pair -> CombineAddress(shape, AccessAddress.(address_pair)), filtered_combinations)

        combine_addrs = map(address_pair -> CombineAddress(shape, AccessAddress.(address_pair)), all_combinations)
        filtered_addrs = Iterators.filter(addr -> estimate_cost(grammar, addr) <= state[:max_cost_in_bank], combine_addrs)

        sorted_addrs = sort(combine_addrs; by = addr -> estimate_cost(grammar, addr))
        append!(addresses, sorted_addrs)
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
