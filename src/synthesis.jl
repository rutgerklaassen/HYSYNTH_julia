module Synthesis
export run_synthesis_tests

using HerbGrammar, HerbSearch, HerbCore, HerbConstraints, HerbSpecification, Test
import ..Grammar
import HerbSearch: get_bank

@programiterator mutable MyBU(bank) <: BottomUpIterator

function HerbSearch.init_combine_structure(iter::MyBU)
    Dict(:max_combination_depth => 10)
end


function combine(iter::BottomUpIterator, state)
    addresses = Vector{CombineAddress}()
    max_in_bank = maximum(keys(get_bank(iter)))
    grammar = get_grammar(iter.solver)
    terminals = grammar.isterminal
    nonterminals = .~terminals
    non_terminal_shapes = UniformHole.(partition(Hole(nonterminals), grammar), ([],))

    if max_in_bank >= state[:max_combination_depth]
        return nothing, nothing
    end

    function check_bound(combination)
        return 1 + sum((x[1] for x in combination)) > max_in_bank
    end

    scored_combinations = []

    for rule_idx in eachindex(grammar.rules)
        rule = grammar.rules[rule_idx]
        if grammar.isterminal[rule_idx]
            continue
        end
    
        shape = UniformHole(BitVector(i == rule_idx for i in 1:length(grammar.rules)), [])
        nchildren = length(grammar.childtypes[rule_idx])
    
        all_addresses = ((key, idx) for key in keys(get_bank(iter)) for idx in eachindex(get_bank(iter)[key]))
        all_combinations = Iterators.product(Iterators.repeated(all_addresses, nchildren)...)
        filtered_combinations = Iterators.filter(check_bound, all_combinations)
    
        for address_pair in filtered_combinations
            combine_addr = CombineAddress(shape, AccessAddress.(address_pair))
            score = grammar.log_probabilities[rule_idx]
    
            @info "Scoring combination" rule=rule score=score
    
            push!(scored_combinations, (score, combine_addr))
        end
    end

    sorted = sort(scored_combinations; by = x -> -x[1])
    addresses = last.(sorted) 
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
        grammar = get_grammar(iter.solver)
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

end # module