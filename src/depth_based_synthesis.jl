module Depth_synthesis

export run_depth_synthesis_tests, run_depth_synthesis, run_priority_robot_test

using HerbGrammar, HerbSearch, HerbCore, HerbConstraints, HerbSpecification, HerbBenchmarks, Test
using DataStructures: DefaultDict
using Printf
using HerbBenchmarks.Karel_2018


import ..Grammar
import HerbSearch: get_bank

@programiterator MyDepthBU(
    bank = DefaultDict{Int,DefaultDict}(() -> (DefaultDict{Symbol,AbstractVector{AbstractRuleNode}}(() -> AbstractRuleNode[]))),
    max_cost_in_bank = 0,
    current_max_cost = 2,
    seen_programs = Set{UniformHole}(),  
    costMatrix = Matrix{Float64}(undef, 0, 0)  
) <: BottomUpIterator

function HerbSearch.init_combine_structure(iter::MyDepthBU)
    return Dict(
        :max_cost_in_bank => iter.max_cost_in_bank,
        :current_max_cost => iter.current_max_cost
    )
end

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

function HerbSearch.populate_bank!(iter::MyDepthBU)::AbstractVector{AccessAddress}
    grammar = HerbSearch.get_grammar(iter.solver)
    COSTLESS_BUCKET = 0  # single bucket

    for (i, is_terminal) in enumerate(grammar.isterminal)
        if is_terminal
            program_type = grammar.types[i]
            bv = falses(length(grammar.isterminal))
            bv[i] = true
            program = UniformHole(bv, [])
            push!(get_bank(iter)[COSTLESS_BUCKET][program_type], program)
        end
    end

    addrs = AccessAddress[]
    for (t, vec) in get_bank(iter)[COSTLESS_BUCKET]
        for x in eachindex(vec)
            push!(addrs, AccessAddress((COSTLESS_BUCKET, t, x)))
        end
    end
    return addrs
end

function program_total_cost(iter::MyDepthBU, prog::UniformHole, depth::Int=1)
    d = clamp(depth, 1, size(iter.costMatrix, 1))
    j = findfirst(prog.domain)
    @assert j !== nothing "UniformHole has no active rule (domain is empty/invalid)"
    j = clamp(j, 1, size(iter.costMatrix, 2))
    self_cost = iter.costMatrix[d, j]
    return isempty(prog.children) ? self_cost :
        self_cost + sum(program_total_cost(iter, c, depth + 1) for c in prog.children)
end

# Same signature as your UniformHole version, but for StateHole
function program_total_cost(iter::MyDepthBU, prog::HerbSearch.StateHole, depth::Int=1)
    # Clamp depth to the cost matrix
    d = clamp(depth, 1, size(iter.costMatrix, 1))

    # Figure out which rule this node selects
    j = findfirst(prog.domain)
    @assert j !== nothing "StateHole has empty/invalid domain (no active rule)"
    j = clamp(j, 1, size(iter.costMatrix, 2))

    # Cost at this depth for this rule
    self_cost = iter.costMatrix[d, j]

    # Recurse into children (dispatch will pick the right method for each child)
    return isempty(prog.children) ? self_cost :
        self_cost + sum(program_total_cost(iter, c, depth + 1) for c in prog.children)
end


function HerbSearch.new_address(iter::MyDepthBU, program_combination::CombineAddress, program_type::Symbol, idx)::AbstractAddress
    return AccessAddress((0, program_type, idx))
end

# function HerbSearch.add_to_bank!(iter::MyDepthBU, program_combination::AbstractAddress, program::AbstractRuleNode, program_type::Symbol)::Bool
#     bank = get_bank(iter)
#     prog_cost = 1 + maximum([x.addr[1] for x in program_combination.addrs])
#     n_in_bank = length(bank[prog_cost][program_type])
#     address = new_address(iter, program_combination, program_type, n_in_bank + 1)

#     push!(get_bank(iter)[address.addr[1]][address.addr[2]], program)

#     return true
# end

function HerbSearch.add_to_bank!(iter::MyDepthBU, program_combination::AbstractAddress, program::AbstractRuleNode, program_type::Symbol)::Bool
    bank = get_bank(iter)
    # No cost computation; we store everything in the costless bucket.
    push!(bank[0][program_type], program)
    return true
end

function reconstruct_program(iter::BottomUpIterator, addr::CombineAddress)::UniformHole
    children = [retrieve(iter, a) for a in addr.addrs]
    return UniformHole(copy(addr.op.domain), children)
end

function HerbSearch.combine(iter::MyDepthBU, state)
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
    # println(current_limit)
    # println(max_total_cost)

    if current_limit > max_total_cost
        return nothing, nothing
    end

    function appropriately_typed(child_types)
        return combination -> child_types == [x[2] for x in combination]
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
            # Build the candidate program and compute full depth-aware cost
            prog = reconstruct_program(iter, addr)
            total_cost = round_cost(program_total_cost(iter, prog))

            if total_cost <= current_limit && total_cost <= max_total_cost
                if check_seen_programs(iter, addr)
                    continue
                else
                    push!(iter.seen_programs, deepcopy(prog))
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

function pretty_print_counts(M::AbstractMatrix)
    nrows, ncols = size(M)
    header = rpad("", 4) * join([lpad(string(j), 6) for j in 1:ncols], " ")
    println(header)
    for d in 1:nrows
        rowvals = [lpad(string(round(M[d, j], digits=2)), 6) for j in 1:ncols]
        println(lpad(string(d) * ".", 4), " ", join(rowvals, " "))
    end
end

function run_depth_synthesis_tests()

    @testset "populate_bank! costless bucket" begin
        # Tiny grammar with two terminals and one nonterminal rule
        G = @csgrammar begin
            Start = A           # nonterminal rule
            A = a()             # terminal
            A = b()             # terminal
        end

        # Make a cost matrix with 1 row (depth=1), cols = number of rules in G
        M = zeros(Float64, 1, length(G.rules))

        it = MyDepthBU(G, :Start; max_cost_in_bank = 10, costMatrix = M)
        addrs = HerbSearch.populate_bank!(it)               # call your new method

        bank = get_bank(it)

        # Only one bucket key (the costless one, 0)
        @test collect(keys(bank)) == [0]

        # Types present in that bucket match grammar.types of *terminal* rules
        term_ids = findall(G.isterminal)
        term_types = unique(G.types[term_ids])
        @test sort(collect(keys(bank[0]))) == sort(collect(term_types))

        # All entries are UniformHole terminals with the correct one-hot bit set
        total = 0
        for (t, vec) in bank[0]
            for prog in vec
                @test isa(prog, UniformHole)
                @test isempty(prog.children)
                @test count(identity, prog.domain) == 1     # one-hot for the terminal rule
                total += 1
            end
        end
        # Number of programs equals number of terminal rules
        @test total == length(term_ids)

        # Test whether number of programs per type equals the number of terminal rules of that type
        for t in keys(bank[0])
            n_terminals_of_t = count(i -> G.isterminal[i] && G.types[i] == t, eachindex(G.rules))
            @test length(bank[0][t]) == n_terminals_of_t
        end

        # Returned addresses align with what's in the bank and have bucket==0
        @test length(addrs) == total
        @test all(a -> a.addr[1] == 0, addrs)
    end

    @testset "combine(): bigger ordering/thresholding with many ties" begin
        # Grammar: Start = 1 | 2 | 3 | 4 | Start + Start
        G = @csgrammar begin
            Start = 1
            Start = 2
            Start = 3
            Start = 4
            Start = Start + Start
        end

        # Identify rule indices
        term_ids = findall(G.isterminal)           # 4 terminals
        @test length(term_ids) == 4
        plus_idx = only(findall(.!G.isterminal))   # the '+' rule

        # Depth-aware matrix (rows = depth; cols = rules).
        # Root '+' cost = 2 at depth 1.
        # Terminal costs at depth 2: cost("1")=1, "2"=2, "3"=3, "4"=4.
        M = fill(100.0, 3, length(G.rules))  # big default to avoid surprises
        M[1, plus_idx] = 2.0

        # Discover terminal labels by printing a UniformHole for each terminal rule.
        # Then set costs based on the label -> value map.
        label_value = Dict("1"=>1.0, "2"=>2.0, "3"=>3.0, "4"=>4.0)
        for j in term_ids
            bv = falses(length(G.isterminal))
            bv[j] = true
            u = UniformHole(bv, [])
            lab = replace(string(rulenode2expr(u, G)), r"[\s\(\)]" => "")
            @assert haskey(label_value, lab) "Unexpected terminal label '$lab' for rule index $j"
            M[2, j] = label_value[lab]
        end

        it = MyDepthBU(G, :Start; max_cost_in_bank = 50, current_max_cost = 4, costMatrix = M)
        HerbSearch.populate_bank!(it)
        state = HerbSearch.init_combine_structure(it)

        # Helper to reconstruct exprs/costs for a returned batch
        reconstruct_exprs_costs = function (addrs)
            progs  = [UniformHole(copy(a.op.domain), [retrieve(it, aa) for aa in a.addrs]) for a in addrs]
            costs  = [program_total_cost(it, p) for p in progs]
            exprs  = [replace(string(rulenode2expr(p, HerbSearch.get_grammar(it.solver))), r"[\s\(\)]" => "") for p in progs]
            return exprs, costs
        end

        # Helper to compute expected NEW exprs for limit L, given what we've already seen.
        # Programs are "x+y" where x,y ∈ {"1","2","3","4"}, cost = 2 + val(x) + val(y).
        labels = ["1","2","3","4"]
        seen = Set{String}()
        expected_new_at = function (L)
            newset = Set{String}()
            for x in labels, y in labels
                c = 2 + label_value[x] + label_value[y]
                if c <= L
                    s = string(x, "+", y)
                    if !(s in seen)
                        push!(newset, s)
                    end
                end
            end
            return newset
        end

        # ---- Step 1: limit = 4 ---- (only "1+1")
        state[:current_max_cost] = 4
        addrs1, state = HerbSearch.combine(it, state)
        @test !isnothing(addrs1)
        exprs1, costs1 = reconstruct_exprs_costs(addrs1)
        @test all(c -> c <= 4, costs1)
        @test costs1 == sort(costs1)
        expected1 = expected_new_at(4)
        @test Set(exprs1) == expected1
        union!(seen, expected1)

        # ---- Step 2: limit = 5 ---- ("1+2", "2+1")
        addrs2, state = HerbSearch.combine(it, state)
        exprs2, costs2 = reconstruct_exprs_costs(addrs2)
        @test all(c -> c <= 5, costs2)
        @test costs2 == sort(costs2)
        expected2 = expected_new_at(5)
        @test Set(exprs2) == expected2
        union!(seen, expected2)

        # ---- Step 3: limit = 6 ---- ("1+3", "3+1", "2+2")
        addrs3, state = HerbSearch.combine(it, state)
        exprs3, costs3 = reconstruct_exprs_costs(addrs3)
        @test all(c -> c <= 6, costs3)
        @test costs3 == sort(costs3)
        expected3 = expected_new_at(6)
        @test Set(exprs3) == expected3
        union!(seen, expected3)

        # ---- Step 4: limit = 7 ---- ("1+4", "4+1", "2+3", "3+2")
        addrs4, state = HerbSearch.combine(it, state)
        exprs4, costs4 = reconstruct_exprs_costs(addrs4)
        @test all(c -> c <= 7, costs4)
        @test costs4 == sort(costs4)
        expected4 = expected_new_at(7)
        @test Set(exprs4) == expected4
        union!(seen, expected4)

        # Optional cumulative sanity check
        expected_cumulative = Set{String}()
        for x in labels, y in labels
            (2 + label_value[x] + label_value[y]) <= 7 && push!(expected_cumulative, string(x, "+", y))
        end
        @test seen == expected_cumulative
    end

    @testset "combine(): type correctness and deduplication" begin
        # Same grammar & a symmetric matrix to let both sides be interchangeable
        G = @csgrammar begin
            Start = 1
            Start = 2
            Start = Start + Start
        end
        term_ids = findall(G.isterminal)
        plus_idx = only(findall(.!G.isterminal))

        M = fill(100.0, 3, length(G.rules))
        M[1, plus_idx]    = 3.0
        M[2, term_ids[1]] = 1.0
        M[2, term_ids[2]] = 1.0

        it = MyDepthBU(G, :Start; max_cost_in_bank = 10, current_max_cost = 6, costMatrix = M)
        HerbSearch.populate_bank!(it)
        state = HerbSearch.init_combine_structure(it)
        state[:current_max_cost] = 6

        addrs, _ = HerbSearch.combine(it, state)

        # All combined programs should be of type :Start
        let Gcur = HerbSearch.get_grammar(it.solver)
            @test all(a -> Gcur.types[findfirst(a.op.domain)] == :Start, addrs)
        end

        # Structural dedup: no duplicates among reconstructed programs
        progs = [UniformHole(copy(a.op.domain), [retrieve(it, aa) for aa in a.addrs]) for a in addrs]
        sig(p) = (findfirst(p.domain), [replace(string(rulenode2expr(c, HerbSearch.get_grammar(it.solver))), r"[\s\(\)]" => "") for c in p.children])
        @test length(progs) == length(unique(sig.(progs)))
    end
    @testset "combine(): advances when a cost level is empty" begin
        # Grammar: Start = 1 | 2 | Start + Start
        G = @csgrammar begin
            Start = 1
            Start = 2
            Start = Start + Start
        end

        term_ids = findall(G.isterminal)
        plus_idx = only(findall(.!G.isterminal))

        # Make the *minimum* total cost be 5 (= 3 for '+' at depth1, + 1 + 1 for leaves at depth2)
        M = fill(100.0, 3, length(G.rules))
        M[1, plus_idx] = 3.0           # '+' at depth 1
        M[2, term_ids[1]] = 1.0        # '1' at depth 2
        M[2, term_ids[2]] = 1.0        # '2' at depth 2

        it = MyDepthBU(G, :Start; max_cost_in_bank = 50, current_max_cost = 4, costMatrix = M)
        HerbSearch.populate_bank!(it)
        state = HerbSearch.init_combine_structure(it)
        state[:current_max_cost] = 4    # one below the minimum (empty layer)

        addrs, state2 = HerbSearch.combine(it, state)  # should auto-advance to the first non-empty layer
        @test !isnothing(addrs)

        # Reconstruct to compute true total costs
        progs  = [UniformHole(copy(a.op.domain), [retrieve(it, aa) for aa in a.addrs]) for a in addrs]
        costs  = [program_total_cost(it, p) for p in progs]

        @test all(c -> c >= 5, costs)   # nothing under the empty level leaked in
        @test all(c -> c <= state2[:current_max_cost], costs)  # returned under the new (advanced) limit
        @test costs == sort(costs)      # still cheapest-first within the returned batch
    end
    @testset "combine(): multi-depth nesting without add_to_bank! (manual push)" begin
        # Grammar: Start = 1 | inc(Start)
        G = @csgrammar begin
            Start = 1
            Start = inc(Start)
        end
        term_idx = only(findall(G.isterminal))
        inc_idx  = only(findall(.!G.isterminal))

        # Depth-aware costs:
        # inc at depth1 = 1, terminal at depth2 = 1  -> cost(inc(1)) = 2
        # inc at depth2 = 2, terminal at depth3 = 1  -> cost(inc(inc(1))) = 1 + 2 + 1 = 4
        M = fill(100.0, 3, length(G.rules))
        M[1, inc_idx] = 1.0
        M[2, term_idx] = 1.0
        M[2, inc_idx] = 2.0
        M[3, term_idx] = 1.0

        it = MyDepthBU(G, :Start; max_cost_in_bank = 10, current_max_cost = 2, costMatrix = M)
        HerbSearch.populate_bank!(it)
        state = HerbSearch.init_combine_structure(it)
        state[:current_max_cost] = 2

        # Step 1: get inc(1) at total cost 2
        addrs1, state = HerbSearch.combine(it, state)
        @test !isnothing(addrs1)
        progs1 = [UniformHole(copy(a.op.domain), [retrieve(it, aa) for aa in a.addrs]) for a in addrs1]
        exprs1 = [replace(string(rulenode2expr(p, G)), r"[\s\(\)]" => "") for p in progs1]
        costs1 = [program_total_cost(it, p) for p in progs1]
        @test "inc1" in exprs1
        @test any(==(2), costs1)

        # Manually push inc(1) into the bank so it can be used as a child next round.
        # We put it in bucket 0 to keep child enumeration simple (combine filters by true total cost anyway).
        bank = HerbSearch.get_bank(it)
        push!(bank[0][:Start], progs1[findfirst(==( "inc1"), exprs1)])

        # Step 2: raise limit to allow inc(inc(1)) and combine again
        state[:current_max_cost] = 4
        addrs2, state = HerbSearch.combine(it, state)
        @test !isnothing(addrs2)
        progs2 = [UniformHole(copy(a.op.domain), [retrieve(it, aa) for aa in a.addrs]) for a in addrs2]
        exprs2 = [replace(string(rulenode2expr(p, G)), r"[\s\(\)]" => "") for p in progs2]
        costs2 = [program_total_cost(it, p) for p in progs2]

        @test "incinc1" in exprs2
        @test costs2[findfirst(==( "incinc1"), exprs2)] == 4
        @test costs2 == sort(costs2)
    end
    @testset "combine(): optional deterministic tie-breaking within equal cost" begin
        # Grammar: Start = 1 | 2 | Start + Start
        G = @csgrammar begin
            Start = 1
            Start = 2
            Start = Start + Start
        end
        term_ids = findall(G.isterminal)
        plus_idx = only(findall(.!G.isterminal))

        # Make 1+2 and 2+1 have the same total and both under the limit:
        # '+' at depth1 = 1, terminals at depth2: cost(1)=2, cost(2)=2 => total = 1+2+2 = 5
        M = fill(100.0, 3, length(G.rules))
        M[1, plus_idx] = 1.0
        M[2, term_ids[1]] = 2.0
        M[2, term_ids[2]] = 2.0

        it = MyDepthBU(G, :Start; max_cost_in_bank = 10, current_max_cost = 5, costMatrix = M)
        HerbSearch.populate_bank!(it)
        state = HerbSearch.init_combine_structure(it)
        state[:current_max_cost] = 5

        addrs, _ = HerbSearch.combine(it, state)
        @test !isnothing(addrs)

        progs = [UniformHole(copy(a.op.domain), [retrieve(it, aa) for aa in a.addrs]) for a in addrs]
        exprs = [replace(string(rulenode2expr(p, G)), r"[\s\(\)]" => "") for p in progs]
        costs = [program_total_cost(it, p) for p in progs]

        # We expect exactly {"1+2","2+1"} among the equal-cost results.
        equal_cost_exprs = Set(exprs[findall(==(5), costs)])
        @test Set(["1+2","2+1"]) ⊆ equal_cost_exprs

        # Optional: enforce stable lexicographic tie order (uncomment if you implement a secondary tiebreak)
        # @test exprs == sort(exprs)
    end
    @testset "Depth-cost iterator: looks at programs in ascending cost" begin
        # ---------- Grammar ----------
        g = @csgrammar begin
            Int = Int - Int
            Int = Int * Int
            Int = Int + Int
            Int = 1
            Int = 2
        end

        M = fill(10_000.0, 4, length(g.rules))
        M[1, 1] = 1.0
        M[1, 2] = 2.0 
        M[1, 3] = 3.0
        M[1, 4] = 4.0
        M[1, 5] = 5.0

        M[2, 1] = 10.0
        M[2, 2] = 1.0 
        M[2, 3] = 2.0
        M[2, 4] = 2.0
        M[2, 5] = 1.0


        M[3, 1] = 10.0
        M[3, 2] = 10.0 
        M[3, 3] = 1.0
        M[3, 4] = 2.0
        M[3, 5] = 1.0
        pretty_print_counts(M)
        iter = MyDepthBU(g, :Int; max_cost_in_bank = 15, costMatrix = M )
        costs = []
        for (i, prog) in enumerate(iter)
            # println(prog)
            # println("PROGRAM #", i, " ",string(rulenode2expr(prog, g)))
            # println("COST!", program_total_cost(iter, prog))
            if i > 2
                push!(costs, program_total_cost(iter, prog))
            end
            if i > 30 
                break
            end
        end
        @test issorted(costs)        
    end
end


function run_depth_synthesis(M, G)  # pcsg built from Karel grammar

    problems = Karel_2018.get_all_problems()  # ~10 IOExamples per problem
    prob = problems[1]  # try one first

    # run with your probabilistic bottom-up iterator
    prog, hits = solve_one(prob, M, G, 10)  # increase 15 if you need a bigger search radius
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

function solve_one(prob::Problem, M, grammar, max_cost::Int)
    it = MyDepthBU(grammar, :Start; max_cost_in_bank = max_cost, costMatrix = M)
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


end # module

