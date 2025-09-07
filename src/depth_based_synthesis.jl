
module Depth_synthesis

export run_depth_synthesis_tests, run_depth_synthesis, run_priority_robot_test

using HerbGrammar, HerbSearch, HerbCore, HerbConstraints, HerbSpecification, HerbBenchmarks, Test
using DataStructures: DefaultDict
using Printf
using HerbBenchmarks.Karel_2018

include("traces.jl")
using .Traces

import ..Grammar
import HerbSearch: get_bank
# --- fake program tree selecting rules 1 and 3 (unary chain is enough) ---
# Minimal StateHole-like struct for the test:
struct MiniNode
    j::Int
    children::Vector{MiniNode}
end
@programiterator MyDepthBU(
    bank = DefaultDict{Int,DefaultDict}(() -> (DefaultDict{Symbol,AbstractVector{AbstractRuleNode}}(() -> AbstractRuleNode[]))),
    max_cost_in_bank = 0,
    current_max_cost = 2,
    seen_programs = Set{UniformHole}(),  
    costMatrix = Matrix{Float64}(undef, 0, 0),
    baseCostMatrix = Matrix{Float64}(undef, 0, 0)
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
function program_total_cost(iter::MyDepthBU, prog::HerbConstraints.StateHole, depth::Int=1)
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

# one-hot rule index and children
_ruleindex(u::HerbConstraints.StateHole) = begin
    j = findfirst(u.domain)
    @assert j !== nothing "Statehole has empty/invalid domain"
    j
end
_children(u::HerbConstraints.StateHole) = u.children

# Count number of rules used in a StateHole tree
function program_rule_count(node::HerbConstraints.StateHole)
    total = 1                       # count this node
    @inbounds for c in node.children
        total += program_rule_count(c)
    end
    return total
end

# Bump Fit ONLY along the current program’s tree
function update_fit_from_program!(Fit::AbstractMatrix{Float64},
                                  prog::HerbConstraints.StateHole,
                                  hits::Int, N::Int; depth::Int=1)
    cov = (N == 0) ? 0.0 : clamp(hits / N, 0.0, 1.0)
    _upd_fit!(Fit, prog, cov, depth);  return Fit
end

function _upd_fit!(Fit, node::HerbConstraints.StateHole, cov::Float64, depth::Int)
    d = clamp(depth, 1, size(Fit,1))
    j = _ruleindex(node)
    Fit[d, j] = max(Fit[d, j], cov)
    for c in _children(node)
        _upd_fit!(Fit, c, cov, depth + 1)
    end
end

function probe_update_costs!(M::AbstractMatrix{Float64},   # destination (current)
                             C0::AbstractMatrix{Float64},  # BASE costs (fixed)
                             G,
                             Fit::AbstractMatrix{<:Real})
    D, R = size(M)
    # @assert size(C0)  == (D, R)
    # @assert size(Fit) == (D, R)
    types = G.types
    for d in 1:D
        for t in unique(types)
            idxs = findall(types .== t)

            # Cost_d(r) = -log_2(P_d(r))
            # P_base_d(r) = 2 ^ -C0_(d,r)
            # So the PROBE formula can now be rewritten in terms of costs :
            # (2 ^ -C0(d,r)) ^ 1 - FIT_d(r) = 2 ^ -((1 - FIT) * C0(d,r))
            #  DEFINITION : b(d, r) === -((1 - FIT) * C0(d,r)) 
            # So now we have the full update formula for cost 
            # C'[d,r] = -log_2(P'_d(r)) = -log_2( 2^-b(d,r) / Z) where Z is the sum of 2^-b(d,r) over all same LHS at same depth
            # because we have -log_2(a / b) = -log_2 (a) + log_2(b) 
            # We get -log_2(2^-b(d,r)) + (Z) = b(d,r) + log_ 2 (Z)

            # Here we calculate the b(d,r) for every same lhs
            b = [max(0.0, 1 - clamp(Fit[d,j], 0.0, 1.0)) * C0[d,j] for j in idxs]

            # Here we compute Z (so all the b's added up)
            m = minimum(b)
            K = sum(2.0 ^ (-(x - m)) for x in b)  # terms are now ≤ 1, no underflow
            Z = -m + log2(K)
            # final updated costs
            @inbounds for (k, j) in enumerate(idxs)
                M[d, j] = b[k] + Z
            end

        end
    end
    return M
end

function run_depth_synthesis(M, G)  # pcsg built from Karel grammar
    exs, trs, meta = load_packed_with_traces("examples_packed.npz")

    println("Loaded ", length(exs), " examples")
    println("Meta: ", meta)

    # Check grouping
    k = meta["num_examples_per_code"]
    problems, problem_traces = group_examples_and_traces(exs, trs; num_examples_per_code=k)

    # --- Pretty-print one program + its k traces (frames included) ---
    # 1) normalize k (npz scalar may load as 0-d array)
    k = Int(k isa AbstractArray ? k[] : k)

    # 2) choose which program to inspect
    prog_idx = 1  # change to any 1..length(problems)

    println("\n==============================")
    println("PROGRAM #", prog_idx)

    # If codes were packed, show token ids of this program
    if haskey(meta, "codes_padded") && haskey(meta, "code_lengths")
        codes_padded = meta["codes_padded"]
        code_lengths = meta["code_lengths"]

        # any of the k examples in the group share the same program;
        # take the first example's row in the flat arrays
        start = (prog_idx - 1) * k + 1
        L = Int(code_lengths[start])
        tok_ids = vec(codes_padded[start, 1:L])

        println("Token IDs: ", tok_ids)
    else
        println("(No token ids found in NPZ: only traces will be shown.)")
    end

    # Helper to show a Karel state (replace with your ASCII renderer if you have one)
    show_state(s::HerbBenchmarks.Karel_2018.KarelState) = (show(s); println())

    # 3) print all k traces (every frame)
    println("\nTRACES (k = ", k, ")")
    trs_for_prog = problem_traces[prog_idx]  # Vector{Trace{KarelState}} length k
    exs_for_prog = problems[prog_idx].spec   # Vector{IOExample} length k (to sanity-check ends)

    @assert length(trs_for_prog) == k
    @assert length(exs_for_prog) == k

    for tix in 1:k
        tr = trs_for_prog[tix]
        ex = exs_for_prog[tix]

        println("\n--- trace ", tix, " ---")
        println("frames: ", length(tr.exec_path))

        # verify first/last match the IOExample (useful integrity check)
        println("matches input?  ", tr.exec_path[1]  == ex.in[:_arg_1])
        println("matches output? ", tr.exec_path[end] == ex.out)

        # print every frame (initial is frame 0)
        for (f, st) in enumerate(tr.exec_path)
            println("frame ", f-1, ":")
            show_state(st)
        end
    end
    println("==============================\n")
    println("Loaded ", length(problems), " programs, each with ", k, " examples")
    println("First problem has ", length(problem_traces[1]), " traces")
    println("First trace length = ", length(problem_traces[1][1].exec_path))

    exit()
    problems = Karel_2018.get_all_problems()  # ~10 IOExamples per problem
    prob = problems[1]  # try one first

    # run with your probabilistic bottom-up iterator
    prog, hits = solve_one(prob, M, G, 10)  # increase 15 if you need a bigger search radius
    if isnothing(prog)
        println("No program found. Best coverage: $hits / $(length(prob.spec))")
    else
        println("Solved with $hits / $(length(prob.spec))")
        println("Program: ", rulenode2expr(prog, G))
    end
end

function count_matches(prog::AbstractRuleNode, prob::Problem, grammar)
    hits = 0
    for ex in prob.spec
        println(ex)
        println(typeof(ex))
        exit()
        # println(typeof(prog))
        # frozen = HerbConstraints.freeze_state(prog)
        # println(typeof(frozen), typeof(grammar), typeof(ex))
        # println(grammar_karel)
        out = Karel_2018.interpret(prog, grammar_karel, ex, Dict{Int,AbstractRuleNode}())  # Karel interpreter
        hits += (out == ex.out) ? 1 : 0
    end
    return hits, length(prob.spec)
end

function solve_one(prob::Problem, M, grammar, max_cost::Int)
    it = MyDepthBU(grammar, :Start; max_cost_in_bank = max_cost, costMatrix = M)

    C0  = copy(M)                                        # BASE costs (fixed)
    Fit = zeros(Float64, size(M,1), length(grammar.rules))  # D×R, starts at 0

    best_prog = nothing
    best_hits = 0
    best_size = typemax(Int)


    for (i, prog) in enumerate(it)
        println("Program: ", rulenode2expr(prog, grammar))
        hits, N = count_matches(prog, prob, grammar)
        size_now = program_rule_count(prog)
        trigger = false
        if hits > best_hits
            best_hits = hits
            best_size = size_now
            best_prog = deepcopy(prog)
            trigger = true
        elseif hits == best_hits && size_now < best_size
            best_size = size_now
            best_prog = deepcopy(prog)
            trigger = true
        end

        #Update ONLY when triggered
        if trigger
            update_fit_from_program!(Fit, prog, hits, N)
            probe_update_costs!(it.costMatrix, C0, grammar, Fit)

            if hits == N; break; end
        end
    end
    println("BEST PROG BEFORE RETURN:", best_prog)
    return best_prog, best_hits
end

###################################################
###################################################
################### TESTS #########################
###################################################
###################################################
###################################################

function run_depth_synthesis_tests()
    function probe_update(p_u::Vector{Float64}, used::Vector{Bool}, fit::Float64)
        @assert 0.0 ≤ fit ≤ 1.0
        @assert length(p_u) == length(used)
        weights = [p_u[i]^(1.0 - (used[i] ? fit : 0.0)) for i in eachindex(p_u)]
        Z = sum(weights)
        p_new = weights ./ Z
        costs = round.(Int, .*(-log.(p_new)))
        return p_new, costs
    end

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
            if i > 2
                push!(costs, program_total_cost(iter, prog))
            end
            if i > 30 
                break
            end
        end
        @test issorted(costs)        
    end
    @testset "Cost matrix & PROBE-style updates" begin
        # Grammar with exactly three rules for one type S:
        #   S = x()           (terminal)
        #   S = y()           (terminal)
        #   S = node(S, S)    (nonterminal)
        G = @csgrammar begin
            S = x()
            S = y()
            S = node(S, S)
        end

        R = length(G.rules)

        # Indices of rules whose LHS is :S (all of them here).
        S_rule_idxs = findall(G.types .== :S)
        @test length(S_rule_idxs) == 3

        # Helper: convert a cost row to normalized probabilities on a subset of columns
        costrow_to_probs = function(costrow::AbstractVector{<:Real}, cols::AbstractVector{Int})
            v = 2.0 .^ (-costrow[cols])
            v ./ sum(v)
        end

        # Base probabilities for the 3 rules at each depth (sum to 1 so Fit=0 is a no-op)
        p0 = [0.5, 0.3, 0.2]
        C_A = -log2.(p0)  # base costs: -log2 p

        # Two-depth matrices (D=2) so we also touch the depth loop in probe_update_costs!
        D = 2
        C0 = fill(1000.0, D, R)   # base costs (big default elsewhere to avoid accidental mass)
        M  = similar(C0)          # destination for updated costs
        Fit = zeros(Float64, D, R)

        # Same base distribution at both depths
        C0[1, S_rule_idxs] .= C_A
        C0[2, S_rule_idxs] .= C_A

        # 1) Fit = 0 everywhere → updated costs equal the base costs (since probs already sum to 1 per LHS)
        probe_update_costs!(M, C0, G, Fit)
        @test isapprox(costrow_to_probs(M[1, :], S_rule_idxs), p0; atol=1e-12, rtol=1e-12)
        @test isapprox(costrow_to_probs(M[2, :], S_rule_idxs), p0; atol=1e-12, rtol=1e-12)

        # 2) Boost the middle rule at depth 1: Fit = [0.0, 0.6, 0.0]
        #    PROBE math: P'_d(r) ∝ P_base_d(r)^(1 - Fit_d(r))
        Fit .= 0.0
        Fit[1, S_rule_idxs] .= [0.0, 0.6, 0.0]
        probe_update_costs!(M, C0, G, Fit)

        p_expected = p0 .^ (1 .- [0.0, 0.6, 0.0]); p_expected ./= sum(p_expected)
        @test isapprox(costrow_to_probs(M[1, :], S_rule_idxs), p_expected; atol=1e-12, rtol=1e-12)
        @test costrow_to_probs(M[1, :], S_rule_idxs)[2] > p0[2]        # boosted rule got more probable
        @test abs(sum(costrow_to_probs(M[1, :], S_rule_idxs)) - 1.0) < 1e-12

        # 3) Depth separation: changing Fit at depth 1 doesn't perturb depth 2
        @test isapprox(costrow_to_probs(M[2, :], S_rule_idxs), p0; atol=1e-12, rtol=1e-12)

        # 4) Sanity on arity lookup: use childtypes, not a nonexistent G.arity
        #    node(S, S) must have arity 2; terminals have arity 0
        arities = map(j -> length(G.childtypes[j]), 1:R)
        @test sort(arities) == [0, 0, 2]
    end
end

###################################################
###################################################
################### END TESTS #####################
###################################################
###################################################
###################################################



end # module

