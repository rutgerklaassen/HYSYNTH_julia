
using Random
using Logging
using Base: Expr
using Printf
using HerbGrammar
using HerbSearch
import HerbSearch
import HerbSearch.combine
# Step 1 : Get the response

#run(`python test5.py`)
if isfile("response.txt")
    println("File exists!")
else
    println("File does not exist.")
end
response = read("response.txt", String)
println(response)

# Okay now that we have the response let's do something with it? 

# Step 2 : Learning PCFG from solution  

# Okay we have a solution here, how do we get to a pcfg? 

# Step 2.1 : Extract Rule sequence 

# Define a parse tree structure
# It works like any other tree, where a node contains the rule used and defines the children
mutable struct ParseTree
    rule::String
    children::Vector{ParseTree}
end

# Tokenize the LLM response into components
function tokenize(response::String)
    # Match words, function calls, and symbols
    # Weird regex but it works like this:
    # \w+\( matches functions like turnleft() or Grab()
    # \w+ matches tandalone words like if or drop
    return collect(eachmatch(r"IF\(([^()]|(\w+\(\)))*\)|WHILE\(([^()]|(\w+\(\)))*\)|\w+\(\)|\w+", response)) .|> x->x.match
end

# Parse a Transformation operation
# Convert token to String to avoid SubString issues
function parse_transformation(token::AbstractString)
    token = String(token)  # Convert to String
    if token in ["moveRight()", "moveDown()", "moveLeft()", "moveUp()", "drop()", "grab()"]
        return ParseTree("Transformation->$token", [])
    end
    return nothing  # Not a valid transformation
end

function parse_condition(token::AbstractString)
    token = String(token)  # Convert to String
    if token in ["atTop()", "atBottom()", "atLeft()", "atRight()", "notAtTop()", "notAtBottom()", "notAtLeft()", "notAtRight()"]
        return ParseTree("Condition->$token", [])
    end
    return nothing  # Not a valid condition
end

############ TESTING ############
# Finds the correct closing parenthesis for IF(...) or WHILE(...)
function find_matching_paren(tokens::Vector{<:AbstractString}, start_index::Int)
    count = 0
    for i in start_index:length(tokens)
        if occursin("(", tokens[i])  # Increase count for any "(" inside token
            count += 1
        end
        if occursin(")", tokens[i])  # Decrease count for any ")" inside token
            count -= 1
            if count == 0
                return i  # Found the correct matching ")"
            end
        end
    end
    return -1  # No matching ")"
end

# Extracts the condition and sequences from IF(...) or WHILE(...)
function extract_parts(tokens::Vector{<:AbstractString}, start_index::Int, end_index::Int)
    content = join(tokens[start_index:end_index], " ")  # Join back into a string
    println("",content)
    return split(content, ",")  # Split by commas
end


    # function parse_control(tokens::Vector{<AbstractString}, Index::Int)
    #     token = String(tokens[index])


# Parses IF(...) and WHILE(...) correctly, using fixed logic
function parse_control(tokens::Vector{<:AbstractString}, index::Int)
    token = String(tokens[index])  # Convert to String
    if startswith(token, "IF(")
        # Find the matching closing ")"
        close_index = find_matching_paren(tokens, index + 1)
        if close_index == -1
            return nothing  # No matching ')'
        end

        # Extract condition, sequence1, sequence2
        inner_content = token[4:end-1] 
        parts = split(inner_content, ",")
        println("Extracted parts: ", parts)  
            
        if length(parts) == 3
            cond = parse_condition(String(parts[1]))
            seq1 = parse_sequence(String(parts[2]))
            seq2 = parse_sequence(String(parts[3]))
            if cond !== nothing && seq1 !== nothing && seq2 !== nothing
                return ParseTree("ControlStatement->IF(Condition, Sequence, Sequence)", [cond, seq1, seq2])
            end
        end
    elseif startswith(token, "WHILE(")
        close_index = find_matching_paren(tokens, index + 1)
        if close_index == -1
            return nothing  # No matching ')'
        end

        # Extract condition, sequence1, sequence2
        inner_content = token[7:end-1] 
        parts = split(inner_content, ",")
        println("Extracted parts: ", parts) 

        if length(parts) == 2
            cond = parse_condition(String(parts[1]))
            seq = parse_sequence(String(parts[2]))
            if cond !== nothing && seq !== nothing
                return ParseTree("ControlStatement->WHILE(Condition, Sequence)", [cond, seq])
            end
        end
    end
    return nothing  # Not a valid control statement
end

function parse_sequence(response::String)
    tokens = tokenize(response)
    println(tokens)

    if length(tokens) == 0
        return nothing  # No valid sequence
    elseif length(tokens) == 1
        # Base case: a single operation
        op_tree = parse_transformation(tokens[1])  # Try parsing transformation
        if op_tree !== nothing
            return ParseTree("Sequence->Operation", [ParseTree("Operation->Transformation", [op_tree])])
        end

        op_tree = parse_control(tokens, 1)  # Try parsing control
        if op_tree !== nothing
            return ParseTree("Sequence->Operation", [ParseTree("Operation->ControlStatement", [op_tree])])
        end

        return nothing  # Not a valid sequence
    else
        # Recursive case: Operation; Sequence
        first_op_tree = parse_transformation(tokens[1])
        if first_op_tree !== nothing
            first_op_tree = ParseTree("Operation->Transformation", [first_op_tree])
        else
            first_op_tree = parse_control(tokens, 1)
            if first_op_tree !== nothing
                first_op_tree = ParseTree("Operation->ControlStatement", [first_op_tree])
            else
                return nothing  # Invalid token
            end
        end

        # Recursively parse the rest of the sequence
        rest_sequence_tree = parse_sequence(join(tokens[2:end], " "))  
        
        if rest_sequence_tree === nothing
            return ParseTree("Sequence->Operation", [first_op_tree])
        else
            return ParseTree("Sequence->(Operation; Sequence)", [first_op_tree, rest_sequence_tree])
        end
    end
end


# Wrapper function to parse an entire LLM response
function parse_llm_response(response::String)
    sequence_tree = parse_sequence(response)
    if sequence_tree !== nothing
        return ParseTree("Start->Sequence", [sequence_tree])
    else
        return nothing  # Parsing failed
    end
end

function print_tree(tree::ParseTree, rule_counts=Dict{String, Int}(), prefix="", is_last=true)
    # Count occurrences of rules
    rule_counts[tree.rule] = get(rule_counts, tree.rule, 0) + 1

    # Define tree branch symbols
    connector = is_last ? "└── " : "├── "

    # Print the current node with the appropriate prefix
    println(prefix * connector * tree.rule * " (x$(rule_counts[tree.rule]))")

    # Update prefix for children
    new_prefix = prefix * (is_last ? "    " : "│   ")

    # Recursively print children
    for (i, child) in enumerate(tree.children)
        print_tree(child, rule_counts, new_prefix, i == length(tree.children))
    end

    return rule_counts  # Return rule counts for further analysis
end

# Example LLM responses
llm_response_1 = "moveRight(); moveDown(); drop(); grab()"
llm_response_2 = "IF(atTop(), moveDown(), moveRight()); grab()"
llm_response_3 = "WHILE(notAtBottom(), moveDown()); drop()"

# Parsing and displaying the parse tree
parsed_tree_1 = parse_llm_response(llm_response_1)
parsed_tree_2 = parse_llm_response(llm_response_2)
parsed_tree_3 = parse_llm_response(llm_response_3)


println("Parsed Tree 1:")
if parsed_tree_1 !== nothing 
    rule_counts_1 = print_tree(parsed_tree_1, Dict{String, Int}())  
    println("\nRule Occurrences for Tree 1: ", rule_counts_1)
else 
    println("Parsing failed.") 
end

println("\nParsed Tree 2:")
if parsed_tree_2 !== nothing 
    rule_counts_2 = print_tree(parsed_tree_2, Dict{String, Int}())  
    println("\nRule Occurrences for Tree 2: ", rule_counts_2)
else 
    println("Parsing failed.") 
end

println("\nParsed Tree 3:")
if parsed_tree_3 !== nothing 
    rule_counts_3 = print_tree(parsed_tree_3, Dict{String, Int}())  
    println("\nRule Occurrences for Tree 3: ", rule_counts_3)
else 
    println("Parsing failed.") 
end

println("AAAAAAAAAAAAAAA")
println(rule_counts_1)
println("AAAAAAAAAAAAAAA")
println(rule_counts_2)
println("AAAAAAAAAAAAAAA")
println(rule_counts_3)

function construct_dict(rule_counts::Dict{String, Int})
    grouped_rules = Dict{String, Dict{String, Float64}}()

    # Group raw counts by LHS
    for (rule, count) in rule_counts
        parts = split(rule, "->")
        if length(parts) != 2
            error("Invalid rule format: $rule")
        end

        lhs, rhs = parts
        if !haskey(grouped_rules, lhs)
            grouped_rules[lhs] = Dict{String, Float64}()
        end

        grouped_rules[lhs][replace(rhs, r"\s+" => "")] = count
    end

    return grouped_rules
end

println("VVVVVVVVVVVVv")
println(construct_dict(rule_counts_1))

function canonical_rule_string(expr::Expr)
    # Handle blocks like: quote Operation; Sequence end
    if expr.head == :block && length(expr.args) == 2 &&
       expr.args[1] == :Operation && expr.args[2] == :Sequence
        return "(Operation;Sequence)"
    end

    return replace(string(expr), r"\s+" => "")
end

function canonical_rule_string(sym::Symbol)
    return replace(string(sym), r"\s+" => "")
end

function canonical_rule_string(x)
    return replace(string(x), r"\s+" => "")
end

function make_pcsg_from_dict(grammar::ContextSensitiveGrammar, prob_dict::Dict{String, Dict{String, Float64}})
    rules = grammar.rules
    bytype_in = grammar.bytype

    selected_rules = Expr[]
    log_probs = Float64[]
    bytype_out = Dict{Symbol, Vector{Int}}()

    for (lhs, rule_indices) in bytype_in
        lhs_str = string(lhs)
        dict_for_lhs = get(prob_dict, lhs_str, Dict{String, Float64}())

        rule_exprs = [rules[i] for i in rule_indices]
        rule_keys = [canonical_rule_string(expr) for expr in rule_exprs]
        counts = [get(dict_for_lhs, key, 0.0) for key in rule_keys]
        weights = [count + 1.0 for count in counts]  # alpha = 1.0
        total = sum(counts) + 1.0 * length(rule_keys)
        probs = weights ./ total
        @show lhs rule_keys counts

        for (expr, prob) in zip(rule_exprs, probs)
            expr = expr isa Expr ? expr : Expr(:call, expr)
            push!(selected_rules, expr)
            push!(log_probs, log(prob))

            if !haskey(bytype_out, lhs)
                bytype_out[lhs] = Int[]
            end
            push!(bytype_out[lhs], length(selected_rules))
        end
    end

    return ContextSensitiveGrammar(
        selected_rules,
        grammar.types,
        grammar.isterminal,
        grammar.iseval,
        bytype_out,
        grammar.domains,
        grammar.childtypes,
        grammar.bychildtypes,
        log_probs,
        grammar.constraints
    )
end



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

# prob_dict = Dict(
#     "Start" => Dict("Sequence" => 1.0),
#     "Sequence" => Dict("Operation" => 0.25, "(Operation;Sequence)" => 0.75),
#     "Operation" => Dict("Transformation" => 0.5, "ControlStatement" => 0.5),
#     "Transformation" => Dict(
#         "moveRight()" => 0.25,
#         "moveDown()" => 0.25,
#         "grab()" => 0.25,
#         "drop()" => 0.25
#     )
# )

pcsg = make_pcsg_from_dict(grammar_robots, construct_dict(rule_counts_1))

function print_pcsg_rules(grammar::ContextSensitiveGrammar)
    println("Probabilistic Grammar Rules:\n")

    for (lhs, rule_indices) in grammar.bytype
        println("  [$lhs]")

        for i in rule_indices
            rule = grammar.rules[i]
            prob = exp(grammar.log_probabilities[i])

            @printf("    %.3f : %s = %s\n", prob, lhs, sprint(show, rule))
        end

        println()
    end
end

print_pcsg_rules(pcsg)
#println(pcsg)

# TESTS

using HerbCore
using HerbSearch
using HerbGrammar
using HerbInterpret
using HerbConstraints
using HerbSpecification
using Test
using Aqua
using Random
import HerbSearch.init_combine_structure
import HerbSearch.get_bank
import HerbSearch.get_grammar
import HerbSearch.combine



@testset "Bottom Up Search" begin
    grammars_to_test = Dict(
        "arity <= 1" => (@csgrammar begin
            Int = 1 | 2
            Int = 3 + Int
        end),
        "arity = 2" => (@csgrammar begin
            Int = 1 | 2
            Int = Int + Int
        end)
    )

    function test_with_grammars(f, grammars)
        for (name, g) in grammars
            @testset "$name" f(g)
        end
    end

    @programiterator mutable MyBU(bank) <: BottomUpIterator

    function HerbSearch.init_combine_structure(iter::MyBU)
        Dict(:max_combination_depth => 10)
    end
    HerbSearch.get_bank(iter::MyBU) = iter.bank
    HerbSearch.get_grammar(iter::MyBU) = iter.solver.grammar

    @testset "basic" begin
        g = grammars_to_test["arity = 2"]
        iter = MyBU(g, :Int, nothing; max_depth=5)
        expected_programs = [
            (@rulenode 1),
            (@rulenode 2),
            (@rulenode 3{1,1}),
            (@rulenode 3{2,1}),
            (@rulenode 3{1,2}),
            (@rulenode 3{2,2})
        ]

        progs = [freeze_state(p) for (i, p) in enumerate(iter) if i <= 6]
        @test issetequal(progs, expected_programs)
        @test length(expected_programs) == length(progs)
    end

    @testset "combine" begin
        test_with_grammars(grammars_to_test) do g
            iter = MyBU(g, :Int, nothing; max_depth=5)
            create_bank!(iter)
            populate_bank!(iter)

            combinations, state = combine(iter, init_combine_structure(iter))
            @test !isempty(combinations)
        end
    end

    @testset "enumeration order" begin
        g = @csgrammar begin
            Int = 1
            Int = 2
            Int = Int + Int
        end
    
        iter = MyBU(g, :Int, nothing; max_depth=5)
    
        # This will collect programs in order as they are returned
        progs = [freeze_state(p) for (i, p) in enumerate(iter) if i <= 6]
        println(progs)
        expected_order = [
            (@rulenode 1),                     # Int = 1
            (@rulenode 2),                     # Int = 2
            (@rulenode 3{1,1}),                # 1 + 1
            (@rulenode 3{1,2}),                # 1 + 2
            (@rulenode 3{2,1}),                # 2 + 1
            (@rulenode 3{2,2})                 # 2 + 2
        ]
    
        @test progs == expected_order  # 🔍 Order matters!
    end
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

@testset "pcfg prioritization in combine" begin
    # Declare rules in order that mismatches their probabilities
    g = @csgrammar begin
        Int = Int + Int   # Rule 1
        Int = Int - Int   # Rule 2
        Int = Int * Int   # Rule 3
        Int = 1
        Int = 2
    end

    # Flip the priority compared to rule declaration order
    rule_counts = Dict(
        "Int->Int+Int" => 10,   # Should be second
        "Int->Int-Int" => 1,    # Should be last
        "Int->Int*Int" => 100,  # Should be chosen FIRST
        "Int->1" => 1,
        "Int->2" => 1
    )

    pcfg = make_pcsg_from_dict(g, construct_dict(rule_counts))
    iter = MyBU(pcfg, :Int, nothing; max_depth=3)

    create_bank!(iter)
    populate_bank!(iter)
    combinations, _ = combine(iter, init_combine_structure(iter))

    @test !isempty(combinations)

    top_n = min(5, length(combinations))
    rule_indices = [findfirst(ca.op.domain) for ca in combinations[1:top_n]]
    grammar = get_grammar(iter)
    rule_names = [sprint(show, grammar.rules[i]) for i in rule_indices]

    @info "Top rule names (should reflect PCFG priority)" rule_names=rule_names

    idx_mul = findfirst(contains("Int * Int"), rule_names)
    idx_plus = findfirst(contains("Int + Int"), rule_names)
    idx_minus = findfirst(contains("Int - Int"), rule_names)

    @test idx_mul !== nothing
    @test idx_plus !== nothing
    @test idx_minus !== nothing

    @test idx_mul < idx_plus < idx_minus  # 'Int * Int' should appear first!
end
