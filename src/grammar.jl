module Grammar
export construct_dict, make_pcsg_from_dict, grammar_robots, print_pcsg_rules, frequencies_to_costs_depth, frequencies_to_costs_flat, apply_flat_costs

using HerbGrammar
using Printf

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
    log_probs = fill(Inf, length(grammar.rules))  # Start with high cost (not -Inf)
    alpha = 1.0  # Laplace smoothing parameter
    println(prob_dict)
    for (lhs, rule_indices) in bytype_in
        lhs_str = string(lhs)
        dict_for_lhs = get(prob_dict, lhs_str, Dict{String, Float64}())

        rule_exprs = [rules[i] for i in rule_indices]
        rule_keys = [canonical_rule_string(expr) for expr in rule_exprs]

        counts = [get(dict_for_lhs, key, 0.0) for key in rule_keys]
        weights = [count + alpha for count in counts]
        total = sum(weights)
        probs = weights ./ total
        for (expr, prob) in zip(rule_exprs, probs)
            cost = max(-log2(prob), 1e-3)  # or 0.01, or some other small value
            for (j, rule) in enumerate(rules)
                if expr == rule
                    log_probs[j] = cost
                    break
                end
            end
        end
    end
    return ContextSensitiveGrammar(
        grammar.rules,
        grammar.types,
        grammar.isterminal,
        grammar.iseval,
        grammar.bytype,
        grammar.domains,
        grammar.childtypes,
        grammar.bychildtypes,
        log_probs,
        grammar.constraints
    )
end

function print_pcsg_rules(grammar::ContextSensitiveGrammar)
    println("Probabilistic Grammar Rules:\n")
    for (lhs, rule_indices) in grammar.bytype
        println("  [$lhs]")
        for i in rule_indices
            rule = grammar.rules[i]
            prob = grammar.log_probabilities[i]

            @printf("    %.3f : %s = %s\n", prob, lhs, sprint(show, rule))
        end
        println()
    end
end

"""
    group_rule_columns_by_lhs(rules) -> Dict{String, Vector{Int}}

Group column indices by LHS ("Start", "Block", "Action", ...),
assuming rule strings are "LHS->RHS".
"""
function group_rule_columns_by_lhs(rules::Vector{String})
    groups = Dict{String, Vector{Int}}()
    @inbounds for (j, r) in pairs(rules)
        lhs_rhs = split(r, "->"; limit=2)
        length(lhs_rhs) == 2 || error("Bad rule string: $r")
        lhs = lhs_rhs[1]
        push!(get!(groups, lhs, Int[]), j)
    end
    return groups
end


function frequencies_to_costs_depth(
    M::AbstractMatrix{<:Real},
    rules::Vector{String};
    alpha::Float64 = 0.01,
    eps::Float64   = 1e-3
)
    nrows, ncols = size(M)
    length(rules) == ncols || error("rules length ($(length(rules))) ≠ number of columns ($ncols)")

    C = Array{Float64}(undef, nrows, ncols)

    @inbounds for i in 1:nrows
        counts  = M[i, :]              # 1 × R row: depth i
        weights = counts .+ alpha      # Laplace smoothing (optional)
        total   = sum(weights)
        total > 0 || error("Unexpected zero total after smoothing at depth $i")

        probs = weights ./ total 
        C[i, :] = max.(-log2.(probs), eps)
    end

    return C
end


function frequencies_to_costs_flat(
    M::AbstractMatrix{<:Real},
    rules::Vector{String};
    alpha::Float64 = 0.01,
    eps::Float64   = 1e-3
)::Vector{Float64}
    nrows, ncols = size(M)
    length(rules) == ncols || error("rules length ($(length(rules))) ≠ number of columns ($ncols)")

    # total counts per rule across all depths / samples
    total_per_rule = vec(sum(M; dims=1))  # 1×R -> R

    weights = total_per_rule .+ alpha
    total   = sum(weights)
    total > 0 || error("Unexpected zero total after smoothing")

    probs = weights ./ total             # normalizeOkay alright over all rules
    return max.(-log2.(probs), eps)
end

# Apply per-rule costs to a Herb grammar (set log_probabilities).
function apply_flat_costs(grammar::ContextSensitiveGrammar,
                          rules::Vector{String},
                          costs::Vector{Float64};
                          default_cost::Float64 = 10.0)
    length(rules) == length(costs) || error("rules and costs must have same length")
    lookup = Dict{String,Float64}()
    for (j, rulekey) in pairs(rules)
        lookup[rulekey] = costs[j]
    end

    logp = fill(default_cost, length(grammar.rules))
    for (lhs, idxs) in grammar.bytype
        lhs_str = String(lhs)
        for i in idxs
            key = string(lhs_str, "->", canonical_rule_string(grammar.rules[i]))
            if haskey(lookup, key)
                logp[i] = lookup[key]
            end
        end
    end

    return ContextSensitiveGrammar(
        grammar.rules, grammar.types, grammar.isterminal, grammar.iseval,
        grammar.bytype, grammar.domains, grammar.childtypes, grammar.bychildtypes,
        logp, grammar.constraints
    )
end

end # module