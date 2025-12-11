module ProbeUpdate
export FlatProbeState, DepthProbeState,
       update_flat!, update_depth!,
       update_fit_from_program!,     # flat
       update_fit_from_program_depth!, # depth-aware    
       kernel_smoothing

using HerbSearch, HerbGrammar, HerbBenchmarks, HerbConstraints, HerbCore, HerbSpecification, HerbInterpret


# ----------------------------- State containers -----------------------------

struct FlatProbeState
    base::Vector{Float64}  # frozen baseline costs C0 (length R)
    fit::Vector{Float64}   # FIT per rule (length R), in [0,1]
    work::Vector{Float64}  # scratch/current updated costs (length R)
end

struct DepthProbeState
    base::Matrix{Float64}  # frozen baseline costs C0 (R × D)
    fit::Matrix{Float64}   # FIT per rule per depth (R × D)
    work::Matrix{Float64}  # scratch/current updated costs (R × D)
end

# Small helpers to init from existing costs
FlatProbeState(c0::Vector{Float64}) = FlatProbeState(copy(c0), zeros(length(c0)), copy(c0))
DepthProbeState(c0::Matrix{Float64}) = DepthProbeState(copy(c0), zeros(size(c0)), copy(c0))

# ----------------------------- Tree walking helpers -------------------------
# These rely on HerbSearch internals that you already use in your codebase:
#   _ruleindex(node)  -> Int   (global rule index of this node)
#   _children(node)   -> Vector{<:Any}
# If your project names differ, adjust these two functions accordingly.

@inline _rid(node) = _ruleindex(node)
@inline _kids(node) = _children(node)


# ---------- helper: rule index + children (UniformHole / StateHole) ----------
_ruleindex(u::UniformHole) = begin
    j = findfirst(u.domain)
    @assert j !== nothing "UniformHole has empty/invalid domain"
    j
end
_children(u::UniformHole) = u.children

_ruleindex(u::HerbConstraints.StateHole) = begin
    j = findfirst(u.domain)
    @assert j !== nothing "StateHole has empty/invalid domain"
    j
end
_children(u::HerbConstraints.StateHole) = u.children
# ----------------------------- FIT updates (flat) ----------------------------

"""
    update_fit_from_program!(fit, prog, hits, total)

Increase per-rule FIT (max with coverage) using coverage = hits/total, flat mode.
No depth bookkeeping here.
"""
function update_fit_from_program!(
    fit::Vector{Float64},
    prog::Union{HerbConstraints.StateHole, HerbSearch.UniformHole},
    hits::Int, total::Int
)
    total == 0 && return fit
    cov = clamp(hits / total, 0.0, 1.0)
    _upd_fit_flat!(fit, prog, cov)
    return fit
end

function _upd_fit_flat!(
    fit::Vector{Float64},
    node::Union{HerbConstraints.StateHole, HerbSearch.UniformHole},
    cov::Float64
)
    j = _rid(node)
    fit[j] = max(fit[j], cov)
    @inbounds for c in _kids(node)
        _upd_fit_flat!(fit, c, cov)
    end
    return fit
end


# -------------------------- FIT updates (depth-aware) -----------------------

"""
    update_fit_from_program_depth!(fit, prog, hits, total; depth0=1)

Increase FIT at (rule, depth) along the program tree, starting at `depth0`.
"""
function update_fit_from_program_depth!(
    fit::Matrix{Float64},                        # (R × D)
    prog::Union{HerbConstraints.StateHole, HerbSearch.UniformHole},
    hits::Int, total::Int; depth0::Int=1
)
    total == 0 && return fit
    cov = clamp(hits / total, 0.0, 1.0)
    _upd_fit_depth!(fit, prog, cov, depth0)
    return fit
end

function _upd_fit_depth!(
    fit::Matrix{Float64},                        # (R × D)
    node::Union{HerbConstraints.StateHole, HerbSearch.UniformHole},
    cov::Float64, d::Int
)
    R, D = size(fit)
    j = _rid(node)
    dd = clamp(d, 1, D)
    fit[j, dd] = max(fit[j, dd], cov)
    @inbounds for c in _kids(node)
        _upd_fit_depth!(fit, c, cov, d + 1)
    end
    return fit
end

# ---------------------- Cost normalization (per-LHS, base-2) ----------------

# Build LHS -> rule indices once from grammar.types
function _group_by_lhs(grammar::AbstractGrammar)
    by_type = Dict{Symbol, Vector{Int}}()
    @inbounds for (j, t) in pairs(grammar.types)
        push!(get!(by_type, t, Int[]), j)
    end
    return by_type
end

# Flat: b(r) = (1 - FIT[r]) * C0[r], then log-sum-exp normalize per LHS.
function _probe_update_costs_flat!(
    dest::Vector{Float64}, base::Vector{Float64},
    grammar::AbstractGrammar, fit::Vector{<:Real}
)
    @assert length(dest) == length(base) == length(fit)
    by_type = _group_by_lhs(grammar)
    @inbounds for (_, idxs) in by_type
        b = similar(base, length(idxs))
        for (k, j) in pairs(idxs)
            b[k] = max(0.0, 1.0 - clamp(fit[j], 0.0, 1.0)) * base[j]
        end
        m = minimum(b)
        K = 0.0
        for x in b
            K += 2.0 ^ (-(x - m))
        end
        Z = -m + log2(K)
        for (k, j) in pairs(idxs)
            dest[j] = b[k] + Z
        end
    end
    return dest
end

# Depth: identical idea but per depth column.
function _probe_update_costs_depth!(
    dest::Matrix{Float64}, base::Matrix{Float64},
    grammar::AbstractGrammar, fit::Matrix{<:Real}
)
    @assert size(dest) == size(base) == size(fit)
    R, D = size(base)
    by_type = _group_by_lhs(grammar)
    @inbounds for d in 1:D
        for (_, idxs) in by_type
            b = similar(base, length(idxs))
            for (k, j) in pairs(idxs)
                b[k] = max(0.0, 1.0 - clamp(fit[j, d], 0.0, 1.0)) * base[j, d]
            end
            m = minimum(b)
            K = 0.0
            for x in b
                K += 2.0 ^ (-(x - m))
            end
            Z = -m + log2(K)
            for (k, j) in pairs(idxs)
                dest[j, d] = b[k] + Z
            end
        end
    end
    return dest
end

# ----------------------------- Public update API ----------------------------

"""
    update_flat!(state, grammar, prog, hits, total) -> Vector{Float64}

Updates `state.fit` from `prog` coverage and returns updated flat costs.
Caller should write them back to the active grammar (e.g., `G.log_probabilities .= …`).
"""
function update_flat!(
    state::FlatProbeState,
    grammar::AbstractGrammar,
    prog::Union{HerbConstraints.StateHole, HerbSearch.UniformHole},
    hits::Int, total::Int
)::Vector{Float64}
    total == 0 && return state.work
    update_fit_from_program!(state.fit, prog, hits, total)
    _probe_update_costs_flat!(state.work, state.base, grammar, state.fit)
    return state.work
end

"""
    update_depth!(state, grammar, prog, hits, total; depth0=1) -> Matrix{Float64}

Depth-aware variant; caller writes result back to the iterator's (R × D) table.
"""
function update_depth!(
    state::DepthProbeState,
    grammar::AbstractGrammar,
    prog::Union{HerbConstraints.StateHole, HerbSearch.UniformHole},
    hits::Int, total::Int; depth0::Int=1
)::Matrix{Float64}
    total == 0 && return state.work
    update_fit_from_program_depth!(state.fit, prog, hits, total; depth0=depth0)
    _probe_update_costs_depth!(state.work, state.base, grammar, state.fit)
    return state.work
end

# -------------------------- Convenience for depth0 --------------------------

"""
    default_depth0(grammar, prog) -> Int

Heuristic start-depth used in your current code. You can keep using your custom
mapping or plug your own function here if you prefer.
"""
function default_depth0(grammar::AbstractGrammar,
                        prog::Union{HerbConstraints.StateHole, HerbSearch.UniformHole})
    rt = HerbGrammar.return_type(grammar, freeze_state(prog))
    # Match your earlier offsets:
    # Start -> +0, Block -> +1, ControlFlow -> +2, Action -> +2
    rt === :Block       && return 2
    rt === :ControlFlow && return 3
    rt === :Action      && return 3
    return 1
end

########################
# Fast rule-key lookup #
########################

const _RULEIDX_CACHE = IdDict{AbstractGrammar, Dict{String,Int}}()

# stringify RHS with minimal whitespace; good enough for this grammar
@inline _canon_rule_rhs(x) = replace(sprint(show, x), r"\s+" => "")
@inline _rulekey(t::Symbol, rhs) = String(t) * "->" * _canon_rule_rhs(rhs)

function _rule_index(grammar::AbstractGrammar, lhs::Symbol, rhs) :: Union{Int,Nothing}
    dict = get!(_RULEIDX_CACHE, grammar) do
        d = Dict{String,Int}()
        @inbounds for i in eachindex(grammar.rules)
            k = _rulekey(grammar.types[i], grammar.rules[i])
            d[k] = i
        end
        d
    end
    return get(dict, _rulekey(lhs, rhs), nothing)
end

@inline function _bump_fit_rule!(fit::Vector{Float64}, grammar::AbstractGrammar,
                                 lhs::Symbol, rhs, cov::Float64)
    idx = _rule_index(grammar, lhs, rhs)
    idx === nothing && return false
    fit[idx] = max(fit[idx], cov)
    return true
end

##############################################
# Lifting partial trees to a Start-rooted fit #
##############################################

"""
    update_fit_lifted_flat!(fit, grammar, prog, hits, total) -> Bool

Flat mode: updates FIT as if `prog` were wrapped to the minimal `Start`-rooted shape.
Returns `true` if any update was applied, `false` if prog type is unsupported to lift.
"""
function update_fit_lifted_flat!(
    fit::Vector{Float64},
    grammar::AbstractGrammar,
    prog,
    hits::Int, total::Int
) :: Bool
    total == 0 && return false
    cov = clamp(hits / total, 0.0, 1.0)

    rt = HerbGrammar.return_type(grammar, freeze_state(prog))

    if rt === :Start
        # normal flat path
        update_fit_from_program!(fit, prog, hits, total)
        return true
    elseif rt === :Block
        # Start -> Block
        _bump_fit_rule!(fit, grammar, :Start, :(Block), cov)
        _upd_fit_flat!(fit, prog, cov) # walk Block subtree
        return true
    elseif rt === :Action
        # Start -> Block ; Block -> Action
        _bump_fit_rule!(fit, grammar, :Start, :(Block), cov)
        _bump_fit_rule!(fit, grammar, :Block, :(Action), cov)
        _upd_fit_flat!(fit, prog, cov) # walk Action leaf (move/turnLeft/...)
        return true
    elseif rt === :ControlFlow
        # Start -> Block ; Block -> ControlFlow
        _bump_fit_rule!(fit, grammar, :Start, :(Block), cov)
        _bump_fit_rule!(fit, grammar, :Block, :(ControlFlow), cov)
        _upd_fit_flat!(fit, prog, cov) # walk IF/WHILE/...
        return true
    else
        # :Condition, :ConditionFlow, :INT, etc. — ambiguous wrapping; skip
        return false
    end
end

function calculate_lhs_ids(rules::Vector{String})
    lhs = map(r -> strip(split(r, "->"; limit=2)[1]), rules)
    uniq = Dict{String,Int}()
    out  = Vector{Int}(undef, length(rules))
    gid = 0
    @inbounds for i in eachindex(lhs)
        if !haskey(uniq, lhs[i])
            gid += 1
            uniq[lhs[i]] = gid
        end
        out[i] = uniq[lhs[i]]
    end
    out
end

function kernel_smoothing(C_dr::AbstractMatrix{<:Real}, rules::Vector{String}; alpha::Real=0.1, lambda::Real=1.0)
    @assert alpha > 0 && lambda > 0
    D, R = size(C_dr)
    @assert length(rules) == R "rules length must equal number of columns in C_dr (depths×rules)."
    lhs_ids = calculate_lhs_ids(rules)

    # Depth kernel W (D×D), column-normalized
    W = Matrix{Float64}(undef, D, D)
    @inbounds for d in 1:D
        s = 0.0
        for dp in 1:D
            w = exp(-lambda * abs(d - dp))
            W[dp, d] = w
            s += w
        end
        invs = 1.0 / s
        for dp in 1:D
            W[dp, d] *= invs
        end
    end

    # Group rule indices by LHS id
    groups = Dict{Int, Vector{Int}}()
    @inbounds for r in 1:R
        push!(get!(groups, lhs_ids[r], Int[]), r)
    end

    C64 = Array{Float64}(C_dr)
    P = zeros(Float64, D, R)

    # Smooth counts per LHS group across depths, then row-normalize within the group
    @inbounds for (_, idxs) in groups
        G = transpose(W) * (C64[:, idxs] .+ alpha)   # (D×D) * (D×|idxs|) → (D×|idxs|)
        for d in 1:D
            s = sum(@view G[d, :])
            invs = 1.0 / s
            for j in 1:length(idxs)
                P[d, idxs[j]] = G[d, j] * invs
            end
        end
    end
    P
end


end # module
