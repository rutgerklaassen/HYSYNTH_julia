#!/usr/bin/env julia

# =================== Bring Herb types into Main BEFORE includes ===================
using HerbSearch, HerbGrammar, HerbBenchmarks, HerbConstraints, HerbCore, HerbSpecification, HerbInterpret
using Serialization
using JSON
using Printf
using Dates
using DataStructures

# =================== Local includes ===================
include("parseSygus.jl")
include("hysynth.jl")
include("utils.jl")
include("depth_based_synthesis.jl")
include("probe_update.jl")
include("string_functions.jl")   # defines interpret_sygus, CVC ops, etc.
include("grammar.jl")            # for frequencies_to_costs_* and kernel_smoothing

using .Hysynth
using .Depth_synthesis
using .Utils
using .ProbeUpdate
using .ParseSyGuS
using .Grammar

# =================== SyGuS dataset types (MUST MATCH GENERATOR) ===================

struct SyGuSIOExample
    inputs::Dict{Symbol, Any}  # e.g. Dict(:_arg_1 => "AIX 5.1")
    output::Any                # String / Int / Bool
end

struct SyGuSProblemRecord
    name::String                   # "problem_11604909"
    grammar_name::String           # "grammar_11604909"
    grammar::ContextSensitiveGrammar
    examples::Vector{SyGuSIOExample}
end

struct SyGuSDataset
    problems::Vector{SyGuSProblemRecord}
end

# =================== CLI Args ===================

function parse_args()
    args = Dict{String, String}()

    # Existing options
    args["--dataset"]          = get(ENV, "SYGUS_DATASET", "sygus_dataset.jls")
    args["--nrows"]            = get(ENV, "DEPTH_ROWS", "6")
    args["--limit"]            = get(ENV, "LIMIT", "")
    args["--updating_enabled"] = get(ENV, "UPDATING", "true")
    args["--depth_aware"]      = get(ENV, "DEPTH_AWARE", "true")

    # New options: prompts / runs / LLM config
    args["--prompts_dir"]        = get(ENV, "PROMPTS_DIR", "prompts")
    args["--runs_dir"]           = get(ENV, "RUNS_DIR", "runs")
    args["--model"]              = get(ENV, "MODEL_FILTER", "")   # optional filter
    args["--max_answers"]        = get(ENV, "MAX_ANSWERS", "0")   # 0 = no cap
    args["--llm_enabled"]        = get(ENV, "LLM_ENABLED", "true")
    args["--smoothing_enabled"]  = get(ENV, "SMOOTHING_ENABLED", "false")

    i = 1
    while i <= length(ARGS)
        if startswith(ARGS[i], "--") && i < length(ARGS)
            args[ARGS[i]] = ARGS[i+1]
            i += 2
        else
            i += 1
        end
    end
    return args
end

# =================== Small helpers (hashing, JSON) ===================

const FNV_OFFSET64 = 0xcbf29ce484222325 % UInt64
const FNV_PRIME64  = 0x00000100000001B3 % UInt64

function _canon(expr::AbstractString)
    return replace(strip(expr), r"\s+" => " ")
end

function prog_fingerprint(expr::AbstractString)::UInt64
    h = FNV_OFFSET64
    ce = _canon(expr)
    @inbounds for b in codeunits(ce)
        h ⊻= UInt64(b)
        h *= FNV_PRIME64
    end
    return h
end

prog_fingerprint(x)::UInt64 = prog_fingerprint(repr(x))

safe_freeze(p) = try
    (freeze_state(p), true)
catch e
    if e isa AssertionError
        (nothing, false)
    else
        rethrow()
    end
end

function write_json(path::AbstractString, obj)
    dir = dirname(path)
    isdir(dir) || mkpath(dir)
    open(path, "w") do io
        JSON.print(io, obj)
    end
end

# =================== SyGuS evaluation via interpret_sygus ===================

# Build tags for a *specific* example:
# - Start from get_relevant_tags(grammar) (primitives, IF, etc.)
# - Override terminals that depend on the example: _arg_i, string literals, ints.
function build_sygus_tags(grammar::ContextSensitiveGrammar, ex::SyGuSIOExample)
    base_tags = get_relevant_tags(grammar)  # from string_functions.jl
    tags = copy(base_tags)

    for (i, rhs_expr) in enumerate(grammar.rules)
        rhs = string(rhs_expr) |> strip

        # --- arguments _arg_1, _arg_2, ... ---
        if startswith(rhs, "_arg_")
            idx_str = replace(rhs, "_arg_" => "")
            idx = try
                parse(Int, idx_str)
            catch
                1
            end
            key = Symbol("_arg_$(idx)")
            if haskey(ex.inputs, key)
                tags[i] = ex.inputs[key]
            else
                tags[i] = base_tags[i]
            end

        # --- string literal: "foo" ---
        elseif startswith(rhs, "\"") && endswith(rhs, "\"")
            tags[i] = rhs[2:end-1]

        # --- boolean literals ---
        elseif rhs == "true"
            tags[i] = true
        elseif rhs == "false"
            tags[i] = false

        else
            # --- integer literal, if any ---
            lit = try
                parse(Int, rhs)
            catch
                nothing
            end
            if lit !== nothing
                tags[i] = lit
            else
                tags[i] = base_tags[i]
            end
        end
    end

    return tags
end

function safe_interpret_sygus(prog::AbstractRuleNode, tags::Dict{Int,Any})
    try
        return interpret_sygus(prog, tags)
    catch
        return nothing
    end
end

# Hits over all examples in one SyGuS problem
function sygus_hits_and_total(fpnode::AbstractRuleNode, prob::SyGuSProblemRecord)
    G = prob.grammar
    hits = 0
    total = length(prob.examples)

    for ex in prob.examples
        tags = build_sygus_tags(G, ex)
        out = safe_interpret_sygus(fpnode, tags)
        if out !== nothing && out == ex.output
            hits += 1
        end
    end

    return hits, total
end

function sygus_satisfies_problem_strict(fpnode::AbstractRuleNode, prob::SyGuSProblemRecord)
    hits, total = sygus_hits_and_total(fpnode, prob)
    return hits == total
end

# =================== Iterator driver ===================

function run_depth_iter_for_problem(
        ds::SyGuSDataset,
        pid::Int,
        C::Matrix{Float64};
        max_cost=1e9,
        max_depth=8,
        max_size=10_000,
        jit_enabled::Bool,
        depth_aware_enabled::Bool,
        probe_state=nothing,
        flat_probe_state=nothing
    )

    prob = ds.problems[pid]
    G = prob.grammar

    # Start symbol depends on the SyGuS sort, but the CF grammar always uses :Start
    start_sym = :Start

    it = DepthCostBasedBottomUpIterator(
        G, start_sym, C;
        bank = HerbSearch.CostBank(),
        max_cost = max_cost,
        max_depth = max_depth,
        max_size = max_size,
        jit_enabled = jit_enabled,
    )

    best_hits  = 0
    best_size  = typemax(Int)
    best_prog  = nothing
    solved     = false

    reset_vars = Depth_synthesis.get_reset_variables(it)
    if reset_vars !== nothing && jit_enabled
        reset_vars.programs_since_reset = 0
        reset_vars.improved_since_reset = false
    end

    st    = nothing
    nxt   = iterate(it)
    steps = 0
    seen  = Set{UInt64}()

    while nxt !== nothing
        (p, st) = nxt
        fpnode, ok = safe_freeze(p)
        if !ok
            nxt = iterate(it, st)
            continue
        end

        steps += 1
        if steps % 10_000 == 0
            @printf("  [pid %d] steps = %d\n", pid, steps)
        end
        if steps >= 100_000
            println("Global step limit reached (100000); stopping on this problem.")
            break
        end

        expr = rulenode2expr(fpnode, G)
        fp   = prog_fingerprint(string(expr))
        if fp ∈ seen
            nxt = iterate(it, st)
            continue
        else
            push!(seen, fp)
        end

        if reset_vars !== nothing && jit_enabled
            reset_vars.programs_since_reset += 1
            if reset_vars.programs_since_reset >= 20_000
                if reset_vars.improved_since_reset
                    println("Budget reached with improvement; triggering reset")
                    reset_vars.depth_exhausted = true
                    reset_vars.programs_since_reset = 0
                    reset_vars.improved_since_reset = false
                else
                    println("Budget reached with no improvement; stopping search")
                    break
                end
            end
        end

        # --- spec checking ---
        strict_ok = sygus_satisfies_problem_strict(fpnode, prob)
        sz        = program_rule_count(p)

        if strict_ok
            println("program ", expr, " is an exact hit with size ", sz)
            solved = true
            return (; steps, best_hits=best_hits, best_size=sz, best_prog=p, solved)
        end

        # PROBE-style improvement: PBE hits only (no traces yet)
        hits, total = sygus_hits_and_total(fpnode, prob)
        improved = false
        if hits > 0
            improved = (hits > best_hits) || (hits == best_hits && sz < best_size)
        end

        if improved && jit_enabled
            best_hits = hits
            best_size = sz
            best_prog = p

            println("program ", expr,
                    " is new best: ", best_hits, " / ", total,
                    " size=", sz,
                    " depth=", Depth_synthesis.tree_depth(p))

            if depth_aware_enabled
                d0 = ProbeUpdate.default_depth0(HerbSearch.get_grammar(it.solver), p)
                new_costs = ProbeUpdate.update_depth!(probe_state, HerbSearch.get_grammar(it.solver), p, hits, total; depth0=d0)
                it.logp_by_depth .= new_costs
            else
                new_costs = ProbeUpdate.update_flat!(flat_probe_state, HerbSearch.get_grammar(it.solver), p, hits, total)
                it.logp_by_depth[:, 1] .= new_costs
            end

            if reset_vars !== nothing
                reset_vars.improved_since_reset = true
            end
        end

        nxt = iterate(it, st)
    end

    return (; steps, best_hits, best_size, best_prog, solved)
end

# =================== JSONL + runs / prompts helpers ===================

function read_jsonl(path::AbstractString)
    rows = Any[]
    open(path, "r") do io
        for line in eachline(io)
            s = strip(line)
            isempty(s) && continue
            push!(rows, JSON.parse(s))
        end
    end
    return rows
end

"""
    load_prompt_index(prompts_dir) -> (rows, by_pid)

`by_pid` maps the integer `pid` (1-based) used in the Julia dataset
to the corresponding prompt index row, which must contain `problem_hash`.
"""
function load_prompt_index(prompts_dir::AbstractString)
    idx_path = joinpath(prompts_dir, "index.jsonl")
    !isfile(idx_path) && error("Prompt index not found at $idx_path")
    rows = read_jsonl(idx_path)

    by_pid = Dict{Int, Any}()
    for r in rows
        if haskey(r, "pid")
            pid = Int(r["pid"])
            by_pid[pid] = r
        end
    end
    return rows, by_pid
end

"""
    collect_run_rows(runs_dir; model_filter="")

Collect all JSONL run rows (DeepSeek outputs) from `runs_dir`.
If `model_filter` is non-empty, only keep rows with that `model`.
"""
function collect_run_rows(runs_dir::AbstractString; model_filter::String="")
    !isdir(runs_dir) && error("Runs directory not found: $runs_dir")
    rows = Any[]
    files = sort(collect(readdir(runs_dir; join=true)))
    for f in files
        endswith(f, ".jsonl") || continue
        for r in read_jsonl(f)
            if model_filter != "" && get(r, "model", "") != model_filter
                continue
            end
            push!(rows, r)
        end
    end
    return rows
end

# Build: problem_hash -> Vector{String} (absolute answer paths)
function map_answers_by_hash(run_rows::Vector{Any}; base::AbstractString=".")
    mp = Dict{String, Vector{String}}()
    for r in run_rows
        if !haskey(r, "problem_hash") || r["problem_hash"] === nothing
            continue
        end
        ph = String(r["problem_hash"])
        ans_rel = String(r["answer_path"])
        ans_abs = isabspath(ans_rel) ? ans_rel : normpath(joinpath(base, ans_rel))
        push!(get!(mp, ph, String[]), ans_abs)
    end
    return mp
end

# =================== Parsing & Aggregation for SyGuS ===================

# Infer the nonterminal sort used on the RHS of Start: ntString / ntInt / ntBool
function infer_start_sort(G::ContextSensitiveGrammar)::Symbol
    for (lhs, rhs) in zip(G.types, G.rules)
        if lhs == :Start
            if rhs isa Symbol
                return rhs
            else
                s = string(rhs) |> strip
                toks = split(s)
                if !isempty(toks)
                    return Symbol(toks[1])
                end
            end
        end
    end
    # Fallback: most SYGUS string benchmarks are ntString
    return :ntString
end

# Try to parse one answer text into a tree; returns Union{Nothing, ParseTree}
function try_parse_answer(txt::AbstractString, G::ContextSensitiveGrammar)
    s = strip(txt)
    isempty(s) && return nothing
    start_sort = infer_start_sort(G)
    try
        return ParseSyGuS.parse_llm_response(String(s); start_sort=start_sort)
    catch e
        @warn "SyGuS parse failed" err=e
        return nothing
    end
end

# Build a counts matrix for SyGuS grammars (no Karel-specific hacks)
function counts_matrix_sygus(
        counts_by_depth::Dict{Int, Dict{String, Int}},
        grammar::ContextSensitiveGrammar;
        nrows::Int = 6)

    rules = [string(lhs, "->", rhs) for (lhs, rhs) in zip(grammar.types, grammar.rules)]
    rule_index = Dict(r => i for (i, r) in enumerate(rules))
    M = zeros(Int, nrows, length(rules))

    for (depth, d) in counts_by_depth
        1 <= depth <= nrows || continue
        for (rule_str, cnt) in d
            rs = replace(strip(rule_str), r"\s+" => " ")
            j = get(rule_index, rs, 0)
            j == 0 && continue
            M[depth, j] += cnt
        end
    end

    return M, rules
end

# For a set of trees, accumulate counts into a single numeric matrix:
function accumulate_counts_matrix_sygus(
        trees::Vector;
        G::ContextSensitiveGrammar,
        nrows::Int)

    M_sum = nothing
    rules_ref = nothing
    valid = 0

    for t in trees
        t === nothing && continue
        _, counts_by_depth = Utils.print_tree(t)
        M, rules = counts_matrix_sygus(counts_by_depth, G; nrows=nrows)
        if M_sum === nothing
            M_sum = copy(M)
            rules_ref = rules
        else
            if length(rules) != length(rules_ref) || any(rules .!= rules_ref)
                error("counts_matrix_sygus returned different rule ordering; please align columns explicitly.")
            end
            M_sum .+= M
        end
        valid += 1
    end

    return M_sum, rules_ref, valid
end

# =================== Main flow ===================

function main()
    cfg = parse_args()

    dataset_path        = cfg["--dataset"]
    nrows               = parse(Int, cfg["--nrows"])
    limit_probs         = isempty(cfg["--limit"]) ? typemax(Int) : parse(Int, cfg["--limit"])
    updating_enabled    = get(cfg, "--updating_enabled", "true") == "true"
    depth_aware_enabled = get(cfg, "--depth_aware", "true") == "true"

    prompts_dir         = cfg["--prompts_dir"]
    runs_dir            = cfg["--runs_dir"]
    model_filter        = cfg["--model"]
    max_answers_raw     = parse(Int, cfg["--max_answers"])
    max_answers         = max_answers_raw <= 0 ? typemax(Int) : max_answers_raw
    llm_enabled         = get(cfg, "--llm_enabled", "true") == "true"
    smoothing_enabled   = get(cfg, "--smoothing_enabled", "false") == "true"

    search_max_cost  = 1e9
    search_max_depth = 8
    search_max_size  = 10_000

    println("=== SyGuS + PROBE + LLM PCFG ===")
    println("dataset       : $dataset_path")
    println("prompts_dir   : $prompts_dir")
    println("runs_dir      : $runs_dir")
    println("model_filter  : $model_filter")
    println("nrows(depth)  : $nrows")
    println("llm_enabled   : $llm_enabled")
    println("smoothing     : $smoothing_enabled")
    println("updating      : $updating_enabled")
    println("depth-aware   : $depth_aware_enabled")

    ds = Serialization.deserialize(dataset_path)
    nprobs = length(ds.problems)
    println("Loaded dataset with $nprobs problems.")

    first_prob = ds.problems[1]
    println("First problem name : ", first_prob.name)
    println("Grammar name       : ", first_prob.grammar_name)
    println("Num examples       : ", length(first_prob.examples))

    # Load prompt index and run rows
    idx_rows, idx_by_pid = load_prompt_index(prompts_dir)
    println("Loaded $(length(idx_rows)) prompt index rows.")

    run_rows = collect_run_rows(runs_dir; model_filter=model_filter)
    println("Loaded $(length(run_rows)) run rows.")

    # Base directory for the answer paths produced by deepseek.py
    # (prompts/ and answers/ usually share the same parent)
    manifest_base = realpath(joinpath(prompts_dir, ".."))
    answers_by_hash = map_answers_by_hash(run_rows; base=manifest_base)
    println("answers_by_hash keys: ", length(answers_by_hash))

    processed = 0
    solved    = 0
    results   = Any[]

    exp_root = "experiments_sygus"
    isdir(exp_root) || mkpath(exp_root)
    ts   = Dates.format(now(UTC), "yyyymmdd_HHMMSS")
    exp_dir = joinpath(exp_root,
                       "sygus__probe-$(updating_enabled)__depth-$(depth_aware_enabled)__llm-$(llm_enabled)__$(ts)")
    mkpath(exp_dir)
    per_problem_path = joinpath(exp_dir, "per_problem.json")
    summary_path     = joinpath(exp_dir, "summary.json")
    bad_answers_dir  = joinpath(exp_dir, "bad_answers")
    mkpath(bad_answers_dir)

    println("\nWriting artifacts to: $exp_dir\n")

    for (pid, prob) in enumerate(ds.problems)
        processed >= limit_probs && break

        G = prob.grammar
        nrules = length(G.rules)

        println("\n[PID $(pid)] $(prob.name)")
        println("  rules: $(nrules)")

        # ---------- LLM → counts → costs ----------
        idx_rec = get(idx_by_pid, pid, nothing)
        problem_hash = nothing
        ans_paths = String[]

        if idx_rec !== nothing && haskey(idx_rec, "problem_hash")
            problem_hash = String(idx_rec["problem_hash"])
            ans_paths = get(answers_by_hash, problem_hash, String[])
        end

        trees = Any[]
        n_parsed = 0

        if llm_enabled && problem_hash !== nothing && !isempty(ans_paths)
            println("  Found $(length(ans_paths)) answers for hash=$problem_hash (taking up to $max_answers).")

            taken = 0
            for ap in ans_paths
                taken += 1
                taken > max_answers && break
                if isfile(ap)
                    txt = read(ap, String)
                    t = try_parse_answer(txt, G)
                    if t === nothing
                        outname = @sprintf("pid_%04d_sample_%04d.txt", pid, taken)
                        open(joinpath(bad_answers_dir, outname), "w") do io
                            write(io, txt)
                        end
                    else
                        push!(trees, t)
                        n_parsed += 1
                    end
                else
                    @warn "Answer file missing" ap
                end
            end
        elseif llm_enabled
            println("  No answers found for this problem (or missing hash); using uniform costs.")
        end

        # Default costs: uniform
        CDepth = ones(Float64, nrules, nrows)
        Cflat  = ones(Float64, nrules)

        if llm_enabled && !isempty(trees)
            println("  Parsed $(n_parsed) / $(length(ans_paths)) answers, building counts matrix...")
            M_sum, rules, valid = accumulate_counts_matrix_sygus(trees; G=G, nrows=nrows)

            if M_sum === nothing || valid == 0
                println("  Could not form counts matrix; falling back to uniform costs.")
            else
                if smoothing_enabled
                    println("  Applying kernel smoothing to frequencies...")
                    M_sum = kernel_smoothing(M_sum, rules)
                end

                # Depth-aware cost matrix (depth × rules)
                Cdepth_raw = Grammar.frequencies_to_costs_depth(M_sum, rules; alpha=1.0, eps=1e-3)
                # Transpose to (rules × depth) for DepthCostBasedBottomUpIterator
                CDepth = permutedims(Cdepth_raw, (2, 1))

                # Flat cost vector
                Cflat = Grammar.frequencies_to_costs_flat(M_sum, rules)

                println("  counts-matrix size: ", size(M_sum), " -> depth-cost size: ", size(CDepth))
            end
        else
            if llm_enabled
                println("  No parseable answers; uniform costs.")
            else
                println("  LLM disabled; uniform costs.")
            end
        end

        # ---------- Run search ----------
        if depth_aware_enabled
            depth_state = ProbeUpdate.DepthProbeState(CDepth)
            res = run_depth_iter_for_problem(
                ds, pid, CDepth;
                max_cost=search_max_cost,
                max_depth=search_max_depth,
                max_size=search_max_size,
                jit_enabled=updating_enabled,
                depth_aware_enabled=true,
                probe_state=depth_state,
            )
        else
            C1 = reshape(Cflat, :, 1)
            flat_state = ProbeUpdate.FlatProbeState(Cflat)
            res = run_depth_iter_for_problem(
                ds, pid, C1;
                max_cost=search_max_cost,
                max_depth=search_max_depth,
                max_size=search_max_size,
                jit_enabled=updating_enabled,
                depth_aware_enabled=false,
                flat_probe_state=flat_state,
            )
        end

        println("  SOLVED: ", res.solved,
                "  steps=", res.steps,
                "  best_hits=", res.best_hits)

        if res.solved
            solved += 1
        end

        push!(results, Dict(
            "pid"            => pid,
            "problem_name"   => prob.name,
            "grammar_name"   => prob.grammar_name,
            "problem_hash"   => problem_hash,
            "answers_total"  => problem_hash === nothing ? 0 : length(ans_paths),
            "answers_parsed" => n_parsed,
            "steps_iterated" => res.steps,
            "best_hits"      => res.best_hits,
            "best_size"      => res.best_size,
            "solved"         => res.solved,
            "timestamp_utc"  => Dates.format(now(UTC), dateformat"yyyy-mm-ddTHH:MM:SS\\Z")
        ))

        processed += 1
    end

    println("\n=== Summary ===")
    println("Processed problems : ", processed)
    println("Fully solved       : ", solved)
    println("Timestamp          : ", Dates.format(now(UTC), dateformat"yyyy-mm-ddTHH:MM:SS\\Z"))

    write_json(per_problem_path, results)

    summary = Dict(
        "processed"         => processed,
        "solved"            => solved,
        "dataset_path"      => dataset_path,
        "prompts_dir"       => prompts_dir,
        "runs_dir"          => runs_dir,
        "model_filter"      => model_filter,
        "nrows"             => nrows,
        "max_answers"       => max_answers,
        "limit"             => limit_probs,
        "llm_enabled"       => llm_enabled,
        "smoothing_enabled" => smoothing_enabled,
        "updating_enabled"  => updating_enabled,
        "depth_aware"       => depth_aware_enabled,
        "search_max_cost"   => search_max_cost,
        "search_max_depth"  => search_max_depth,
        "search_max_size"   => search_max_size,
        "timestamp_utc"     => Dates.format(now(UTC), dateformat"yyyy-mm-ddTHH:MM:SS\\Z")
    )
    write_json(summary_path, summary)

    println("\nWrote per-problem stats to: $(per_problem_path)")
    println("Wrote summary to           : $(summary_path)")
end

# =================== Entry ===================
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
