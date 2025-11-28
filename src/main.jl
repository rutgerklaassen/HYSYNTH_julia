#!/usr/bin/env julia

# =================== Includes (your local modules) ===================
include("parseKarel.jl")
include("grammar.jl")
include("hysynth.jl")
include("utils.jl")
include("depth_based_synthesis.jl")
include("karel_utils.jl")
include("probe_update.jl")

using .ParseKarel
using .Grammar
using .Hysynth
using .Depth_synthesis
using .Utils
using .KarelUtils
using .ProbeUpdate

using HerbSearch, HerbGrammar, HerbBenchmarks, HerbConstraints, HerbCore, HerbSpecification, HerbInterpret
using Serialization
using JSON
using Printf
using Dates
using StatProfilerHTML
using DataStructures

# =================== CLI Args ===================
function parse_args()
    args = Dict{String, String}()

    # defaults
    args["--dataset"]     = get(ENV, "KAREL_DATASET", "karel_dataset_by_depth.jls")
    args["--prompts-dir"] = get(ENV, "PROMPTS_DIR", "prompts")
    args["--runs-dir"]    = get(ENV, "RUNS_DIR", "runs")
    args["--model"]       = get(ENV, "MODEL_FILTER", "")             # optional: only use runs for this model
    args["--max-answers"] = get(ENV, "MAX_ANSWERS", "")              # optional: cap answers per problem
    args["--nrows"]       = get(ENV, "DEPTH_ROWS", "6")              # rows = depth buckets for counts_matrix
    args["--limit"]       = get(ENV, "LIMIT", "")                    # optional: max problems to process

    # crude flag parser
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

# =================== IO Helpers ===================
function read_jsonl(path::AbstractString)
    entries = Any[]
    open(path, "r") do io
        for ln in eachline(io)
            s = strip(ln)
            isempty(s) && continue
            push!(entries, JSON.parse(s))
        end
    end
    return entries
end

function load_prompt_index(prompts_dir::AbstractString)
    idx_path = joinpath(prompts_dir, "index.jsonl")
    if !isfile(idx_path)
        error("Missing prompts index: $(idx_path). Generate prompts first.")
    end
    by_file = Dict{String, Any}()
    by_pid  = Dict{String, Any}()
    by_hash = Dict{String, Any}()
    for rec in read_jsonl(idx_path)
        file = String(rec["file"])
        pid  = String(rec["problem_id"])
        by_file[file] = rec
        by_pid[pid]   = rec
        if haskey(rec, "problem_hash")
            by_hash[String(rec["problem_hash"])] = rec
        end
    end
    return by_file, by_pid, by_hash
end

function collect_run_rows(runs_dir::AbstractString; model_filter::String="")
    rows = Any[]
    if !isdir(runs_dir)
        @warn "Runs dir not found" runs_dir
        return rows
    end
    for (root, _, files) in walkdir(runs_dir)
        for f in files
            if startswith(f, "run_") && endswith(f, ".jsonl")
                path = joinpath(root, f)
                for rec in read_jsonl(path)
                    if isempty(model_filter) || get(rec, "model", "") == model_filter
                        push!(rows, rec)
                    end
                end
            end
        end
    end
    return rows
end

# Build: problem_id -> Vector{String} (answer paths)
function map_answers_by_problem(run_rows::Vector{Any}; base::AbstractString=".")
    mp = Dict{String, Vector{String}}()
    for r in run_rows
        pid = String(r["problem_id"])
        ans_rel = String(r["answer_path"])
        ans_abs = isabspath(ans_rel) ? ans_rel : normpath(joinpath(base, ans_rel))
        push!(get!(mp, pid, String[]), ans_abs)
    end
    return mp
end

# Build: problem_hash -> Vector{String} (answer paths)
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

function write_json(path::AbstractString, obj)
    dir = dirname(path)
    isdir(dir) || mkpath(dir)
    open(path, "w") do io
        JSON.print(io, obj;)
    end
end

# =================== Parsing & Aggregation ===================

# Try to parse one answer text into a tree; returns Union{Nothing, Any}
function try_parse_answer(txt::AbstractString)
    s = strip(txt)
    isempty(s) && return nothing
    try
        return ParseKarel.parse_llm_response(String(s))
    catch e
        @warn "Parse failed" err=e
        return nothing
    end
end

# For a set of trees, accumulate counts into a single numeric matrix:
function accumulate_counts_matrix(trees::Vector; G::AbstractGrammar, nrows::Int)
    M_sum = nothing
    rules_ref = nothing
    valid = 0
    for t in trees
        t === nothing && continue
        _, counts_by_depth = Utils.print_tree(t)
        M, rules = Utils.counts_matrix(counts_by_depth, G; nrows=nrows)
        if M_sum === nothing
            M_sum = copy(M)
            rules_ref = rules
        else
            if length(rules) != length(rules_ref) || any(rules .!= rules_ref)
                error("counts_matrix returned different rule ordering; please align columns explicitly.")
            end
            M_sum .+= M
        end
        valid += 1
    end
    return M_sum, rules_ref, valid
end

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

# =================== Iterator driver ===================
function run_depth_iter_for_problem(ds, pid::Int, G::AbstractGrammar, C::Matrix{Float64};
                                    max_cost=1e9, max_depth=10, max_size=50, jit_enabled, traces_enabled, depth_aware_enabled, probe_state=nothing, flat_probe_state=nothing)
    rec  = ds.programs[pid]
    prob = Problem("karel-traces", rec.traces)

    if depth_aware_enabled
        it = DepthCostBasedBottomUpIterator(G, :Start, C;
            bank=HerbSearch.CostBank(),
            max_cost=max_cost, max_depth=max_depth, max_size=max_size,
            jit_enabled=jit_enabled
        )
    else
        println("DEPTH TURNED OFF!!!!!!!!!!!!!!!!!!!!!")
        # Non-depth path uses the same iterator, but with a single cost column (R × 1)
        it = DepthCostBasedBottomUpIterator(G, :Start, C;
            bank=HerbSearch.CostBank(),
            max_cost=max_cost, max_depth=max_depth, max_size=max_size,
            jit_enabled=jit_enabled
        )
    end
    
    best_hits = 0
    best_size = typemax(Int)
    best_extra = 0
    best_prog = nothing
    solved = false
    reset_variables = Depth_synthesis.get_reset_variables(it)
    if reset_variables !== nothing && jit_enabled
        reset_variables.programs_since_reset = 0
        reset_variables.improved_since_reset = false
    end

    st = nothing
    nxt = iterate(it)
    steps = 0
    seen = Set{UInt64}()
    while nxt !== nothing
        (p, st) = nxt
        fpnode, ok = safe_freeze(p)
        if !ok
            nxt = iterate(it, st)
            continue
        end

        steps += 1
        if steps % 1000 == 0
            println(steps)
        end
        if steps >= 100000
            println("Global step limit reached (100000); stopping search on this problem.")
            break
        end
        expr = rulenode2expr(fpnode, G)
        fp   = prog_fingerprint(expr)
        if fp ∈ seen
            nxt = iterate(it, st)
            continue
        else
            push!(seen, fp)
        end

        if reset_variables !== nothing && jit_enabled
            reset_variables.programs_since_reset += 1

            if reset_variables.programs_since_reset >= 20_000
                if reset_variables.improved_since_reset
                    # Case 1: budget reached AND we improved → PROBE-style reset
                    println("Budget reached with improvement; triggering reset")
                    reset_variables.depth_exhausted = true
                    reset_variables.programs_since_reset = 0
                    reset_variables.improved_since_reset = false
                else
                    # Case 2: budget reached and no improvement → stop search on this problem
                    println("Budget reached with no improvement; stopping search")
                    break
                end
            end
        end


        hits, total, extra = KarelUtils.count_matches(fpnode, prob, G)
        strict_ok = KarelUtils.satisfies_problem_strict(fpnode, prob, G)
        sz = program_rule_count(p)

        if strict_ok
            println("program ", rulenode2expr(fpnode, G), " is an exact hit with size ", sz)
            solved = true
            return (; steps, best_hits, best_size, best_prog, solved)
        end
        improved = false
        if traces_enabled
            hits, total, extra = KarelUtils.count_matches(fpnode, prob, G)     # trace-level progress
            if hits > 0 
                improved = (hits > best_hits) ||
                    (hits == best_hits && extra < best_extra) ||
                    (hits == best_hits && extra == best_extra && sz < best_size)
            end
        else
            # world-level: only reward *complete* worlds
            hits, total = KarelUtils.count_world_matches(fpnode, prob, G)
            extra = 0
            if hits > 0
                improved = (hits > best_hits) || (hits == best_hits && sz < best_size)
            end
        end

        if improved && jit_enabled
            best_hits = hits; best_size = sz; best_prog = p; best_extra = extra;
            println("program ", rulenode2expr(freeze_state(p), G),
                    " is a new best hit! ", best_hits , " / ", total,
                    " extra=", extra, " size=", sz, " depth=", Depth_synthesis.tree_depth(p))
            if depth_aware_enabled
                d0 = ProbeUpdate.default_depth0(HerbSearch.get_grammar(it.solver), p)
                new_costs = ProbeUpdate.update_depth!(probe_state, HerbSearch.get_grammar(it.solver), p, hits, total; depth0=d0)
                it.logp_by_depth .= new_costs
            else
                new_costs = ProbeUpdate.update_flat!(flat_probe_state, HerbSearch.get_grammar(it.solver), p, hits, total)
                it.logp_by_depth[:, 1] .= new_costs
            end

            if reset_variables !== nothing
                reset_variables.improved_since_reset = true
                # DO NOT set depth_exhausted here anymore
            end
        end

        nxt = iterate(it, st)
    end

    return (; steps, best_hits, best_size, best_prog, solved)
end

# =================== Experiment folder naming ===================
function build_experiment_dir(manifest_base::AbstractString;
    llm_enabled::Bool, traces_enabled::Bool, updating_enabled::Bool, depth_aware_enabled::Bool,
    model_filter::String, nrows::Int, max_depth::Int, kernel_smoothing_enabled::Bool)

    parts = String[
        llm_enabled      ? "llm"      : "nollm",
        traces_enabled   ? "traces"   : "notraces",
        updating_enabled ? "probe"    : "noprobe",
        depth_aware_enabled ? "depth" : "nodepth",
        kernel_smoothing_enabled ? "smoothed" : "",
        isempty(model_filter) ? "model-any" : "model-" * replace(model_filter, r"[^A-Za-z0-9_.-]" => "_"),
        "nrows-$(nrows)",
        "md-$(max_depth)",
        Dates.format(now(UTC), "yyyymmdd_HHMMSS")
    ]
    exp_root = joinpath(manifest_base, "experiments")
    exp_dir  = joinpath(exp_root, join(parts, "__"))
    isdir(exp_dir) || mkpath(exp_dir)
    return exp_dir
end

# =================== Main flow ===================
function main()
    cfg = parse_args()
    dataset_path = cfg["--dataset"]
    prompts_dir  = cfg["--prompts-dir"]
    runs_dir     = cfg["--runs-dir"]
    model_filter = cfg["--model"]

    nrows        = parse(Int, cfg["--nrows"])
    max_answers  = isempty(cfg["--max-answers"]) ? typemax(Int) : parse(Int, cfg["--max-answers"])
    limit_probs  = isempty(cfg["--limit"]) ? typemax(Int) : parse(Int, cfg["--limit"])
    llm_enabled      = get(cfg, "--llm_enabled", "false") == "true"
    updating_enabled = get(cfg, "--updating_enabled", "false") == "true"
    traces_enabled   = get(cfg, "--traces_enabled", "false") == "true"
    depth_aware_enabled = get(cfg, "--depth_aware_enabled", "false") == "true"
    kernel_smoothing_enabled = get(cfg, "--smoothing-enabled", "false") == "true"
    oracle = get(cfg, "--oracle_enabled", "false") == "true"

    # Search knobs (lift to CLI later if desired)
    search_max_cost  = 1e9
    search_max_depth = 8
    search_max_size  = 10000

    println("=== Build PCFGs from answers & run depth-based synthesis ===")
    println("dataset     : $dataset_path")
    println("prompts_dir : $prompts_dir")
    println("runs_dir    : $runs_dir")
    println("model_filter: $(isempty(model_filter) ? "<any>" : model_filter)")
    println("nrows(depth): $nrows")

    # Collect run rows and map answers
    run_rows = collect_run_rows(runs_dir; model_filter=model_filter)
    manifest_base = realpath(joinpath(prompts_dir, ".."))  # project root: parent of prompts/
    answers_by_hash = map_answers_by_hash(run_rows; base=manifest_base)

    # Experiment directory
    exp_dir = build_experiment_dir(manifest_base;
        llm_enabled=llm_enabled, traces_enabled=traces_enabled, updating_enabled=updating_enabled, depth_aware_enabled=depth_aware_enabled,
        model_filter=model_filter, nrows=nrows, max_depth=search_max_depth, kernel_smoothing_enabled=kernel_smoothing_enabled)

    per_problem_path = joinpath(exp_dir, "per_problem.json")
    summary_path     = joinpath(exp_dir, "summary.json")
    bad_answers_dir  = joinpath(exp_dir, "bad_answers")
    isdir(bad_answers_dir) || mkpath(bad_answers_dir)

    println("\nWriting artifacts to: $exp_dir\n")

    # Load dataset
    ds = Serialization.deserialize(dataset_path)
    nprogs = length(ds.programs)
    println("Loaded dataset with $nprogs programs.")

    # Load prompt index maps
    _, by_pid_map, _ = load_prompt_index(prompts_dir)

    # Grammar
    G = KarelUtils.grammar_karel

    processed = 0
    solved = 0
    results = Any[]

    # Iterate problems
    for (pid, rec) in enumerate(ds.programs)
        # find index record for this pid
        idx_rec = nothing
        for (_, recidx) in by_pid_map
            if Int(recidx["pid"]) == pid
                idx_rec = recidx
                break
            end
        end
        idx_rec === nothing && continue

        problem_hash = haskey(idx_rec, "problem_hash") ? String(idx_rec["problem_hash"]) : ""
        if isempty(problem_hash)
            @info "No problem_hash in index; skipping for safety" pid
            continue
        end

        ans_paths  = get(answers_by_hash, problem_hash, String[])
        problem_id = String(idx_rec["problem_id"])
        depth      = Int(idx_rec["depth"])

        if isempty(ans_paths)
            @info "No answers for problem" pid problem_id depth
            push!(results, Dict(
                "pid"            => pid,
                "problem_id"     => problem_id,
                "problem_hash"   => problem_hash,
                "depth"          => depth,
                "answers_total"  => 0,
                "answers_parsed" => 0,
                "steps_iterated" => 0,
                "best_hits"      => 0,
                "best_size"      => nothing,
                "solved"         => false,
                "timestamp_utc"  => Dates.format(now(UTC), dateformat"yyyy-mm-ddTHH:MM:SS\\Z")
            ))
            processed += 1
            processed >= limit_probs && break
            continue
        end

        # Read answers and parse
        trees = Any[]
        taken = 0
        for ap in ans_paths
            taken += 1
            taken > max_answers && break
            if isfile(ap)
                txt = ""
                if oracle
                    txt = rec.program
                else
                    txt = read(ap, String)
                end
                t = try_parse_answer(txt)
                if t === nothing
                    outname = @sprintf("pid_%04d_sample_%04d.txt", pid, taken)
                    open(joinpath(bad_answers_dir, outname), "w") do io
                        write(io, txt)
                    end
                else
                    push!(trees, t)
                end
            else
                @warn "Answer file missing" ap
            end
        end

        if isempty(trees)
            @info "All answers failed to parse" pid problem_id
            push!(results, Dict(
                "pid"            => pid,
                "problem_id"     => problem_id,
                "problem_hash"   => problem_hash,
                "depth"          => depth,
                "answers_total"  => length(ans_paths),
                "answers_parsed" => 0,
                "steps_iterated" => 0,
                "best_hits"      => 0,
                "best_size"      => nothing,
                "solved"         => false,
                "timestamp_utc"  => Dates.format(now(UTC), dateformat"yyyy-mm-ddTHH:MM:SS\\Z")
            ))
            processed += 1
            processed >= limit_probs && break
            continue
        end

        # Accumulate and convert to costs
        M_sum, rules, nvalid = accumulate_counts_matrix(trees; G=G, nrows=nrows)
        if kernel_smoothing_enabled
            println("smoothing frequencies...")
            M_sum = kernel_smoothing(M_sum, rules) #transpose it so it works with the function 
        end
        #println(size(M_sum))
        if M_sum === nothing
            @info "Could not form counts matrix" pid
            push!(results, Dict(
                "pid"            => pid,
                "problem_id"     => problem_id,
                "problem_hash"   => problem_hash,
                "depth"          => depth,
                "answers_total"  => length(ans_paths),
                "answers_parsed" => 0,
                "steps_iterated" => 0,
                "best_hits"      => 0,
                "best_size"      => nothing,
                "solved"         => false,
                "timestamp_utc"  => Dates.format(now(UTC), dateformat"yyyy-mm-ddTHH:MM:SS\\Z")
            ))
            processed += 1
            processed >= limit_probs && break
            continue
        end

        CDepth = Grammar.frequencies_to_costs_depth(M_sum, rules; alpha=1.0, eps=1e-3)

        CDepth = permutedims(CDepth, (2,1)) # We transpose it here to match the way we use it in the depthiterator

        # exit()
        # println(CDepth)
        if !llm_enabled
            CDepth .= 1.0
        end
        # println(CDepth)
        @printf("\n[PID %4d | depth %2d] answers=%d (parsed=%d)\n",
                pid, depth, length(ans_paths), nvalid)
        println("  counts-matrix size: ", size(M_sum), " -> cost size: ", size(CDepth))

        if depth_aware_enabled
            # existing depth-aware iterator call 
            depth_state = ProbeUpdate.DepthProbeState(CDepth)  # baseline = the costs you just computed
            println(CDepth)
            # exit(0)
            res = run_depth_iter_for_problem(ds, pid, G, CDepth;
                max_cost=search_max_cost, max_depth=search_max_depth,
                max_size=search_max_size, jit_enabled=updating_enabled,
                traces_enabled=traces_enabled, depth_aware_enabled=true, probe_state=depth_state)
        else
            # --- flat costs from LLM frequencies ---
            Cflat = Grammar.frequencies_to_costs_flat(M_sum, rules)  # Vector{Float64} (R)
            if !llm_enabled
                Cflat .= 1.0
            end
            # 1-column adapter → depth-agnostic costs
            C1 = reshape(Cflat, :, 1)                                # (R × 1)
            flat_state = ProbeUpdate.FlatProbeState(Cflat)
            println(C1)
            # exit(0)
            # Pass C1 (not CDepth) to the iterator driver:
            res = run_depth_iter_for_problem(
                ds, pid, G, C1;                 
                max_cost=search_max_cost, max_depth=search_max_depth,
                max_size=search_max_size,
                jit_enabled=updating_enabled,
                traces_enabled=traces_enabled,
                depth_aware_enabled=false,
                flat_probe_state=flat_state
            )
        end

        println("SOLVED : ", res.solved)

        processed += 1
        if res.solved
            solved += 1
        end

        push!(results, Dict(
            "pid"            => pid,
            "problem_id"     => problem_id,
            "problem_hash"   => problem_hash,
            "depth"          => depth,
            "answers_total"  => length(ans_paths),
            "answers_parsed" => nvalid,
            "steps_iterated" => res.steps,
            "best_hits"      => res.best_hits,
            "best_size"      => res.best_size,
            "solved"         => res.solved,
            "timestamp_utc"  => Dates.format(now(UTC), dateformat"yyyy-mm-ddTHH:MM:SS\\Z")
        ))

        @printf("  search steps: %-8d  best_hits: %d  size: %d\n",
                res.steps, res.best_hits, res.best_size)

        processed >= limit_probs && break
    end

    println("\n=== Summary ===")
    println("Processed problems : ", processed)
    println("Fully solved       : ", solved)
    println("Timestamp          : ", Dates.format(now(UTC), dateformat"yyyy-mm-ddTHH:MM:SS\\Z"))

    # ---- Write JSON outputs ----
    # Per-problem: array of Dicts
    write_json(per_problem_path, results)

    # Summary: single Dict (+ config for reproducibility)
    summary = Dict(
        "processed"        => processed,
        "solved"           => solved,
        "dataset_path"     => dataset_path,
        "prompts_dir"      => prompts_dir,
        "runs_dir"         => runs_dir,
        "model_filter"     => model_filter,
        "nrows"            => nrows,
        "max_answers"      => max_answers,
        "limit"            => limit_probs,
        "llm_enabled"      => llm_enabled,
        "traces_enabled"   => traces_enabled,
        "updating_enabled" => updating_enabled,
        "search_max_cost"  => search_max_cost,
        "search_max_depth" => search_max_depth,
        "search_max_size"  => search_max_size,
        "timestamp_utc"    => Dates.format(now(UTC), dateformat"yyyy-mm-ddTHH:MM:SS\\Z")
    )
    write_json(summary_path, summary)

    println("\nWrote per-problem stats to: $(per_problem_path)")
    println("Wrote summary to           : $(summary_path)")
end

# =================== Entry ===================
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
