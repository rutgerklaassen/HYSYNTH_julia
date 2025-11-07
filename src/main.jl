#!/usr/bin/env julia

# =================== Includes (your local modules) ===================
# Make sure these files are in the same folder or adjust the paths below.
include("parseKarel.jl")
include("grammar.jl")
include("hysynth.jl")
include("utils.jl")
include("depth_based_synthesis.jl")
include("karel_utils.jl")

using .ParseKarel
using .Grammar
using .Hysynth
using .Depth_synthesis
using .Utils
using .KarelUtils

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
    args["--dataset"] = get(ENV, "KAREL_DATASET", "karel_dataset_by_depth.jls")
    args["--prompts-dir"] = get(ENV, "PROMPTS_DIR", "prompts")
    args["--runs-dir"] = get(ENV, "RUNS_DIR", "runs")
    args["--model"] = get(ENV, "MODEL_FILTER", "")             # optional: only use runs for this model
    args["--max-answers"] = get(ENV, "MAX_ANSWERS", "")        # optional: cap answers per problem
    args["--nrows"] = get(ENV, "DEPTH_ROWS", "6")              # rows = depth buckets for counts_matrix
    args["--limit"] = get(ENV, "LIMIT", "")                    # optional: max problems to process

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

# For a set of trees, accumulate counts into a *single numeric matrix*:
# We avoid depending on internal layout of counts_by_depth by converting
# each tree's counts into a matrix via Utils.counts_matrix(...), then summing.
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
            # Basic sanity: rules order should be identical; if not, we could align,
            # but Utils.counts_matrix for the same grammar should be stable.
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
    # collapse whitespace & trim; tweak if you want stricter/looser canon
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

# returns (frozen_node_or_nothing, ok::Bool)
safe_freeze(p) = try
    (freeze_state(p), true)
catch e
    if e isa AssertionError
        (nothing, false)   # <- ALWAYS a 2-tuple
    else
        rethrow()
    end
end

 prog_fingerprint(x)::UInt64 = prog_fingerprint(repr(x))
# =================== Synthesis runner (your new iterator) ===================

# Evaluate a single problem (by pid) with a cost matrix:
function run_depth_iter_for_problem(ds, pid::Int, G::AbstractGrammar, C::Matrix{Float64};
                                    max_cost=1e9, max_depth=10, max_size=50, jit_enabled=false)

    rec  = ds.programs[pid]
    prob = Problem("karel-traces", rec.traces)
    # println(prob)
    it = DepthCostBasedBottomUpIterator(G, :Start, C;
        bank=HerbSearch.CostBank(),
        max_cost=max_cost, max_depth=max_depth, max_size=max_size,
        jit_enabled=jit_enabled   
    )
    best_hits = 0
    best_size = typemax(Int)
    best_extra = 0
    best_prog = nothing
    solved = false
    reset_variables = Depth_synthesis.get_reset_variables(it)
    st = nothing
    nxt = iterate(it)
    steps = 0
    seen = Set{UInt64}()   # note the parentheses!
    saw = 0
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
        expr = rulenode2expr(fpnode, G)
        fp   = prog_fingerprint(expr)
        if fp ∈ seen
            saw += 1
            nxt = iterate(it, st)   # <-- advance the iterator!
            continue # Already saw this program
        else
            push!(seen, fp)
        end

        

        hits, total, extra = KarelUtils.count_matches(fpnode, prob, G)
        strict_ok = KarelUtils.satisfies_problem_strict(fpnode, prob, G)

        sz = program_rule_count(p)
        # if steps % 500 == 0
        #     println(steps)
        #     println(sz)
        # end
        if strict_ok
            println("program ", rulenode2expr(fpnode, G), "(",fpnode,")", " is an exact hit with size ", sz)
            solved = true
            return (; steps, best_hits, best_size, best_prog, solved)
        end
        improved = (hits > best_hits) || (hits == best_hits && extra < best_extra) || (hits == best_hits && extra == best_extra && sz < best_size)
        if improved
            best_hits = hits; best_size = sz; best_prog = p; best_extra = extra;
            println("program ", rulenode2expr(freeze_state(p), G), "(",fpnode,")", " is a new best hit! ", best_hits , " out of ", total, " with ",extra," extra steps ", " and size ", sz, " and depth ", Depth_synthesis.tree_depth(p))
            apply_probe_update!(it, p, hits, total)
            reset_variables.depth_exhausted = true
            # if hits == total
            #     return (; steps, best_hits, best_size, best_prog)
            # end
        end
        nxt = iterate(it, st)
    end
    return (; steps, best_hits, best_size, best_prog, solved)
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
        JSON.print(io, obj;)  # pretty-printed JSON
    end
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

    # Output paths (JSON, not JSONL)
    out_dir      = joinpath(manifest_base, "runs")
    isdir(out_dir) || mkpath(out_dir)
    per_problem  = joinpath(out_dir, "synthesis_stats.json")
    summary_path = joinpath(out_dir, "synthesis_summary.json")

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
            # still record a row so the JSON mirrors the dataset coverage
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
                txt = read(ap, String)
                t = try_parse_answer(txt)
                if t === nothing
                    baddir = joinpath(runs_dir, "bad_answers")
                    isdir(baddir) || mkpath(baddir)
                    outname = @sprintf("pid_%04d_sample_%04d.txt", pid, taken)
                    open(joinpath(baddir, outname), "w") do io
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

        C = Grammar.frequencies_to_costs(M_sum, rules; alpha=1.0, eps=1e-3)
        C = permutedims(C, (2,1))
        
        @printf("\n[PID %4d | depth %2d] answers=%d (parsed=%d)\n",
                pid, depth, length(ans_paths), nvalid)
        println("  counts-matrix size: ", size(M_sum), " -> cost size: ", size(C))

        # Run iterator
        res = run_depth_iter_for_problem(ds, pid, G, C;
            max_cost=1e9, max_depth=8, max_size=10000, jit_enabled=true
        )
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
    write_json(per_problem, results)

    # Summary: single Dict
    summary = Dict(
        "processed"      => processed,
        "solved"         => solved,
        "dataset_path"   => dataset_path,
        "prompts_dir"    => prompts_dir,
        "runs_dir"       => runs_dir,
        "model_filter"   => model_filter,
        "nrows"          => nrows,
        "max_answers"    => max_answers,
        "limit"          => limit_probs,
        "timestamp_utc"  => Dates.format(now(UTC), dateformat"yyyy-mm-ddTHH:MM:SS\\Z")
    )
    write_json(summary_path, summary)

    println("\nWrote per-problem stats to: $(per_problem)")
    println("Wrote summary to           : $(summary_path)")
end


# =================== Entry ===================
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
