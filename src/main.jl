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

# =================== Synthesis runner (your new iterator) ===================

# Evaluate a single problem (by pid) with a cost matrix:
function run_depth_iter_for_problem(ds, pid::Int, G::AbstractGrammar, C::Matrix{Float64};
                                    max_cost=1e9, max_depth=10, max_size=50, jit_enabled=false)

    rec  = ds.programs[pid]
    prob = Problem("karel-traces", rec.traces)
    println("\n\n\n\n\n\n\n\n\n\n\nStarting to find program : ")
    # println(prob)
    it = DepthCostBasedBottomUpIterator(G, :Start, C;
        bank=HerbSearch.CostBank(),
        max_cost=max_cost, max_depth=max_depth, max_size=max_size,
        jit_enabled=jit_enabled   
    )
    best_hits = 0
    best_size = typemax(Int)
    best_prog = nothing
    solved = false

    st = nothing
    nxt = iterate(it)
    steps = 0
    while nxt !== nothing
        steps += 1
        (p, st) = nxt
        hits, total = KarelUtils.count_matches(p, prob, G)
        strict_ok = KarelUtils.satisfies_problem_strict(p, prob, G)

        sz = program_rule_count(p)
        # if steps % 500 == 0
        #     println(steps)
        #     println(sz)
        # end
        if strict_ok
            println("program ", rulenode2expr(freeze_state(p), G), "(",freeze_state(p),")", " is an exact hit with size ", sz)
            solved = true
            return (; steps, best_hits, best_size, best_prog, solved)
        end
        improved = (hits > best_hits) || (hits == best_hits && sz < best_size)
        if improved
            best_hits = hits; best_size = sz; best_prog = p
            println("program ", rulenode2expr(freeze_state(p), G), "(",freeze_state(p),")", " is a new best hit! ", best_hits , " out of ", total, " with size ", sz)
            apply_probe_update!(it, p, hits, total)
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

    println("=== Build PCFGs from answers & run depth-based synthesis ===")
    println("dataset     : $dataset_path")
    println("prompts_dir : $prompts_dir")
    println("runs_dir    : $runs_dir")
    println("model_filter: $(isempty(model_filter) ? "<any>" : model_filter)")
    println("nrows(depth): $nrows")

    # Load dataset
    ds = Serialization.deserialize(dataset_path)
    nprogs = length(ds.programs)
    println("Loaded dataset with $nprogs programs.")
    # Load prompt index maps
    by_file, by_pid_map, by_hash = load_prompt_index(prompts_dir)

    # Collect all run rows (optionally filtered by model) and map to problem_id -> answers
    run_rows = collect_run_rows(runs_dir; model_filter=model_filter)
    manifest_base = realpath(joinpath(prompts_dir, ".."))  # project root: parent of prompts/
    answers_by_hash = map_answers_by_hash(run_rows; base=manifest_base)

    # Grammar to use everywhere
    G = KarelUtils.grammar_karel

    processed = 0
    solved = 0
    first = 0
    # Iterate problems in the dataset
    for (pid, rec) in enumerate(ds.programs)
        # Their problem_id keys look like: karel_by_depth::<sha8>::p0001
        # Find any prompt index entry that has this pid and build problem_id strings.
        # Prompt index JSONL already contains "problem_id" per pid.
        # We'll locate the record by scanning (pid match) once for speed.

        # find the index record for this pid (O(n) the first time; acceptable)
        idx_rec = nothing
        for (_, recidx) in by_pid_map
            # by_pid_map keyed by problem_id; values contain pid
            # Be tolerant to Int/Number vs String
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
        ans_paths = get(answers_by_hash, problem_hash, String[])
        print(ans_paths)
        problem_id = String(idx_rec["problem_id"])
        depth      = Int(idx_rec["depth"])
        # ans_paths  = get(answers_by_problem, problem_id, String[])

        if isempty(ans_paths)
            @info "No answers for problem" pid problem_id depth
            continue
        end

        # Read all answers (up to max_answers) and parse
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
            continue
        end

        # Accumulate matrices over trees
        M_sum, rules, nvalid = accumulate_counts_matrix(trees; G=G, nrows=nrows)
        if M_sum === nothing
            @info "Could not form counts matrix" pid
            continue
        end

        # Convert frequencies to costs
        C = Grammar.frequencies_to_costs(M_sum, rules; alpha=1.0, eps=1e-3)
        C = permutedims(C, (2,1)) # Transpose because I built the other thing with different columns /

        # Pretty-print a tiny summary (optional)
        @printf("\n[PID %4d | depth %2d] answers=%d (parsed=%d)\n",
                pid, depth, length(ans_paths), nvalid)
        println("  counts-matrix size: ", size(M_sum), " -> cost size: ", size(C))

        # Run your new iterator for this problem
        res = run_depth_iter_for_problem(ds, pid, G, C;
            max_cost=1e9, max_depth=8, max_size=10000, jit_enabled=true
        )
        println(res.solved)

        processed += 1
        @printf("  search steps: %-8d  best_hits: %d  size: %d\n",
                res.steps, res.best_hits, res.best_size)
        if res.solved
            solved += 1
        end
        # exit()
        # Optional: stop after limit for quick sanity checks
        processed >= limit_probs && break
    end
    print(length(all_answers))
    println("\n=== Summary ===")
    println("Processed problems : ", processed)
    println("Fully solved       : ", solved)
    println("Timestamp          : ", Dates.format(now(UTC), dateformat"yyyy-mm-ddTHH:MM:SS\Z"))
end

# =================== Entry ===================
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
