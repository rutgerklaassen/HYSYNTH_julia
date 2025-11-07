#!/usr/bin/env julia

using Serialization
using HerbBenchmarks.Karel_2018
using HerbCore
using HerbSpecification
using HerbGrammar
using HerbConstraints
using REPL.TerminalMenus
using Printf

# Where to save curated picks (change if you like)
const CURATED_PATH = "karel_curated_dataset.jls"

# ---- helpers ---------------------------------------------------------------
# Build an index of programs per depth; returns (depths_sorted, Dict depth=>Vector)
function group_programs_by_depth(progs)
    T = eltype(progs)  # usually NamedTuple{...}
    bydepth = Dict{Int, Vector{T}}()
    for rec in progs
        d = rec.depth
        v = get!(bydepth, d, Vector{T}())  # <-- make an empty Vector of the right element type
        push!(v, rec)
    end
    depths = sort!(collect(keys(bydepth)))
    return depths, bydepth
end

# Count "intermediate states" for a single world of a program record.
# Defined as states strictly between initial and final: max(length(exec_path) - 2, 0)
function intermediate_states_for_world(rec, world_idx::Int)
    tr = rec.traces[world_idx]
    isempty(tr.exec_path) && return 0
    return max(length(tr.exec_path) - 2, 0)
end

# Group programs by their number of intermediate states on a specific world (default: world 1).
# Returns (sorted_counts, Dict{Int, Vector{T}})
function group_programs_by_intermediates(progs; world_idx::Int=1)
    T = eltype(progs)
    bycount = Dict{Int, Vector{T}}()
    for rec in progs
        local widx = (world_idx <= length(rec.worlds)) ? world_idx : 1
        cnt = intermediate_states_for_world(rec, widx)
        v = get!(bycount, cnt, Vector{T}())
        push!(v, rec)
    end
    counts = sort!(collect(keys(bycount)))
    return counts, bycount
end

# Load fixed dataset file; tries .jls, then .jl
function load_fixed_dataset(path_primary::AbstractString="karel_dataset_by_depth.jls";
                            path_alt::AbstractString="karel_dataset_by_depth.jl")
    path =
        isfile(path_primary) ? path_primary :
        isfile(path_alt)     ? path_alt     :
        error("Dataset not found. Looked for '$path_primary' and '$path_alt' in current directory.")
    ds = safe_deserialize(path)
    hasproperty(ds, :programs) || error("The file '$path' doesn't have a 'programs' field.")
    progs = ds.programs
    isempty(progs) && error("No programs found in '$path'.")
    return progs
end

function pick_from_menu(title::AbstractString, items::Vector{String})
    isempty(items) && error("No items to choose from.")
    menu = RadioMenu(items; pagesize=min(length(items), 10))
    choice = request(title, menu)
    choice === nothing && error("Selection cancelled.")
    return Int(choice)
end

# Show full multi-line program text (handles "\n" escapes)
pretty_prog_text(s) = Base.unescape_string(String(s))

function safe_deserialize(path::AbstractString)
    try
        return deserialize(path)
    catch e
        error("Failed to deserialize '$path': $(e)")
    end
end

# ---- export utilities ------------------------------------------------------

# Safe load of curated dataset; returns Vector{Any} of program records (or empty)
function load_curated_programs(path::AbstractString)::Vector{Any}
    if !isfile(path)
        return Any[]
    end
    ds = safe_deserialize(path)
    if hasproperty(ds, :programs)
        return Vector{Any}(ds.programs)
    else
        # If someone saved a plain Vector before, accept it too.
        return ds isa Vector ? Vector{Any}(ds) : Any[]
    end
end

# Save curated vector back in a struct with a `.programs` field for compatibility
save_curated_programs!(programs::Vector; path::AbstractString=CURATED_PATH) = serialize(path, (; programs))

# Normalize program text for consistent deduping (pretty-printed, no trailing ws)
normalize_prog_text(x) = strip(Base.unescape_string(String(x)))

# Export a single record; dedupe by normalized program text
function export_program!(rec; path::AbstractString=CURATED_PATH)
    curated = load_curated_programs(path)
    target = normalize_prog_text(rec.program)

    # NOTE: use 2-arg getfield/getproperty; no "default" arg
    is_dup = any(r -> begin
        # tolerate either NamedTuple/struct with :program or raw String entries
        if r isa AbstractString
            normalize_prog_text(r) == target
        elseif hasproperty(r, :program)
            normalize_prog_text(getproperty(r, :program)) == target
        elseif r isa NamedTuple && :program in propertynames(r)
            normalize_prog_text(getfield(r, :program)) == target
        else
            false
        end
    end, curated)

    if is_dup
        println("\n⚠ Already in curated dataset (same program text). Path: $path")
        return :skipped
    end

    push!(curated, rec)
    save_curated_programs!(curated; path=path)
    println("\n✅ Exported to curated dataset: $path")
    return :exported
end

# --- deletion helpers -------------------------------------------------------

normalize_prog_text(x) = strip(Base.unescape_string(String(x)))  # already defined above

# true if a record/string matches the given program text (normalized)
function same_prog_text(r, target_norm::AbstractString)
    if r isa AbstractString
        return normalize_prog_text(r) == target_norm
    elseif hasproperty(r, :program)
        return normalize_prog_text(getproperty(r, :program)) == target_norm
    elseif r isa NamedTuple && :program in propertynames(r)
        return normalize_prog_text(getfield(r, :program)) == target_norm
    else
        return false
    end
end

# Delete from curated file; returns :deleted or :not_found
function delete_from_curated_by_text!(prog_text::AbstractString; path::AbstractString=CURATED_PATH)
    curated = load_curated_programs(path)
    isempty(curated) && return :not_found

    target = normalize_prog_text(prog_text)
    keep = [r for r in curated if !same_prog_text(r, target)]
    if length(keep) == length(curated)
        return :not_found
    end
    save_curated_programs!(keep; path=path)
    return :deleted
end

# Delete from a dataset file (the one opened with --dataset/--alt).
# Tries to preserve :target and :max_depth metadata if present.
function delete_from_dataset_by_text!(dataset_path::AbstractString, prog_text::AbstractString)
    ds = safe_deserialize(dataset_path)
    hasproperty(ds, :programs) || error("Dataset '$dataset_path' has no :programs field.")
    progs = Vector{Any}(ds.programs)

    target = normalize_prog_text(prog_text)
    keep = [r for r in progs if !same_prog_text(r, target)]
    if length(keep) == length(progs)
        return :not_found
    end

    # preserve metadata if present
    target_meta    = hasproperty(ds, :target)    ? getproperty(ds, :target)    : nothing
    max_depth_meta = hasproperty(ds, :max_depth) ? getproperty(ds, :max_depth) : nothing

    out_obj = if target_meta === nothing && max_depth_meta === nothing
        (; programs = keep)
    elseif target_meta === nothing
        (; max_depth = max_depth_meta, programs = keep)
    elseif max_depth_meta === nothing
        (; target = target_meta, programs = keep)
    else
        (; target = target_meta, max_depth = max_depth_meta, programs = keep)
    end

    serialize(dataset_path, out_obj)
    return :deleted
end

# simple yes/no prompt (default = No)
function confirm(prompt::AbstractString)
    print(prompt * " [y/N]: ")
    ans = lowercase(strip(readline(stdin)))
    return ans in ("y","yes")
end


function print_world(rec, world_idx::Int)
    nw = length(rec.worlds)
    println("\n================= SELECTED PROGRAM =================")
    println(pretty_prog_text(rec.program))
    println("\n-- WORLD $(world_idx) / $(nw) --")

    tr = rec.traces[world_idx]
    if isempty(tr.exec_path)
        println("(Empty trace)\n")
        return
    end

    # Helper to print markers + bag size
    function print_markers_and_bag(state)
        println("Bag size: ", state.hero.marker_count)
        if isempty(state.markers)
            println("Markers: (none)")
        else
            println("Markers: ", join([string(pos, "→", cnt) for (pos,cnt) in state.markers], ", "))
        end
    end

    # Initial state
    println("\nInput state:")
    println(tr.exec_path[1])
    print_markers_and_bag(tr.exec_path[1])

    println("\nTRACES:")
    for (t, s) in enumerate(tr.exec_path)
        println("\nt=$(t-1)")
        println(s)
        print_markers_and_bag(s)
    end
    println()
end

# ---- program catalog (multi-line + pagination) -----------------------------

function print_program_page(progs; page::Int=1, per_page::Int=10)
    total = length(progs)
    n_pages = max(1, cld(total, per_page))
    p = clamp(page, 1, n_pages)
    first_idx = (p-1)*per_page + 1
    last_idx  = min(p*per_page, total)

    println("\n========== Programs (page $p / $n_pages) ==========\n")
    for i in first_idx:last_idx
        rec = progs[i]
        println(@sprintf("[%3d]", i))
        println(pretty_prog_text(rec.program))
        println("-"^60)
    end
    println("Commands: enter a program number; or n=next page, p=prev page, b=back")
end

function prompt_program_index(progs)
    per_page = 10
    page = 1
    total = length(progs)

    while true
        print_program_page(progs; page=page, per_page=per_page)
        print("> ")
        input = readline(stdin)

        if isempty(input)
            continue
        end

        if input in ("b", "B")
            return :back
        elseif input in ("n", "N")
            page = min(page + 1, max(1, cld(total, per_page)))
            continue
        elseif input in ("p", "P")
            page = max(page - 1, 1)
            continue
        end

        # try number
        idx = tryparse(Int, input)
        if idx === nothing || idx < 1 || idx > total
            println("Invalid choice: '$input'. Please type a program number, or n/p/b.")
            continue
        end
        return idx
    end
end

# ---- main flow -------------------------------------------------------------
# ---- CLI args --------------------------------------------------------------
"""
parse_cli_args(ARGS) -> (dataset, alt, curated)
Flags:
  --dataset PATH     dataset to open (default: karel_dataset_by_depth.jls)
  --alt PATH         fallback dataset (default: karel_dataset_by_depth.jl)
  --curated PATH     curated output file (default: karel_curated_dataset.jls)
Also supports: -d PATH, -c PATH, and two positionals: DATASET [CURATED]
"""
function parse_cli_args(args::Vector{String})
    dataset = "karel_dataset_by_depth.jls"
    alt     = "karel_dataset_by_depth.jl"
    curated = CURATED_PATH  # keep your const as default

    i = 1
    while i <= length(args)
        a = args[i]
        if a in ("-h","--help")
            println("""
Usage:
  julia view_dataset.jl [--dataset PATH] [--alt PATH] [--curated PATH]
  julia view_dataset.jl [DATASET_PATH] [CURATED_PATH]

Flags:
  --dataset, -d   dataset to open        (default: $dataset)
  --alt           fallback dataset       (default: $alt)
  --curated, -c   curated output file    (default: $curated)
  -h, --help      show this help
""")
            exit(0)
        elseif a in ("--dataset","-d")
            i += 1; @assert i <= length(args) "Missing value after $a"; dataset = args[i]
        elseif a == "--alt"
            i += 1; @assert i <= length(args) "Missing value after $a"; alt = args[i]
        elseif a in ("--curated","-c")
            i += 1; @assert i <= length(args) "Missing value after $a"; curated = args[i]
        elseif startswith(a, "--dataset=")
            dataset = split(a, "=", limit=2)[2]
        elseif startswith(a, "--alt=")
            alt = split(a, "=", limit=2)[2]
        elseif startswith(a, "--curated=")
            curated = split(a, "=", limit=2)[2]
        elseif !startswith(a, "--")
            # positionals: 1st -> dataset, 2nd -> curated
            if dataset == "karel_dataset_by_depth.jls"
                dataset = a
            elseif curated == CURATED_PATH
                curated = a
            else
                @warn "Unrecognized extra argument: $a"
            end
        else
            @warn "Unrecognized flag: $a"
        end
        i += 1
    end
    return (dataset, alt, curated)
end

function main()
    # Parse CLI options
    dataset, alt, curated_path = parse_cli_args(ARGS)

    # 1) Load the chosen dataset
    progs = load_fixed_dataset(dataset; path_alt=alt)

    # 2) Index by depth
    depths, bydepth = group_programs_by_depth(progs)

    # 3) Depth selection (no "back" here)
    while true
        depth_labels = [@sprintf("Depth %d  (%d programs)", d, length(bydepth[d])) for d in depths]
        dsel = pick_from_menu("Select a depth:", depth_labels)
        chosen_depth = depths[dsel]
        progs_at_depth = bydepth[chosen_depth]

        # 4) Intermediate-states bucket selection (based on world 1)
        while true
            counts, bycount = group_programs_by_intermediates(progs_at_depth; world_idx=1)
            count_menu = ["◀ Back to depth selection"]
            append!(count_menu, [@sprintf("%d states  (%d programs)", c, length(bycount[c])) for c in counts])
            csel = pick_from_menu("Select # intermediate states (based on World 1):", count_menu)

            if csel == 1
                # Back to depth selection
                break
            end

            chosen_count = counts[csel - 1]
            filtered = bycount[chosen_count]

            # 5) Program catalog (with back to the "states" menu)
            while true
                choice = prompt_program_index(filtered)
                if choice === :back
                    # go back to "states" menu for this depth
                    break
                end
                pidx = choice::Int
                rec = filtered[pidx]

                # 6) World menu loop (with back to program menu)
                nw = length(rec.worlds)
                world_labels = [@sprintf("World %d", i) for i in 1:nw]
                while true
                    world_menu = [
                        "◀ Back to program list",
                        "Export this program to curated dataset",
                        "Delete this program from curated dataset",
                        "Delete this program from SOURCE dataset",
                        world_labels...
                    ]

                    wsel = pick_from_menu("Select an action/world:", world_menu)
                    if wsel == 1
                        # back to program list
                        break
                    elseif wsel == 2
                        export_program!(rec; path=CURATED_PATH)
                        # stay here
                    elseif wsel == 3
                        if confirm("Remove from curated dataset?")
                            status = delete_from_curated_by_text!(rec.program; path=CURATED_PATH)
                            println(status == :deleted ? "Deleted from curated dataset." : "Not found in curated dataset.")
                        end
                    elseif wsel == 4
                        # delete from the dataset currently opened (dataset or alt)
                        # 'dataset' comes from parse_cli_args; thread it down by capturing in a closure
                        if confirm("Remove from SOURCE dataset?")
                            status = delete_from_dataset_by_text!(dataset, rec.program)
                            println(status == :deleted ? "Deleted from source dataset." : "Not found in source dataset.")
                            if status == :deleted
                                # Also remove from the in-memory 'progs' so the UI stays consistent
                                progs = load_fixed_dataset(dataset; path_alt=alt)
                                depths, bydepth = group_programs_by_depth(progs)
                                # jump back up to refresh the lists
                                break
                            end
                        end
                    else
                        widx = wsel - 4  # adjust because we inserted 3 extra actions
                        print_world(rec, widx)
                    end
                end
            end
        end
    end
end

# ---- run -------------------------------------------------------------------
main()
