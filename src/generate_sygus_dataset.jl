#!/usr/bin/env julia

# -------------------------------------------------------------------
# SyGuS PBE (SLIA) dataset generator
#
# Usage:
#   julia generate_sygus_dataset.jl                 # writes sygus_dataset.jls
#   julia generate_sygus_dataset.jl mydata.jls      # custom output path
#
# This mirrors generate_string_dataset.jl but for the PBE_SLIA_Track_2019
# SyGuS string benchmarks, and includes the grammar for each problem.
# -------------------------------------------------------------------

include("PBE_SLIA_Track_2019.jl")
using .PBE_SLIA_Track_2019
using Serialization

# Lightweight representation of one I/O example for a SyGuS problem
struct SyGuSIOExample
    inputs::Dict{Symbol, Any}  # e.g. Dict(:_arg_1 => "AIX 5.1", :_arg_2 => 3)
    output::Any                # String / Int / Bool, depending on the benchmark
end

# One problem in the dataset: name, grammar name, grammar object, examples.
struct SyGuSProblemRecord
    name::String              # Problem name (e.g. "problem_11604909")
    grammar_name::String      # Grammar name (e.g. "grammar_11604909")
    grammar::Any              # The @cfgrammar object (HerbGrammar CF grammar)
    examples::Vector{SyGuSIOExample}
end

# Top-level container (analogous to StringDataset and your Karel dataset)
struct SyGuSDataset
    problems::Vector{SyGuSProblemRecord}
end

"""
    extract_examples(prob, ProblemType, IOExampleType) -> Vector{IOExampleType}

Generic helper: look through all fields of `prob` and return the one that is
a vector of `IOExample`s. We do this so we don't depend on the internal
field name inside `Problem` (same trick as in generate_string_dataset.jl).
"""
function extract_examples(prob, ProblemType, IOExampleType)
    for fld in fieldnames(ProblemType)
        v = getfield(prob, fld)
        if v isa AbstractVector
            if !isempty(v)
                first_v = first(v)
                if first_v isa IOExampleType
                    return v
                end
            else
                # empty vector but of the right element type?
                T = Base.eltype(v)
                if T === IOExampleType
                    return v
                end
            end
        end
    end
    error("Could not find a Vector{$(IOExampleType)} field inside Problem $(prob).")
end

"""
    grammar_for_problem_name(name::String) -> (grammar_name::String, grammar::Any)

Given a Problem name like "problem_11604909", look up the corresponding
grammar in the PBE_SLIA_Track_2019 module, assuming the convention that
"problem_" is replaced by "grammar_".
"""
function grammar_for_problem_name(name::String)
    startswith(name, "problem_") || error("Unexpected problem name: $name")
    gsym = Symbol(replace(name, "problem_" => "grammar_"))
    if !isdefined(PBE_SLIA_Track_2019, gsym)
        error("No grammar $(String(gsym)) found for problem $name")
    end
    g = getfield(PBE_SLIA_Track_2019, gsym)
    return String(gsym), g
end

"""
    collect_all_sygus_problems() :: Vector{SyGuSProblemRecord}

Scan the `PBE_SLIA_Track_2019` module for all values of type `Problem`,
extract their `IOExample`s and corresponding grammars, and convert them
into `SyGuSProblemRecord`s.
"""
function collect_all_sygus_problems()
    Mod = PBE_SLIA_Track_2019
    ProblemType   = Mod.Problem
    IOExampleType = Mod.IOExample

    records = SyGuSProblemRecord[]

    # Go over all names defined in the PBE_SLIA_Track_2019 module
    for name in names(Mod; all = true)
        # Skip the type names etc.
        name === :Problem   && continue
        name === :IOExample && continue

        value = try
            getfield(Mod, name)
        catch
            # Some bindings may fail; ignore them
            continue
        end

        # Only keep actual benchmark problems
        value isa ProblemType || continue
        prob = value

        # Find the IOExample vector inside this Problem (field-agnostic)
        io_vec = extract_examples(prob, ProblemType, IOExampleType)

        # Look up the matching grammar
        grammar_name, grammar = grammar_for_problem_name(prob.name)

        exs = SyGuSIOExample[]
        for ex in io_vec
            # ex.in is Dict{Symbol,Any}, ex.out is Any
            # We keep them as-is to preserve multi-arg + typed outputs.
            push!(exs, SyGuSIOExample(ex.in, ex.out))
        end

        push!(records, SyGuSProblemRecord(prob.name, grammar_name, grammar, exs))
    end

    # Sort by name for determinism
    sort!(records; by = r -> r.name)
    return records
end

"""
    write_sygus_dataset(outpath::AbstractString = "sygus_dataset.jls")

Build the dataset from all SyGuS problems and serialize it to `outpath`.
"""
function write_sygus_dataset(outpath::AbstractString = "sygus_dataset.jls";
                             max_problems::Int = typemax(Int))
    probs = collect_all_sygus_problems()

    # Optionally limit number of problems (for debugging, etc.)
    if length(probs) > max_problems
        probs = probs[1:max_problems]
    end

    ds = SyGuSDataset(probs)

    open(outpath, "w") do io
        serialize(io, ds)
    end

    println("Wrote $(length(probs)) SyGuS problems to \"", outpath, "\".")
end


# -------------------------------------------------------------------
# CLI entrypoint
# -------------------------------------------------------------------

if abspath(PROGRAM_FILE) == @__FILE__
    outpath = length(ARGS) >= 1 ? ARGS[1] : "sygus_dataset.jls"
    write_sygus_dataset(outpath)
end
