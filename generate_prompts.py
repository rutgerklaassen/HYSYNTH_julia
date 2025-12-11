#!/usr/bin/env python3
"""
Generate LLM prompts for the SyGuS dataset.

Usage:
    python generate_prompts.py \
        --dataset path/to/sygus_dataset.jls \
        --prompt prompt.txt \
        --outdir prompts/ \
        --prefix sygus

The prompt template must contain:
    {GRAMMAR_BLOCK}
    {EXAMPLES_BLOCK}
"""

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

# Adjust this if you ever move the repo
ABS_SRC_DIR = Path(
    "/home/rutger/Desktop/Thesis/LLM/SYGUS/HYSYNTH_julia/src"
).resolve()

# ---------------------------------------------------------------------
# Embedded Julia dumper for SyGuSDataset
# ---------------------------------------------------------------------

JULIA_DUMPER = r"""
try
    using JSON
catch
    import Pkg
    Pkg.add("JSON")
    using JSON
end

using Serialization

# Types must match generate_sygus_dataset.jl
struct SyGuSIOExample
    inputs::Dict{Symbol, Any}
    output::Any
end

struct SyGuSProblemRecord
    name::String
    grammar_name::String
    grammar::Any
    examples::Vector{SyGuSIOExample}
end

struct SyGuSDataset
    problems::Vector{SyGuSProblemRecord}
end

function safe_deserialize(path::AbstractString)
    try
        return deserialize(path)
    catch e
        error("Failed to deserialize '$(path)': $(e)")
    end
end

function grammar_to_string(g)
    # g is a HerbGrammar.ContextSensitiveGrammar
    syms   = getfield(g, 1)  # Vector{Any} of symbols/term constructors
    lhsvec = getfield(g, 2)  # Vector{Union{Nothing,Symbol}}: LHS nonterminal per rule
    rhsvec = getfield(g, 7)  # Vector{Vector}: RHS "argument" nonterminals per rule

    # Heuristic: functions/terminals are exactly the "syms" beyond the first few atoms.
    # Rule i uses symbol syms[i] as its "head", lhsvec[i] as LHS nonterminal,
    # and rhsvec[i] as the argument nonterminals.
    io = IOBuffer()

    # Collect nonterminals that actually appear on the LHS
    nts = collect(unique(filter(!isnothing, lhsvec)))
    println(io, "Nonterminals:")
    println(io, "  ", join(string.(nts), ", "))
    println(io)
    println(io, "Start:")
    println(io, "  Start -> ntString")  # these SyGuS PBE grammars always synthesize ntString

    # Group rules by LHS nonterminal
    for nt in nts
        println(io)
        println(io, string(nt), ":")
        for (i, lhs) in enumerate(lhsvec)
            lhs === nt || continue

            head = syms[i]
            args = rhsvec[i]

            # Turn head into a nice string
            head_str = head isa Symbol ? string(head) : sprint(show, head)

            rhs_str = if isempty(args)
                head_str
            else
                string(head_str, "(", join(string.(args), ", "), ")")
            end

            println(io, "  ", string(nt), " -> ", rhs_str)
        end
    end

    return String(take!(io))
end

function main(dataset_path::AbstractString, src_dir::AbstractString)
    # Load grammar definitions and HerbGrammar types used inside the grammar objects
    include(joinpath(src_dir, "PBE_SLIA_Track_2019.jl"))

    ds = safe_deserialize(dataset_path)::SyGuSDataset

    for (pid, prob) in enumerate(ds.problems)
        exs = Any[]
        for ex in prob.examples
            d = Dict{String, Any}()
            for (k, v) in ex.inputs
                d[string(k)] = v
            end
            push!(exs, Dict("inputs" => d, "output" => ex.output))
        end

        JSON.print(Dict(
            "pid" => pid,
            "name" => prob.name,
            "grammar_name" => prob.grammar_name,
            "grammar_str" => grammar_to_string(prob.grammar),
            "examples" => exs,
        ))
        println()
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) < 2
        error("Usage: julia dump_sygus.jl DATASET.jls /abs/path/to/src")
    end
    main(ARGS[1], ARGS[2])
end
"""

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def run_julia_dump(dataset_path: Path) -> List[Dict[str, Any]]:
    """Run the embedded Julia dumper and parse JSONL output."""
    if not ABS_SRC_DIR.is_dir():
        raise SystemExit(f"ABS_SRC_DIR does not exist: {ABS_SRC_DIR}")

    with tempfile.TemporaryDirectory() as td:
        dump_path = Path(td) / "dump_sygus.jl"
        dump_path.write_text(JULIA_DUMPER, encoding="utf-8")

        proc = subprocess.run(
            ["julia", str(dump_path), str(dataset_path), str(ABS_SRC_DIR)],
            check=False,
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            sys.stderr.write("Julia dumper stderr:\n")
            sys.stderr.write(proc.stderr)
            raise RuntimeError(
                f"Julia SyGuS dump failed (code {proc.returncode}). "
                f"Is the dataset path correct? ({dataset_path})"
            )

        rows: List[Dict[str, Any]] = []
        for line in proc.stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
        return rows


def build_examples_block(examples: List[Dict[str, Any]]) -> str:
    """Format {EXAMPLES_BLOCK} for SyGuS with multi-arg inputs."""
    blocks: List[str] = []
    for i, ex in enumerate(examples, start=1):
        inputs = ex["inputs"]
        output = ex["output"]

        lines: List[str] = []
        lines.append(f"EXAMPLE {i}")
        lines.append("Inputs:")
        for key in sorted(inputs.keys()):
            val = inputs[key]
            val_str = json.dumps(val)
            lines.append(f"  {key} = {val_str}")
        lines.append(f"Output: {json.dumps(output)}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def _to_plain(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_plain(v) for v in obj]
    return obj


def compute_problem_hash(grammar_str: str, examples: List[Dict[str, Any]]) -> str:
    """Stable short hash over (grammar, examples) for matching prompts/answers."""
    payload = {
        "grammar": (grammar_str or "").strip(),
        "worlds": [_to_plain(e) for e in examples],
    }
    s = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]


@dataclass
class Args:
    dataset: Path
    prompt: Path
    outdir: Path
    prefix: str
    limit: int


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def parse_args() -> Args:
    ap = argparse.ArgumentParser(description="Generate prompts for SyGuS dataset.")
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--prompt", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--prefix", type=str, default="sygus")
    ap.add_argument("--limit", type=int, default=0)
    ns = ap.parse_args()
    return Args(
        dataset=ns.dataset,
        prompt=ns.prompt,
        outdir=ns.outdir,
        prefix=ns.prefix,
        limit=ns.limit,
    )


def main() -> None:
    args = parse_args()

    dataset_path = args.dataset.resolve()
    if not dataset_path.is_file():
        raise SystemExit(f"Dataset file not found: {dataset_path}")

    template = args.prompt.read_text(encoding="utf-8")
    if "{GRAMMAR_BLOCK}" not in template or "{EXAMPLES_BLOCK}" not in template:
        raise SystemExit(
            "Prompt template must contain both {GRAMMAR_BLOCK} and {EXAMPLES_BLOCK}."
        )

    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    index_path = outdir / "index.jsonl"

    try:
        ds_bytes = dataset_path.read_bytes()
        dataset_sha8 = hashlib.sha1(ds_bytes).hexdigest()[:8]
    except Exception:
        dataset_sha8 = "nosha"

    print(f"Loading SyGuS dataset from: {dataset_path}")
    rows = run_julia_dump(dataset_path)
    print(f"Found {len(rows)} SyGuS problems.")

    index_records: List[Dict[str, Any]] = []
    written = 0
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    for row in rows:
        pid = int(row["pid"])
        name = row.get("name", f"p{pid}")
        grammar_name = row.get("grammar_name")
        grammar_str = row["grammar_str"]
        examples = row["examples"]

        examples_block = build_examples_block(examples)
        grammar_block = grammar_str

        prompt_text = (
            template
            .replace("{GRAMMAR_BLOCK}", grammar_block)
            .replace("{EXAMPLES_BLOCK}", examples_block)
        )

        examples_plain = [
            {"inputs": ex["inputs"], "output": ex["output"]}
            for ex in examples
        ]
        problem_hash = compute_problem_hash(grammar_str, examples_plain)
        problem_id = f"sygus::{dataset_sha8}::{name}"

        fname = f"{args.prefix}_{problem_hash}_p{pid:04d}.txt"
        (outdir / fname).write_text(prompt_text, encoding="utf-8")

        index_records.append(
            {
                "problem_id": problem_id,
                "file": fname,
                "pid": pid,
                "name": name,
                "dataset_path": str(dataset_path),
                "example_count": len(examples),
                "written_at": now_str,
                "problem_hash": problem_hash,
                "grammar_name": grammar_name,
            }
        )

        written += 1
        if args.limit and written >= args.limit:
            break

    with index_path.open("w", encoding="utf-8") as f:
        for rec in index_records:
            f.write(json.dumps(rec, sort_keys=True))
            f.write("\n")

    print(f"Wrote {written} prompts to {outdir}")
    print(f"Wrote index to {index_path}")


if __name__ == "__main__":
    main()

