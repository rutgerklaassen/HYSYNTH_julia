#!/usr/bin/env python3
import argparse
import json
import sys
import tempfile
import subprocess
from pathlib import Path
from collections import defaultdict
import hashlib
from datetime import datetime, timezone
import hashlib
import json

# ----------------------- Embedded Julia dumper -----------------------
JULIA_DUMPER = r"""
# --- Julia helper: dumps dataset as JSONL to STDOUT ---
try
    using JSON
catch
    import Pkg
    Pkg.add("JSON")
    using JSON
end

using Serialization
using HerbBenchmarks.Karel_2018
using HerbCore
using HerbSpecification
using HerbGrammar
using HerbConstraints

function safe_deserialize(path::AbstractString)
    try
        return deserialize(path)
    catch e
        error("Failed to deserialize '$(path)': $(e)")
    end
end

state_grid(x) = string(x)  # ASCII rendering similar to your viewer

# Serialize a KarelState into a compact Dict:
function serialize_state(st::KarelState)
    # markers: Dict{Tuple{Int,Int},Int} → sorted vector of [x,y,count]
    mvec = Vector{Vector{Int}}()
    for (pos, cnt) in st.markers
        push!(mvec, [pos[1], pos[2], cnt])
    end
    sort!(mvec)  # sort by x, then y, then count
    return Dict(
        "grid" => state_grid(st),
        "bag" => st.hero.marker_count,
        "markers" => mvec,
    )
end

# Return serialized input, vector of serialized intermediates, serialized output
function split_path(exec_path::Vector{KarelState})
    n = length(exec_path)
    if n == 0
        return Dict("grid"=>"", "bag"=>0, "markers"=>Int[]), Vector{Dict}(), Dict("grid"=>"", "bag"=>0, "markers"=>Int[])
    elseif n == 1
        s = serialize_state(exec_path[1])
        return s, Vector{Dict}(), s
    else
        input  = serialize_state(exec_path[1])
        inters = [serialize_state(s) for s in exec_path[2:end-1]]
        output = serialize_state(exec_path[end])
        return input, inters, output
    end
end

function main(dataset_path::AbstractString)
    ds = safe_deserialize(dataset_path)
    if !hasproperty(ds, :programs)
        error("Dataset missing .programs: $(dataset_path)")
    end
    progs = ds.programs

    for (pid, rec) in enumerate(progs)
        prog_txt = String(rec.program)
        depth    = getfield(rec, :depth)
        nw       = length(rec.worlds)
        for w in 1:nw
            tr = rec.traces[w]
            input, inters, output = split_path(tr.exec_path)
            JSON.print(Dict(
                "pid" => pid,
                "world" => w,
                "depth" => depth,
                "program" => prog_txt,
                "input_state" => input,
                "intermediate_states" => inters,
                "output_state" => output,
            ))
            print('\n')
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) < 1
        error("Usage: julia dump.jl DATASET.jls")
    end
    main(ARGS[1])
end
"""

# ---------- State string sanitizers ----------
BOX_TOP_LEFT  = "┌┏╔+"
BOX_TOP_RIGHT = "┐┓╗+"
BOX_BOTTOM_LEFT  = "└┗╚+"
BOX_BOTTOM_RIGHT = "┘┛╝+"
BOX_H = "─━═-"
BOX_V = "│┃║|"

def _is_border_line(s: str) -> bool:
    s = s.strip("\n\r")
    if not s:
        return False
    return (
        (s[0] in BOX_TOP_LEFT + BOX_BOTTOM_LEFT) and
        (s[-1] in BOX_TOP_RIGHT + BOX_BOTTOM_RIGHT) and
        all(ch in BOX_H for ch in s[1:-1])
    )

def _strip_side_border_chars(s: str) -> str:
    if not s:
        return s
    left = 1 if s and s[0] in BOX_V else 0
    right = len(s) - 1 if s and s[-1] in BOX_V else len(s)
    return s[left:right]

def strip_box_borders(state_str: str) -> str:
    lines = [ln.rstrip("\n\r") for ln in state_str.splitlines()]
    if not lines:
        return state_str
    if _is_border_line(lines[0]):
        lines = lines[1:]
    if lines and _is_border_line(lines[-1]):
        lines = lines[:-1]
    stripped = [_strip_side_border_chars(ln) for ln in lines]
    while stripped and not stripped[0].strip():
        stripped.pop(0)
    while stripped and not stripped[-1].strip():
        stripped.pop()
    return "\n".join(stripped)

# ---------- Rendering helpers for grid + metadata ----------
def _format_markers(markers_list) -> str:
    """
    markers_list is [[x,y,count], ...] or possibly [] or dicts with x/y/count.
    Presented as '(x,y)×c' joined by ', '.
    """
    if not markers_list:
        return "none"
    parts = []
    for triplet in markers_list:
        if isinstance(triplet, (list, tuple)) and len(triplet) >= 3:
            x, y, c = triplet[:3]
        elif isinstance(triplet, dict):
            x, y, c = triplet.get("x"), triplet.get("y"), triplet.get("count", 1)
        else:
            parts.append(str(triplet))
            continue
        parts.append(f"({x},{y})×{c}")
    return ", ".join(parts)

def _render_grid_with_meta(state_obj, strip_borders: bool) -> str:
    """
    state_obj expected keys: 'grid' (str), 'bag' (int), 'markers' (list of [x,y,count]).
    If given a plain string (old dumps), just render the string.
    """
    if isinstance(state_obj, str):
        grid = strip_box_borders(state_obj) if strip_borders else state_obj
        return grid

    grid = state_obj.get("grid", "")
    grid = strip_box_borders(grid) if strip_borders else grid
    bag  = state_obj.get("bag", 0)
    markers = state_obj.get("markers", [])
    meta = f"Bag: {bag} | Markers: {_format_markers(markers)}"
    if grid:
        return f"{grid}\n{meta}"
    else:
        return meta

# ---------- Julia bridge ----------
def run_julia_dump(dataset_path: Path) -> list[dict]:
    """Runs the embedded Julia dumper and returns a list of JSON dicts (per world)."""
    with tempfile.TemporaryDirectory() as td:
        dump_path = Path(td) / "dump.jl"
        dump_path.write_text(JULIA_DUMPER, encoding="utf-8")

        proc = subprocess.run(
            ["julia", str(dump_path), str(dataset_path)],
            check=False,
            capture_output=True,
            text=True,
        )

        if proc.returncode != 0:
            sys.stderr.write(proc.stderr)
            raise RuntimeError(
                f"Julia dump failed (code {proc.returncode}). "
                f"Is the path correct? ({dataset_path})"
            )

        records = []
        for line in proc.stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise RuntimeError(f"Bad JSON from Julia: {e}\nLine was:\n{line}") from e
        return records

# ---------- Prompt formatting ----------
def format_example_block(idx: int, input_state, inters: list, output_state, strip_borders: bool) -> str:
    lines = []
    lines.append(f"EXAMPLE {idx}")
    lines.append("Input:")
    lines.append(_render_grid_with_meta(input_state, strip_borders))
    lines.append("")
    lines.append("Intermediate states:")
    if not inters:
        lines.append("(no intermediate states)")
    else:
        for t, s in enumerate(inters, start=1):
            lines.append(f"t={t}")
            lines.append(_render_grid_with_meta(s, strip_borders))
            lines.append("")
        if lines and lines[-1] == "":
            lines.pop()
    lines.append("")
    lines.append("Output:")
    lines.append(_render_grid_with_meta(output_state, strip_borders))
    return "\n".join(lines)

def build_examples_block(inputs, all_inters, outputs, strip_borders: bool) -> str:
    blocks = []
    for i in range(len(inputs)):
        blocks.append(format_example_block(i + 1, inputs[i], all_inters[i], outputs[i], strip_borders))
    # visual separator between examples
    return ("\n" + "-" * 72 + "\n\n").join(blocks)

def build_legacy_blocks(inputs, all_inters, outputs, strip_borders: bool) -> tuple[str, str, str]:
    # Enrich legacy placeholders with bag/marker metadata too
    inputs_block = "\n\n".join(
        [f"EXAMPLE {i+1}\n{_render_grid_with_meta(inputs[i], strip_borders)}" for i in range(len(inputs))]
    )

    inter_groups = []
    for i, inters in enumerate(all_inters, start=1):
        if not inters:
            inter_groups.append(f"EXAMPLE {i}\n(no intermediate states)")
        else:
            lines = [f"EXAMPLE {i}"]
            for t, s in enumerate(inters, start=1):
                lines.append(f"t={t}\n{_render_grid_with_meta(s, strip_borders)}")
            inter_groups.append("\n".join(lines))
    inters_block = "\n\n" + ("\n" + "-" * 60 + "\n\n").join(inter_groups)

    outputs_block = "\n\n".join(
        [f"EXAMPLE {i+1}\n{_render_grid_with_meta(outputs[i], strip_borders)}" for i in range(len(outputs))]
    )
    return inputs_block, inters_block, outputs_block

def _to_plain(obj):
    """Recursively convert obj into plain JSON-serializable types, deterministically."""
    # dict
    if isinstance(obj, dict):
        # sort keys to make deterministic
        return {k: _to_plain(obj[k]) for k in sorted(obj.keys())}
    # list/tuple
    if isinstance(obj, (list, tuple)):
        return [_to_plain(x) for x in obj]
    # numpy arrays / tensors
    if hasattr(obj, "tolist"):
        try:
            return obj.tolist()
        except Exception:
            pass
    # dataclass / custom object
    if hasattr(obj, "__dict__"):
        return _to_plain(vars(obj))
    # primitives
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    # final fallback (stable textual form)
    return repr(obj)

def compute_problem_hash(program_str, worlds5):
    """
    Hash the *program* and the 5 *worlds* only.
    The serialization is canonical (sorted keys, no whitespace differences).
    """
    payload = {
        "program": (program_str or "").strip(),
        "worlds": [_to_plain(w) for w in worlds5],  # exactly 5
    }
    s = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    # short but collision-resistant enough for filenames and joins
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]


# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser(
        description="Generate ONE LLM prompt per program (bundling all worlds) from a Julia .jls dataset."
    )
    ap.add_argument("--dataset", "-d", required=True, help="Path to dataset .jls")
    ap.add_argument("--prompt", "-p", required=True, help="Path to prompt.txt template")
    ap.add_argument("--outdir", "-o", required=True, help="Directory to write prompts")
    ap.add_argument("--prefix", default="karel", help="Filename prefix for outputs")
    ap.add_argument("--limit", type=int, default=None, help="Max number of programs to write")
    ap.add_argument("--strip-borders", action="store_true",
                    help="Remove outer box-drawing borders from grids to save tokens")
    args = ap.parse_args()

    dataset_path = Path(args.dataset).expanduser().resolve()
    prompt_path = Path(args.prompt).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    template = prompt_path.read_text(encoding="utf-8")

    # 1) pull all (program,world) rows from Julia
    rows = run_julia_dump(dataset_path)

    # 2) group worlds by program id (pid), keep depth and program text
    grouped = defaultdict(lambda: {
        "depth": None,
        "program": None,
        "by_world": {}  # world -> dict
    })
    for row in rows:
        pid = row["pid"]
        grouped[pid]["depth"] = row["depth"]
        grouped[pid]["program"] = row["program"]
        grouped[pid]["by_world"][row["world"]] = row

    # compute dataset fingerprint (stable ID for joins)
    try:
        ds_bytes = dataset_path.read_bytes()
        dataset_sha8 = hashlib.sha1(ds_bytes).hexdigest()[:8]
    except Exception:
        dataset_sha8 = "nosha"

    # 3) for each program, order by world index and build a single prompt
    uses_examples_block = "{EXAMPLES_BLOCK}" in template

    inputs = outputs = inters = None  # for scope readability only
    index = []
    written = 0
    for pid in sorted(grouped.keys()):
        depth = grouped[pid]["depth"]
        prog  = grouped[pid]["program"]
        worlds = [grouped[pid]["by_world"][w] for w in sorted(grouped[pid]["by_world"].keys())]

        # Keep full state objects; border stripping happens in the renderer
        inputs  = [w["input_state"] for w in worlds]
        outputs = [w["output_state"] for w in worlds]
        inters  = [w["intermediate_states"] for w in worlds]

        # Prefer single EXAMPLES_BLOCK if present
        if uses_examples_block:
            examples_block = build_examples_block(inputs, inters, outputs, args.strip_borders)
            prompt_text = template.replace("{EXAMPLES_BLOCK}", examples_block)
        else:
            # Fill legacy placeholders with bundled, labeled content (enriched with bag/markers)
            inputs_block, inters_block, outputs_block = build_legacy_blocks(inputs, inters, outputs, args.strip_borders)
            prompt_text = (
                template
                .replace("{INPUT_STATE}", inputs_block)
                .replace("{INTERMEDIATE_TRACES}", inters_block)
                .replace("{OUTPUT_STATE}", outputs_block)
            )

        # Optionally include the true program/depth in the prompt if your template wants them
        # Example placeholders: {REFERENCE_PROGRAM} and {DEPTH}
        if "{REFERENCE_PROGRAM}" in prompt_text:
            prompt_text = prompt_text.replace("{REFERENCE_PROGRAM}", str(prog))
        if "{DEPTH}" in prompt_text:
            prompt_text = prompt_text.replace("{DEPTH}", str(depth))

        # stable problem_id
        problem_id = f"karel_by_depth::{dataset_sha8}::p{pid:04d}"
        worlds_states_only = [
            {
                "input_state":  w["input_state"],
                "intermediate_states": w["intermediate_states"],
                "output_state": w["output_state"],
            }
            for w in worlds
        ]
        problem_hash = compute_problem_hash(prog, worlds_states_only)  

        # Write one file per PROGRAM (no world suffix)
        fname = f"{args.prefix}_{problem_hash}_d{depth}_p{pid:04d}.txt"
        (outdir / fname).write_text(prompt_text, encoding="utf-8")

        index.append({
            "problem_id": problem_id,
            "file": fname,
            "pid": pid,
            "depth": depth,
            "dataset_path": str(dataset_path),
            "world_count": len(worlds),
            "written_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "problem_hash": problem_hash,
        })
        written += 1
        if args.limit and written >= args.limit:
            break

    # 4) write an index for bookkeeping (JSONL)
    (outdir / "index.jsonl").write_text(
        "".join(json.dumps(rec) + "\n" for rec in index),
        encoding="utf-8",
    )

    print(f"Wrote {written} prompt files to {outdir}")
    print(f"Index: {outdir / 'index.jsonl'}")

if __name__ == "__main__":
    main()
