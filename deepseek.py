#!/usr/bin/env python3
# pip install openai
import os
import json
from pathlib import Path
from time import sleep
from datetime import datetime, timezone
from typing import Iterable
from openai import OpenAI

# --- Config ---
with open('api_keys.json') as f:
    api_json = json.load(f)
os.environ["OPENAI_API_KEY"] = api_json["DEEPSEEK_API"]  # or set in shell
MODEL = "deepseek-chat"
BASE_URL = "https://api.deepseek.com"
API_KEY = os.environ.get("OPENAI_API_KEY")

N_SAMPLES = int(os.environ.get("N_SAMPLES", "10"))  # how many answers per prompt
SCRIPT_DIR = Path(__file__).resolve().parent
PROMPTS_DIR = Path(os.environ.get("PROMPTS_DIR", str(SCRIPT_DIR / "prompts"))).expanduser().resolve()
ANSWERS_DIR = Path(os.environ.get("ANSWERS_DIR", str(SCRIPT_DIR / "answers"))).expanduser().resolve()
RUNS_DIR    = Path(os.environ.get("RUNS_DIR",    str(SCRIPT_DIR / "runs"))).expanduser().resolve()

RUNS_DIR.mkdir(parents=True, exist_ok=True)

def yield_prompt_files(dirpath: Path) -> Iterable[Path]:
    # recurse through prompts/
    for p in sorted(dirpath.rglob("*.txt")):
        # Skip already-generated answers if they accidentally live in prompts
        if p.stem.endswith("_answer"):
            continue
        yield p

def _load_prompt_index_map() -> dict:
    """
    Reads prompts/index.jsonl and returns filename->record mapping for quick lookup.
    """
    index_path = PROMPTS_DIR / "index.jsonl"
    mapping = {}
    if index_path.is_file():
        for ln in index_path.read_text(encoding="utf-8").splitlines():
            if not ln.strip():
                continue
            rec = json.loads(ln)
            mapping[rec["file"]] = rec
    return mapping

def infer_problem_id(prompt_file: Path, index_map: dict) -> str:
    rec = index_map.get(prompt_file.name)
    if rec and "problem_id" in rec:
        return rec["problem_id"]
    # fallback if index missing (keeps working but less ideal)
    return f"unknown::{prompt_file.name}"
def infer_problem_hash(prompt_file: Path, index_map: dict):
    """
    Pull a per-problem hash (computed in generate_prompts.py) from prompts/index.jsonl.
    Returns None if not present (so older indices won't crash).
    """
    rec = index_map.get(prompt_file.name)
    if rec:
        return rec.get("problem_hash")
    return None

def main():
    if not PROMPTS_DIR.is_dir():
        raise SystemExit(f"Prompts folder not found: {PROMPTS_DIR}")
    if not API_KEY:
        raise SystemExit("Set OPENAI_API_KEY in your environment before running.")

    ANSWERS_DIR.mkdir(parents=True, exist_ok=True)
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    files = list(yield_prompt_files(PROMPTS_DIR))
    if not files:
        print("No prompt .txt files found.")
        return

    index_map = _load_prompt_index_map()

    # per-run manifest (JSONL)
    run_ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_manifest = RUNS_DIR / f"run_{MODEL}_{run_ts}.jsonl"
    mf = run_manifest.open("w", encoding="utf-8")

    print(f"Found {len(files)} prompt file(s) in {PROMPTS_DIR}")
    for i, prompt_file in enumerate(files, 1):
        try:
            prompt_text = prompt_file.read_text(encoding="utf-8").strip()
            if not prompt_text:
                print(f"[{i}/{len(files)}] {prompt_file}: skipped (empty).")
                continue
            abs_prompt = prompt_file.resolve()

            # Mirror directory tree under answers/
            rel_path = abs_prompt.relative_to(PROMPTS_DIR)   # both absolute now
            out_parent = ANSWERS_DIR / rel_path.parent
            out_parent.mkdir(parents=True, exist_ok=True)

            stem = abs_prompt.stem
            legacy = out_parent / f"{stem}_answer.txt"
            existing = list(out_parent.glob(f"{stem}_answer_*.txt"))

            def _idx_from_name(p: Path) -> int:
                s = p.stem  # e.g. karel_d7_p0001_answer_0003
                try:
                    return int(s.rsplit("_", 1)[-1])
                except Exception:
                    return 1  # treat legacy as 1
            max_idx = 1 if legacy.exists() else 0
            for p in existing:
                max_idx = max(max_idx, _idx_from_name(p))
            start_idx = max_idx + 1

            problem_id = infer_problem_id(prompt_file, index_map)
            problem_hash = infer_problem_hash(prompt_file, index_map)

            for k in range(start_idx, start_idx + N_SAMPLES):
                out_file = out_parent / f"{stem}_answer_{k:04d}.txt"

                # Call model
                resp = client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": ""},
                        {"role": "user", "content": prompt_text},
                    ],
                    stream=False,
                )
                answer = resp.choices[0].message.content or ""
                out_file.write_text(answer, encoding="utf-8")

                # One manifest row per sample
                row = {
                    "problem_id": problem_id,
                    "problem_hash": problem_hash,  # <-- NEW
                    "prompt_path": os.path.relpath(abs_prompt, start=SCRIPT_DIR),
                    "answer_path": os.path.relpath(out_file.resolve(), start=SCRIPT_DIR),
                    "model": MODEL,
                    "base_url": BASE_URL,
                    "timestamp": run_ts,
                    "sample_id": k,
                }

                try:
                    if hasattr(resp, "usage") and resp.usage is not None:
                        row["usage"] = {
                            "prompt_tokens": getattr(resp.usage, "prompt_tokens", None),
                            "completion_tokens": getattr(resp.usage, "completion_tokens", None),
                            "total_tokens": getattr(resp.usage, "total_tokens", None),
                        }
                except Exception:
                    pass

                mf.write(json.dumps(row) + "\n")
                print(f"[{i}/{len(files)}] Wrote: {out_file.relative_to(SCRIPT_DIR)}")
                sleep(0.2)  # gentle pacing

        except Exception as e:
            print(f"[{i}/{len(files)}] ERROR on {prompt_file}: {e}")

    mf.close()
    print(f"Run manifest: {run_manifest}")

if __name__ == "__main__":
    main()
