from pathlib import Path
import json
from glob import glob

SCRIPT_DIR = Path(__file__).resolve().parent
PROMPTS_DIR = SCRIPT_DIR / "prompts"
RUNS_DIR = SCRIPT_DIR / "runs"

def load_prompt_index():
    idx_path = PROMPTS_DIR / "index.jsonl"
    by_fname = {}
    by_hash = {}
    with open(idx_path, "r", encoding="utf-8") as f:
        for ln in f:
            rec = json.loads(ln)
            fname = rec.get("file") or rec.get("fname")  # your field name in index
            phash = rec.get("problem_hash")
            if fname:
                by_fname[fname] = rec
            if phash:
                by_hash[phash] = rec
    return by_fname, by_hash

def load_answers_by_hash():
    answers_for = {}
    for run_path in glob(str(RUNS_DIR / "run_*.jsonl")):
        with open(run_path, "r", encoding="utf-8") as f:
            for ln in f:
                rec = json.loads(ln)
                phash = rec.get("problem_hash")
                if not phash:
                    continue
                answers_for.setdefault(phash, []).append(rec["answer_path"])
    return answers_for

def build_pairs():
    by_fname, by_hash = load_prompt_index()
    answers_for = load_answers_by_hash()

    pairs = []  # [{prompt_path, problem_hash, answer_paths: [...]}, ...]
    for prompt_path in sorted((PROMPTS_DIR).glob("**/*.txt")):
        fname = prompt_path.name
        idx = by_fname.get(fname)
        if not idx:
            continue
        phash = idx.get("problem_hash")
        answers = answers_for.get(phash, [])
        pairs.append({
            "prompt_path": str(prompt_path),
            "problem_hash": phash,
            "answer_paths": answers,
        })
    return pairs

if __name__ == "__main__":
    pairs = build_pairs()
    # print / write as needed
    for row in pairs:
        print(row["problem_hash"], len(row["answer_paths"]), row["prompt_path"])
