#!/usr/bin/env python3
import json, re
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROMPTS_DIR = SCRIPT_DIR / "prompts"
ANSWERS_DIR = SCRIPT_DIR / "answers"
OUT_JSONL   = SCRIPT_DIR / "prompt_answer_manifest.jsonl"

def stem_id(path: Path) -> str:
    """
    A stable ID from filename. Examples:
      karel_d7_p0001.txt      -> karel_d7_p0001
      p0001.txt               -> p0001
    """
    return path.stem

def guess_problem_index(stem: str):
    """
    Optional: extract numeric index like p0001 -> 1 (helps align with your dataset order).
    Returns int or None if not found.
    """
    m = re.search(r'[pP](\d{1,6})', stem)
    return int(m.group(1)) if m else None

def main():
    if not PROMPTS_DIR.is_dir():
        raise SystemExit(f"Prompts folder not found: {PROMPTS_DIR}")
    OUT_JSONL.unlink(missing_ok=True)

    files = sorted(PROMPTS_DIR.rglob("*.txt"))
    total = 0
    paired = 0
    missing = 0

    with OUT_JSONL.open("w", encoding="utf-8") as out:
        for p in files:
            if p.stem.endswith("_answer"):  # skip accidental answers in prompts
                continue
            total += 1
            rel = p.relative_to(PROMPTS_DIR)
            ans = (ANSWERS_DIR / rel.parent / f"{p.stem}_answer.txt")
            has_answer = ans.is_file()

            entry = {
                "id": stem_id(p),
                "problem_index": guess_problem_index(p.stem),  # optional, may be null
                "prompt_path": str(p),
                "answer_path": str(ans) if has_answer else None,
                "has_answer": has_answer,
            }
            out.write(json.dumps(entry, ensure_ascii=False) + "\n")
            paired += int(has_answer)
            missing += int(not has_answer)

    print(f"Manifest: {OUT_JSONL.name}")
    print(f"Prompts found: {total}")
    print(f"Answers paired: {paired}")
    print(f"Missing answers: {missing}")

if __name__ == "__main__":
    main()