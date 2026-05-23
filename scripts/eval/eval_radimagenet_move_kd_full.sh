#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/baduc/MoVE-KD"
PYTHON="/home/baduc/miniconda3/envs/move-kd/bin/python"

MODEL_PATH="${MODEL_PATH:-${ROOT}/checkpoints/finetune/move-kd-biomed7b-v1.1-formatfixed}"
MODEL_BASE="${MODEL_BASE:-lmsys/vicuna-7b-v1.5}"
DATA_ROOT="${DATA_ROOT:-${ROOT}/playground/data/radimagenet_test}"
OUT_DIR="${OUT_DIR:-${DATA_ROOT}/eval_full_move_kd}"
export DATA_ROOT OUT_DIR

QUESTIONS="${OUT_DIR}/questions_full.jsonl"
ANSWER_KEY="${OUT_DIR}/answer_key_full.jsonl"
ANSWERS="${OUT_DIR}/answers_full_move_kd.jsonl"
SCORED="${OUT_DIR}/scored_full_move_kd.jsonl"
METRICS="${OUT_DIR}/metrics_full_move_kd.json"

mkdir -p "${OUT_DIR}"

echo "[1/3] Preparing RadImageNet questions..."
"${PYTHON}" - <<'PY'
import csv
import json
import os
from pathlib import Path

root = Path(os.environ["DATA_ROOT"])
out_dir = Path(os.environ["OUT_DIR"])
csv_path = root / "dataset_labels.csv"
q_path = out_dir / "questions_full.jsonl"
key_path = out_dir / "answer_key_full.jsonl"
letters = ["A", "B", "C", "D"]

count = 0
with csv_path.open(newline="") as f, q_path.open("w") as qf, key_path.open("w") as kf:
    reader = csv.DictReader(f)
    for i, row in enumerate(reader):
        question = row["question"].strip()
        choices = [c.strip() for c in row.get("choices", "").split("|") if c.strip()]
        qtype = row.get("question_type", "").strip()

        if choices:
            choice_lines = "\n".join(f"{letters[j]}. {choice}" for j, choice in enumerate(choices))
            question = f"{question}\n{choice_lines}\nAnswer with exactly one option letter: A, B, C, or D."
        elif qtype == "closed":
            question = f"{question} Answer yes or no only."

        qid = f"radimagenet_test_{i:04d}"
        qf.write(json.dumps({"question_id": qid, "image": row["image_file"], "text": question}, ensure_ascii=False) + "\n")
        kf.write(json.dumps({
            "question_id": qid,
            "image": row["image_file"],
            "answer": row["answer"].strip(),
            "question_type": qtype,
            "choices": row.get("choices", ""),
            "prompt": question,
        }, ensure_ascii=False) + "\n")
        count += 1

print(f"Prepared {count} samples")
PY

echo "[2/3] Running MoVE-KD generation..."
(
    cd "${ROOT}"
    PYTHONPATH="${ROOT}" \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
    TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/triton-cache-move-kd}" \
    "${PYTHON}" -m llava.eval.model_vqa_loader \
        --model-path "${MODEL_PATH}" \
        --model-base "${MODEL_BASE}" \
        --image-folder "${DATA_ROOT}" \
        --question-file "${QUESTIONS}" \
        --answers-file "${ANSWERS}" \
        --conv-mode v1 \
        --temperature 0 \
        --num_beams 1 \
        --max_new_tokens 64
)

echo "[3/3] Scoring answers..."
"${PYTHON}" - <<'PY'
import collections
import json
import os
import re
from pathlib import Path

out_dir = Path(os.environ["OUT_DIR"])
answers_path = out_dir / "answers_full_move_kd.jsonl"
key_path = out_dir / "answer_key_full.jsonl"
metrics_path = out_dir / "metrics_full_move_kd.json"
scored_path = out_dir / "scored_full_move_kd.jsonl"

def norm(text):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", text.lower())).strip()

def first_letter(text):
    match = re.search(r"\b([ABCD])\b", text.upper())
    if match:
        return match.group(1)
    match = re.search(r"^[\s\(\[]*([ABCD])", text.upper())
    return match.group(1) if match else None

def yesno(text):
    normalized = norm(text)
    tokens = normalized.split()
    if not tokens:
        return None
    if tokens[0] in {"yes", "no"}:
        return tokens[0]
    padded = f" {normalized} "
    has_yes = " yes " in padded
    has_no = " no " in padded
    if has_yes and not has_no:
        return "yes"
    if has_no and not has_yes:
        return "no"
    return None

keys = {json.loads(line)["question_id"]: json.loads(line) for line in key_path.open()}
answers = [json.loads(line) for line in answers_path.open()]
by_type = collections.defaultdict(lambda: {"correct": 0, "total": 0})
rows = []

for answer in answers:
    key = keys[answer["question_id"]]
    qtype = key["question_type"]
    prediction = answer["text"]
    gold = key["answer"]

    if qtype == "multiple_choice":
        parsed = first_letter(prediction)
        correct = parsed == gold.upper()
    elif qtype == "closed":
        parsed = yesno(prediction)
        correct = parsed == norm(gold)
    else:
        parsed = norm(prediction)
        gold_norm = norm(gold)
        correct = bool(gold_norm) and (gold_norm == parsed or gold_norm in parsed)

    by_type[qtype]["total"] += 1
    by_type[qtype]["correct"] += int(correct)
    rows.append({**key, "prediction": prediction, "parsed_prediction": parsed, "correct": correct})

summary = {
    "overall": {
        "correct": sum(value["correct"] for value in by_type.values()),
        "total": sum(value["total"] for value in by_type.values()),
    },
    "by_type": {},
}
summary["overall"]["accuracy"] = summary["overall"]["correct"] / summary["overall"]["total"]
for qtype, value in sorted(by_type.items()):
    summary["by_type"][qtype] = {**value, "accuracy": value["correct"] / value["total"]}

with metrics_path.open("w") as f:
    json.dump(summary, f, indent=2)
with scored_path.open("w") as f:
    for row in rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

print(json.dumps(summary, indent=2))
print(f"Metrics: {metrics_path}")
print(f"Scored outputs: {scored_path}")
PY
