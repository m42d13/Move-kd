import json
 
INPUT = "/home/baduc/MoVE-KD/playground/data/radimagenet/radimagenet/radimagenet_llava_instruct.json"
OUTPUT = "/home/baduc/MoVE-KD/playground/data/radimagenet/radimagenet/radimagenet_llava_instruct_tiny.json"
N = 20000
 
with open(INPUT, "r", encoding="utf-8") as f:
    data = json.load(f)
 
print(f"Loaded {len(data):,} entries")
 
trimmed = data[:N]
 
with open(OUTPUT, "w", encoding="utf-8") as f:
    json.dump(trimmed, f, ensure_ascii=False)
 
print(f"Wrote {len(trimmed):,} entries to {OUTPUT}")
