import sys, os, json
from pathlib import Path
from collections import Counter
from datetime import datetime, timedelta

ROOT = Path("D:/Dataset")
DATA_DIR = ROOT / "LifeMate_CS_ASR/data/train+public_test"
sys.path.insert(0, str(DATA_DIR / "public_test"))

from zaic_tools.executor import Simulator
from zaic_tools.utils import state_hash, canon
from zaic_tools import catalog_tools

catalog = json.load(open(DATA_DIR / "train/train_tools_catalog.json", encoding="utf-8"))
train_samples = [json.loads(line) for line in open(DATA_DIR / "train/train.jsonl", encoding="utf-8") if line.strip()]

from deep_train_evaluator import grounded_reactive_solver

failures_by_intent = Counter()

for sample in train_samples[:300]:
    sim_gold = Simulator(sample, catalog)
    gold_calls = sample["label"]["tool_calls"]
    for c in gold_calls:
        sim_gold.call(c["name"], c.get("arguments", {}))
    gold_hash = state_hash(sim_gold.states)
    
    pred_calls, pred_hash = grounded_reactive_solver(sample, catalog)
    
    if pred_hash != gold_hash:
        # Determine what app was modified in gold
        gold_modified_apps = []
        for app, st in sim_gold.states.items():
            init_st = sample["context"]["installed_apps"].get(app, {}).get("state", {})
            if canon(st) != canon(init_st):
                gold_modified_apps.append(app)
        
        failures_by_intent[tuple(sorted(gold_modified_apps))] += 1

print("\n--- Failure breakdown by modified apps in gold ---")
for apps, count in failures_by_intent.most_common(15):
    print(f"{count:3d} failures: {apps}")
