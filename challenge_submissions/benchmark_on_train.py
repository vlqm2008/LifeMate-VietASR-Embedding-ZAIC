import sys, os, json
from pathlib import Path
from collections import Counter

ROOT = Path("D:/Dataset")
DATA_DIR = ROOT / "LifeMate_CS_ASR/data/train+public_test"
sys.path.insert(0, str(DATA_DIR / "public_test"))
from zaic_tools import ToolEnv
from zaic_tools.utils import state_hash, canon

train = [json.loads(line) for line in open(DATA_DIR / "train/train.jsonl", encoding="utf-8")]
print(f"Loaded {len(train)} train samples.")

def eval_method_on_train(method_name, solver_fn):
    state_matches = 0
    f1_scores = []
    
    for i, s in enumerate(train[:300]): # test first 300 for speed
        gold_calls = s["label"]["tool_calls"]
        gold_state = s["label"]["final_state"]
        gold_hash = state_hash(gold_state)
        
        pred_calls, pred_hash = solver_fn(s)
        
        if pred_hash == gold_hash:
            state_matches += 1
            
        # Call F1
        def key(c):
            return json.dumps({"name": c["name"], "arguments": c.get("arguments", {})}, sort_keys=True)
        gold_keys = [key(c) for c in gold_calls]
        pred_keys = [key(c) for c in pred_calls]
        matched = sum((Counter(pred_keys) & Counter(gold_keys)).values())
        total = len(pred_keys) + len(gold_keys)
        f1 = (2 * matched / total) if total > 0 else 1.0
        f1_scores.append(f1)
        
    state_acc = state_matches / 300
    avg_f1 = sum(f1_scores) / 300
    combined = 0.5 * state_acc + 0.5 * avg_f1
    print(f"[{method_name}] State Match: {state_acc*100:.2f}% | Tool F1: {avg_f1*100:.2f}% | Combined: {combined*100:.2f}%")
    return state_acc, avg_f1, combined

# Test M3 logic on train
from datetime import datetime, timedelta

def m3_solver(item):
    s_id = item["id"]
    # build simulator
    from zaic_tools.executor import Simulator
    sim = Simulator(item, json.load(open(DATA_DIR / "train/train_tools_catalog.json", encoding="utf-8")))
    installed = sim.allowed
    now_str = item.get("context", {}).get("current_time", "2026-11-13T12:00:00+07:00")
    try: now_dt = datetime.fromisoformat(now_str)
    except: now_dt = datetime(2026, 11, 13, 12, 0, 0)
    context_apps = list(sim.apps.keys())
    
    proposed = []
    if "alarm" in context_apps and "alarm.list" in installed:
        proposed.append(("alarm.list", {}))
        if "alarm.create" in installed:
            t = (now_dt + timedelta(days=1)).replace(hour=6, minute=30, second=0).isoformat()
            proposed.append(("alarm.create", {"time": t, "label": "Báo thức sáng"}))
    if "smart_home" in context_apps and "smart_home.get_status" in installed:
        proposed.append(("smart_home.get_status", {}))
        if "smart_home.light_set" in installed:
            proposed.append(("smart_home.light_set", {"device_id": "light_bedroom", "power": "off"}))
    if "calendar" in context_apps and "calendar.list_events" in installed:
        proposed.append(("calendar.list_events", {}))
        if "calendar.create_event" in installed:
            st = (now_dt + timedelta(days=2)).replace(hour=14, minute=0, second=0).isoformat()
            en = (now_dt + timedelta(days=2)).replace(hour=15, minute=0, second=0).isoformat()
            proposed.append(("calendar.create_event", {"title": "Họp", "start": st, "end": en, "location": "A"}))
    if "weather" in context_apps and "weather.get_forecast" in installed:
        proposed.append(("weather.get_forecast", {"date": (now_dt + timedelta(days=1)).strftime("%Y-%m-%d"), "location": "Đà Nẵng"}))
    if "contacts" in context_apps and "contacts.search" in installed:
        proposed.append(("contacts.search", {"query": "Dũng"}))
        
    seen = set()
    for name, args in proposed:
        k = (name, json.dumps(args, sort_keys=True))
        if k not in seen:
            seen.add(k)
            sim.call(name, args)
    return sim.log, state_hash(sim.states)

eval_method_on_train("Method 3 (Portal Baseline)", m3_solver)
