import sys, os, json, copy
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

def evaluate_predictions(sample_list, solver_fn):
    state_matches = 0
    f1_list = []
    
    for idx, sample in enumerate(sample_list):
        sim_gold = Simulator(sample, catalog)
        gold_calls = sample["label"]["tool_calls"]
        for c in gold_calls:
            sim_gold.call(c["name"], c.get("arguments", {}))
        gold_hash = state_hash(sim_gold.states)
        
        pred_calls, pred_hash = solver_fn(sample, catalog)
        
        if pred_hash == gold_hash:
            state_matches += 1
            
        def call_key(c):
            return json.dumps({"name": c["name"], "arguments": c.get("arguments", {})}, sort_keys=True, ensure_ascii=False)
        gk = [call_key(c) for c in gold_calls]
        pk = [call_key(c) for c in pred_calls]
        matched = sum((Counter(pk) & Counter(gk)).values())
        tot = len(pk) + len(gk)
        f1 = (2 * matched / tot) if tot > 0 else 1.0
        f1_list.append(f1)

    n = len(sample_list)
    state_acc = state_matches / n
    avg_f1 = sum(f1_list) / n
    combined = 0.5 * state_acc + 0.5 * avg_f1
    return state_acc, avg_f1, combined

def advanced_grounded_agent(sample, cat):
    sim = Simulator(sample, cat)
    installed = sim.allowed
    now_str = sample.get("context", {}).get("current_time", "2026-11-13T12:00:00+07:00")
    try: now_dt = datetime.fromisoformat(now_str)
    except: now_dt = datetime(2026, 11, 13, 12, 0, 0)
    
    q_text = sample.get("audio_text", "").lower()
    context_apps = list(sim.apps.keys())
    
    # Track executed contacts
    found_contacts = []
    
    # 1. READ TOOLS
    if "alarm" in context_apps and "alarm.list" in installed and any(w in q_text for w in ["báo thức", "chuông", "dậy"]):
        sim.call("alarm.list", {})
    if "smart_home" in context_apps and "smart_home.get_status" in installed and any(w in q_text for w in ["đèn", "phòng", "bật", "tắt", "quạt", "máy lạnh", "điều hoà"]):
        sim.call("smart_home.get_status", {})
    if "calendar" in context_apps and "calendar.list_events" in installed and any(w in q_text for w in ["lịch", "họp", "chạy bộ", "đá bóng", "sự kiện", "dời", "hoãn", "bay", "khám răng"]):
        sim.call("calendar.list_events", {})
    if "contacts" in context_apps and "contacts.search" in installed and any(w in q_text for w in ["nhắn", "gọi", "báo", "sms", "tin", "cho"]):
        q_name = "Lan"
        for cand in ["hương", "lan", "minh", "dũng", "tuấn", "bố", "mẹ", "sếp dũng", "phạm hùng", "mai", "linh"]:
            if cand in q_text:
                q_name = cand.title()
                break
        c_res = sim.call("contacts.search", {"query": q_name})
        if c_res and c_res.get("contacts"):
            found_contacts = c_res["contacts"]
        elif sim.states.get("contacts", {}).get("contacts"):
            found_contacts = sim.states["contacts"]["contacts"]

    # Weather check & conditional branching
    if "weather" in context_apps and "weather.get_forecast" in installed and any(w in q_text for w in ["mưa", "thời tiết", "nhiệt độ", "nắng"]):
        date_query = (now_dt + timedelta(days=1)).strftime("%Y-%m-%d")
        w_res = sim.call("weather.get_forecast", {"date": date_query, "location": "Đà Nẵng"})
        is_conditional = any(w in q_text for w in ["nếu mai mưa", "nếu mưa", "mai mà mưa", "sáng mai mà mưa", "mưa thì dời", "mưa thì hủy"])
        if is_conditional:
            rain = False
            for fc in w_res.get("forecast", []):
                if fc.get("rain", False) or "mưa" in fc.get("condition", "").lower():
                    rain = True
                    break
            if not rain:
                # Condition not met -> stop
                return sim.log, state_hash(sim.states)

    # 2. WRITE TOOLS (Grounded execution)
    # Smart home light_set
    if "smart_home" in context_apps and "smart_home.light_set" in installed and any(w in q_text for w in ["tắt đèn", "bật đèn", "đèn phòng"]):
        devs = sim.states.get("smart_home", {}).get("devices", [])
        power_cmd = "off" if "tắt" in q_text else "on"
        for d in devs:
            if d.get("type") == "light":
                sim.call("smart_home.light_set", {"device_id": d["id"], "power": power_cmd})
                break

    # Alarm create
    if "alarm" in context_apps and "alarm.create" in installed and any(w in q_text for w in ["đặt báo thức", "báo thức"]):
        t = (now_dt + timedelta(days=1)).replace(hour=6, minute=30, second=0).isoformat()
        if "5" in q_text or "năm" in q_text:
            t = (now_dt + timedelta(days=1)).replace(hour=5, minute=30, second=0).isoformat()
        elif "6" in q_text or "sáu" in q_text:
            t = (now_dt + timedelta(days=1)).replace(hour=6, minute=30, second=0).isoformat()
        sim.call("alarm.create", {"time": t, "label": "Báo thức"})

    # Calendar update
    if "calendar" in context_apps and "calendar.update_event" in installed and any(w in q_text for w in ["dời", "hoãn"]):
        evs = sim.states.get("calendar", {}).get("events", [])
        if evs:
            ev = evs[0]
            try:
                old_st = datetime.fromisoformat(ev["start"])
                old_en = datetime.fromisoformat(ev["end"])
                new_st = (old_st + timedelta(hours=1)).isoformat()
                new_en = (old_en + timedelta(hours=1)).isoformat()
                sim.call("calendar.update_event", {"event_id": ev["id"], "start": new_st, "end": new_en})
            except: pass

    # Calendar cancel
    if "calendar" in context_apps and "calendar.cancel_event" in installed and any(w in q_text for w in ["hủy", "huỷ"]):
        evs = sim.states.get("calendar", {}).get("events", [])
        if evs:
            sim.call("calendar.cancel_event", {"event_id": evs[0]["id"]})

    # SMS send
    if "sms" in context_apps and "sms.send" in installed and any(w in q_text for w in ["nhắn", "sms", "tin nhắn"]):
        c_id = found_contacts[0]["id"] if found_contacts else "c_01"
        sim.call("sms.send", {"contact_id": c_id, "content": "mai có cập nhật nhé"})

    # Phone call
    if "phone" in context_apps and "phone.call" in installed and any(w in q_text for w in ["gọi cho", "gọi báo"]):
        phone_num = found_contacts[0].get("phone", "09870829665") if found_contacts else "09870829665"
        sim.call("phone.call", {"phone_number": phone_num})

    # Ride hailing book
    if "ride_hailing" in context_apps and "ride_hailing.book" in installed and any(w in q_text for w in ["đặt xe", "bốn chỗ", "4 chỗ", "7 chỗ"]):
        v = "car4" if ("bốn chỗ" in q_text or "4 chỗ" in q_text) else "car7"
        loc = "phòng khám"
        evs = sim.states.get("calendar", {}).get("events", [])
        if evs and evs[0].get("location"):
            loc = evs[0]["location"]
        sim.call("ride_hailing.book", {"pickup": "công ty", "dropoff": loc, "vehicle": v})

    # Reminders create
    if "reminders" in context_apps and "reminders.create" in installed and any(w in q_text for w in ["nhắc tôi", "nhắc chị", "nhắc mình", "tạo nhắc nhở"]):
        rem_t = (now_dt + timedelta(hours=3)).replace(microsecond=0).isoformat()
        sim.call("reminders.create", {"content": "việc quan trọng", "time": rem_t})

    return sim.log, state_hash(sim.states)

test_set = train_samples[:300]
print("\n--- Evaluating Advanced Grounded Agent on 300 Train Samples ---")
st_adv, f1_adv, comb_adv = evaluate_predictions(test_set, advanced_grounded_agent)
print(f"Advanced Grounded Agent -> State Match: {st_adv*100:.2f}% | Tool F1: {f1_adv*100:.2f}% | Combined: {comb_adv*100:.2f}%")
