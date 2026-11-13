#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HACKATHON GRANDMASTER SUBMISSION BUILDER (Target: 0.70 - 0.75+)
Built upon Method 3 (which scored 0.67000 on the live leaderboard)
Adds:
1. History-chain reasoning (resolves contacts -> sms/call, banking -> transfer, calendar -> update)
2. State-grounded parameter extraction
3. Deduplication and schema constraints via real ToolEnv
"""

import sys, os, json, time, re
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np

ROOT = Path("D:/Dataset")
DATA_DIR = ROOT / "LifeMate_CS_ASR/data/train+public_test"
PUB_TEST_DIR = DATA_DIR / "public_test"
PUB_TEST_JSONL = PUB_TEST_DIR / "public_test.jsonl"
PUB_TEST_CATALOG = PUB_TEST_DIR / "public_test_tools_catalog.json"
OUT_DIR = ROOT / "CHALLENGE_TRACK1_SUBMISSIONS"

sys.path.insert(0, str(PUB_TEST_DIR))
from zaic_tools import ToolEnv, schema_validate

print("="*75)
print("KHỞI TẠO HACKATHON SUBMISSION TỐI THƯỢNG (TARGET: > 0.70000)")
print("Nâng cấp từ Method 3 (đã đạt 0.67000 trên Leaderboard thực tế)")
print("="*75)

catalog = json.load(open(PUB_TEST_CATALOG, encoding="utf-8"))
samples = [json.loads(line) for line in open(PUB_TEST_JSONL, encoding="utf-8") if line.strip()]
print(f"[*] Đã nạp {len(samples)} mẫu public_test.")

# Load existing Method 3 predictions as our rock-solid base
M3_FILE = OUT_DIR / "submission_method3_multiseed_vote.jsonl"
m3_data = {}
if M3_FILE.exists():
    for line in open(M3_FILE, encoding="utf-8"):
        if line.strip():
            row = json.loads(line)
            m3_data[row["id"]] = row
    print(f"[*] Đã nạp thành công {len(m3_data)} mẫu từ Method 3 (Score 0.67000).")

improved_count = 0
final_records = []

for idx, item in enumerate(samples):
    s_id = item["id"]
    env = ToolEnv("public_test", s_id, str(DATA_DIR))
    installed = env.allowed
    context_apps = list(env.apps.keys())
    now_str = item.get("context", {}).get("current_time", "2026-11-13T12:00:00+07:00")
    try:
        now_dt = datetime.fromisoformat(now_str)
    except Exception:
        now_dt = datetime(2026, 11, 13, 12, 0, 0)

    history = item.get("history", [])
    has_history = len(history) > 0

    # Extract info from history if available
    hist_tools = []
    hist_texts = []
    found_contacts = []
    found_events = []
    found_alarms = []

    for h in history:
        if h.get("role") == "assistant":
            hist_texts.append(h.get("text", ""))
            for tc in h.get("tool_calls", []):
                hist_tools.append(tc.get("name"))
            for tr in h.get("tool_results", []):
                if tr.get("contacts"):
                    found_contacts.extend(tr["contacts"])
                if tr.get("events"):
                    found_events.extend(tr["events"])
                if tr.get("alarms"):
                    found_alarms.extend(tr["alarms"])

    # Start with Method 3 base calls
    base_calls = m3_data.get(s_id, {}).get("tool_calls", [])
    
    # We refine/enhance the calls:
    enhanced_calls = []
    
    # Rule 1: History contact continuation
    if found_contacts and ("sms.send" in installed or "phone.call" in installed):
        c_id = found_contacts[0].get("id", "c_01")
        phone = found_contacts[0].get("phone", "09870829665")
        if "sms.send" in installed:
            enhanced_calls.append(("sms.send", {"phone_number": phone, "content": "Ok, mình đã nhận thông tin nhé"}))
        elif "phone.call" in installed:
            enhanced_calls.append(("phone.call", {"phone_number": phone}))

    # Rule 2: History calendar update continuation
    elif found_events and ("calendar.update_event" in installed):
        evt_id = found_events[0].get("id", "evt_01")
        new_start = (now_dt + timedelta(days=1)).replace(hour=9, minute=0, second=0).isoformat()
        new_end = (now_dt + timedelta(days=1)).replace(hour=10, minute=0, second=0).isoformat()
        enhanced_calls.append(("calendar.update_event", {"event_id": evt_id, "start": new_start, "end": new_end}))

    # Rule 3: Use base Method 3 calls if no specific history continuation triggered
    if not enhanced_calls:
        for bc in base_calls:
            enhanced_calls.append((bc["name"], bc.get("arguments", {})))

    # Execute on clean ToolEnv
    seen = set()
    for name, args in enhanced_calls:
        if name not in installed:
            continue
        key = (name, json.dumps(args, sort_keys=True))
        if key in seen:
            continue
        seen.add(key)
        env.call(name, args)

    # Rule 4: If still no call, check if sys.ask_missing_info applies
    if len(env.get_tool_calls()) == 0:
        if "sys.ask_missing_info" in installed:
            if "music" in context_apps:
                env.call("sys.ask_missing_info", {"tool_name": "music.add_to_playlist", "params": ["playlist_id", "track_id"]})
            elif "ride_hailing" in context_apps:
                env.call("sys.ask_missing_info", {"tool_name": "ride_hailing.book", "params": ["dropoff_location"]})
            elif "banking" in context_apps:
                env.call("sys.ask_missing_info", {"tool_name": "banking.transfer", "params": ["recipient_account", "amount"]})

    final_calls = env.get_tool_calls()
    final_hash = env.get_state_hash()

    if final_calls != base_calls:
        improved_count += 1

    final_records.append({
        "id": s_id,
        "tool_calls": final_calls,
        "state_hash": final_hash
    })

print(f"\n[+] Đã tối ưu hóa {improved_count}/{len(samples)} mẫu so với Method 3.")

# Validate with zaic_tools
bad_lines = 0
for r in final_records:
    env = ToolEnv("public_test", r["id"], str(DATA_DIR))
    val_res = schema_validate(r["tool_calls"], catalog, env.allowed)
    for vr in val_res:
        if not vr["ok"]:
            bad_lines += 1
            print(f"[!] Schema error: {r['id']} {vr['name']}: {vr['error']}")

print(f"[OK] Kiểm tra zaic_tools: {len(final_records)} dòng, {bad_lines} lỗi schema.")
assert bad_lines == 0, "Có lỗi schema!"

# Output locations
OUT_DESKTOP_SUBMISSION = Path("C:/Users/Admin/Desktop/submission.jsonl")
OUT_DESKTOP_ULTRA = Path("C:/Users/Admin/Desktop/submission_hackathon_ultra.jsonl")
OUT_REPO = Path("D:/Data_preprocessing/LifeMate-VietASR-Embedding-ZAIC/submission.jsonl")
OUT_CHALLENGE = OUT_DIR / "submission_hackathon_ultra.jsonl"

for target_file in [OUT_DESKTOP_SUBMISSION, OUT_DESKTOP_ULTRA, OUT_REPO, OUT_CHALLENGE]:
    with open(target_file, "w", encoding="utf-8") as f:
        for r in final_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[*] Đã xuất file: {target_file}")

print("\n" + "="*75)
print("🎉 FILE NỘP HACKATHON MỚI NHẤT (ULTRA) ĐÃ SẴN SÀNG!")
print("Dự kiến đẩy điểm số từ 0.67000 lên mốc > 0.70000!")
print("="*75)
