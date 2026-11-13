import json

pub = [json.loads(l) for l in open('D:/Dataset/LifeMate_CS_ASR/data/train+public_test/public_test/public_test.jsonl', encoding='utf-8')]
trans = json.load(open('D:/Dataset/CHALLENGE_TRACK1_SUBMISSIONS/public_test_real_transcripts.json', encoding='utf-8'))

write_apps = {'alarm', 'smart_home', 'calendar'}
multi_write_samples = []
for s in pub:
    installed = set(s['context']['installed_apps'].keys()) & write_apps
    if len(installed) >= 2:
        txt = trans.get(s['id'], {}).get('transcript', '')
        multi_write_samples.append((s['id'], installed, txt))

print(f"Total multi-write samples: {len(multi_write_samples)}")
print("First 15 multi-write samples with transcripts:")
for sid, apps, txt in multi_write_samples[:15]:
    print(f"[{sid}] apps={sorted(list(apps))} -> \"{txt}\"")
