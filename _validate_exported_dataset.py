import os
import glob
import json

base = r"c:/Users/54661/Desktop/工作/论文/写作/IWQOS/code/qos_lsma/dataset_paper_2378"
target = {
    "Tourism": 518,
    "Finance": 463,
    "Technology": 497,
    "Medical": 432,
    "Cross_Domain": 468,
}

required_top = {"user_query", "domain", "TaskList"}
required_task = {"id", "description", "domain", "required_inputs"}

ok = True

for d, n in target.items():
    files = glob.glob(os.path.join(base, d, "*.json"))
    print(f"{d}: {len(files)} (target={n})")
    if len(files) != n:
        ok = False

issues = []
for d in target:
    files = glob.glob(os.path.join(base, d, "*.json"))
    for fp in files:
        rel = os.path.relpath(fp, base)
        try:
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            issues.append((rel, f"bad_json: {e}"))
            continue

        if not isinstance(data, dict):
            issues.append((rel, "not_dict"))
            continue

        miss_top = required_top - set(data.keys())
        if miss_top:
            issues.append((rel, f"missing_top: {sorted(list(miss_top))}"))

        q = data.get("user_query")
        if not isinstance(q, str) or not q.strip():
            issues.append((rel, "empty_user_query"))

        tl = data.get("TaskList")
        if not isinstance(tl, list) or len(tl) == 0:
            issues.append((rel, "empty_tasklist"))
            continue

        for i, t in enumerate(tl):
            if not isinstance(t, dict):
                issues.append((rel, f"task[{i}] not_dict"))
                continue
            miss_task = required_task - set(t.keys())
            if miss_task:
                issues.append((rel, f"task[{i}] missing: {sorted(list(miss_task))}"))

print("issues:", len(issues))
if issues:
    for x in issues[:10]:
        print(" ", x)
    ok = False

print("VALID_EXPORT:", ok)
