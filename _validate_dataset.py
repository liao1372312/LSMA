import os
import glob
import json
from collections import Counter

base = r"c:/Users/54661/Desktop/工作/论文/写作/IWQOS/code/qos_lsma/dataset"
domains = ["Tourism", "Finance", "Technology", "Medical", "Cross_Domain"]

required_top = {"user_query", "domain", "TaskList"}
required_task = {"id", "description", "domain", "required_inputs"}

bad_json = []
missing_top = []
empty_query = []
empty_tasklist = []
missing_task_keys = []
domain_mismatch = []
task_domain_counter = Counter()

for d in domains:
    folder = os.path.join(base, d)
    files = glob.glob(os.path.join(folder, "*.json"))
    for fp in files:
        rel = os.path.relpath(fp, base)
        try:
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            bad_json.append((rel, str(e)))
            continue

        if not isinstance(data, dict):
            missing_top.append((rel, "not_dict"))
            continue

        if not required_top.issubset(set(data.keys())):
            missing_top.append((rel, sorted(list(required_top - set(data.keys())))))

        q = data.get("user_query")
        if not isinstance(q, str) or not q.strip():
            empty_query.append(rel)

        tl = data.get("TaskList")
        if not isinstance(tl, list) or len(tl) == 0:
            empty_tasklist.append(rel)
            continue

        # domain consistency
        file_domain = d
        data_domain = str(data.get("domain", "")).strip()
        if file_domain == "Cross_Domain":
            expected = {"cross_domain", "cross-domain", "cross domain", "cross_domain_task", "crossdomain"}
            if data_domain.lower().replace(" ", "_") not in expected and data_domain.lower() != "cross_domain":
                domain_mismatch.append((rel, data_domain, file_domain))
        else:
            if data_domain.lower() != file_domain.lower():
                domain_mismatch.append((rel, data_domain, file_domain))

        for idx, t in enumerate(tl):
            if not isinstance(t, dict):
                missing_task_keys.append((rel, idx, "not_dict"))
                continue
            miss = required_task - set(t.keys())
            if miss:
                missing_task_keys.append((rel, idx, sorted(list(miss))))
            td = str(t.get("domain", "")).strip().lower()
            task_domain_counter[td] += 1

print("Summary:")
print(" bad_json:", len(bad_json))
print(" missing_top:", len(missing_top))
print(" empty_query:", len(empty_query))
print(" empty_tasklist:", len(empty_tasklist))
print(" missing_task_keys:", len(missing_task_keys))
print(" domain_mismatch:", len(domain_mismatch))
print(" task_domain_top10:", task_domain_counter.most_common(10))

# print first few issues
for name, arr in [
    ("bad_json", bad_json),
    ("missing_top", missing_top),
    ("empty_query", empty_query),
    ("empty_tasklist", empty_tasklist),
    ("missing_task_keys", missing_task_keys),
    ("domain_mismatch", domain_mismatch),
]:
    if arr:
        print(f"\n{name} examples:")
        for x in arr[:5]:
            print(" ", x)
