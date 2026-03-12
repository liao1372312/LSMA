import os
import glob
import json

base = r"c:/Users/54661/Desktop/工作/论文/写作/IWQOS/code/qos_lsma/dataset"
domains = ["Tourism", "Finance", "Technology", "Medical", "Cross_Domain"]

print("Domain counts and schema summary:")
for d in domains:
    files = glob.glob(os.path.join(base, d, "*.json"))
    print(f"{d}: {len(files)}")
    if files:
        p = sorted(files)[0]
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        print("  sample_file=", os.path.basename(p))
        if isinstance(data, dict):
            print("  top_keys=", list(data.keys()))
            if isinstance(data.get("TaskList"), list) and data["TaskList"]:
                print("  task_keys=", list(data["TaskList"][0].keys()))
        else:
            print("  top_type=", type(data).__name__)
    print("---")

print("Other dirs:")
for d in ["test_data", "Cross_Domain_back", "error"]:
    p = os.path.join(base, d)
    if os.path.isdir(p):
        n = sum(
            len(glob.glob(os.path.join(root, "*.json")))
            for root, _, _ in os.walk(p)
        )
        print(f"{d}: {n}")
