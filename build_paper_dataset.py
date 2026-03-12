import os
import json
import glob
import random
from typing import Any, Dict, List

# ------------------------------
# Config
# ------------------------------
SEED = 20260311
SOURCE_BASE = r"c:/Users/54661/Desktop/工作/论文/写作/IWQOS/code/qos_lsma/dataset"
OUTPUT_BASE = r"c:/Users/54661/Desktop/工作/论文/写作/IWQOS/code/qos_lsma/dataset_paper_2378"

TARGET_COUNTS = {
    "Tourism": 518,
    "Finance": 463,
    "Technology": 497,
    "Medical": 432,
    "Cross_Domain": 468,
}

TASK_DOMAIN_MAP = {
    "Tourism": "tourism",
    "Finance": "finance",
    "Technology": "technology",
    "Medical": "medical",
    "Cross_Domain": "cross-domain",
}


def _ensure_list(v: Any) -> List[Any]:
    if isinstance(v, list):
        return v
    if v is None:
        return []
    return [v]


def clean_record(data: Dict[str, Any], folder_domain: str) -> Dict[str, Any]:
    """Normalize one json record to paper format:
    {
      user_query: str,
      domain: str,
      TaskList: [
        {id, description, domain, required_inputs}
      ]
    }
    """
    out: Dict[str, Any] = {}

    # user_query
    user_query = data.get("user_query")
    if not isinstance(user_query, str) or not user_query.strip():
        # fallback from task descriptions
        tl = data.get("TaskList") if isinstance(data.get("TaskList"), list) else []
        descs = []
        for t in tl:
            if isinstance(t, dict) and isinstance(t.get("description"), str):
                d = t["description"].strip()
                if d:
                    descs.append(d)
        if descs:
            user_query = "Please complete the following tasks: " + " ; ".join(descs)
        else:
            user_query = "Please complete this service composition request."
    out["user_query"] = user_query.strip()

    # domain (top-level)
    out["domain"] = folder_domain

    # TaskList
    raw_tasks = data.get("TaskList")
    raw_tasks = raw_tasks if isinstance(raw_tasks, list) else []

    clean_tasks: List[Dict[str, Any]] = []
    for i, t in enumerate(raw_tasks, start=1):
        if not isinstance(t, dict):
            continue

        desc = t.get("description")
        if not isinstance(desc, str) or not desc.strip():
            desc = f"Subtask {i}: execute required operation."

        req_inputs = t.get("required_inputs", [])
        req_inputs = _ensure_list(req_inputs)

        # normalize task domain by folder
        task_domain = TASK_DOMAIN_MAP[folder_domain]

        clean_tasks.append(
            {
                "id": i,
                "description": desc.strip(),
                "domain": task_domain,
                "required_inputs": req_inputs,
            }
        )

    if not clean_tasks:
        clean_tasks = [
            {
                "id": 1,
                "description": "Analyze the request and invoke suitable services.",
                "domain": TASK_DOMAIN_MAP[folder_domain],
                "required_inputs": [],
            }
        ]

    out["TaskList"] = clean_tasks
    return out


def load_domain_records(folder_domain: str) -> List[Dict[str, Any]]:
    folder = os.path.join(SOURCE_BASE, folder_domain)
    files = glob.glob(os.path.join(folder, "*.json"))
    records: List[Dict[str, Any]] = []

    for fp in files:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                continue
            rec = clean_record(data, folder_domain)
            records.append(rec)
        except Exception:
            continue
    return records


def sample_to_target(records: List[Dict[str, Any]], target: int, rng: random.Random) -> List[Dict[str, Any]]:
    if len(records) >= target:
        return rng.sample(records, target)

    # upsample with replacement when shortage exists
    out = records.copy()
    need = target - len(out)
    if records:
        out.extend(rng.choices(records, k=need))
    else:
        # impossible edge case fallback
        for _ in range(need):
            out.append(
                {
                    "user_query": "Please complete this service composition request.",
                    "domain": "Unknown",
                    "TaskList": [
                        {
                            "id": 1,
                            "description": "Analyze the request and invoke suitable services.",
                            "domain": "unknown",
                            "required_inputs": [],
                        }
                    ],
                }
            )
    return out


def export_domain(folder_domain: str, samples: List[Dict[str, Any]]) -> None:
    out_dir = os.path.join(OUTPUT_BASE, folder_domain)
    os.makedirs(out_dir, exist_ok=True)

    # deterministic naming
    for i, rec in enumerate(samples):
        out_fp = os.path.join(out_dir, f"response_{i}.json")
        with open(out_fp, "w", encoding="utf-8") as f:
            json.dump(rec, f, ensure_ascii=False, indent=2)


def main() -> None:
    rng = random.Random(SEED)

    # reset output
    if os.path.isdir(OUTPUT_BASE):
        for root, _, files in os.walk(OUTPUT_BASE):
            for fn in files:
                if fn.endswith(".json"):
                    try:
                        os.remove(os.path.join(root, fn))
                    except Exception:
                        pass
    os.makedirs(OUTPUT_BASE, exist_ok=True)

    summary = {}
    total = 0

    for domain, target in TARGET_COUNTS.items():
        records = load_domain_records(domain)
        sampled = sample_to_target(records, target, rng)
        export_domain(domain, sampled)

        summary[domain] = {
            "source_valid": len(records),
            "target": target,
            "exported": len(sampled),
            "upsampled": max(0, target - len(records)),
        }
        total += len(sampled)

    summary["total_exported"] = total
    summary["paper_target_total"] = sum(TARGET_COUNTS.values())

    with open(os.path.join(OUTPUT_BASE, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("Done. Exported dataset to:", OUTPUT_BASE)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
