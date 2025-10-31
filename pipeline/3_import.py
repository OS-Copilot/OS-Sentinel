import os
import csv
import json

csv_path = "unsafe_types.csv"
data_path = "data"

with open(csv_path, mode="r", encoding="utf-8") as r:
    reader = csv.reader(r)
    data = [row for row in reader][1:]

    field = [
        "Destructive Actions",
        "Malicious Use",
        "Privacy Violations",
        "Security Mechanism Evasion",
        "Prompt Injection",
        "UI Interference Attacks",
        "Harmful Content Generation and Dissemination",
        "Resource Abuse",
        "Legal and Compliance Issues",
        "Over-Automation"
    ]

    for task in data:
        typing = [field[index] for index, item in enumerate(task[4:14]) if len(item) > 0]
        _1, _2, _3 = task[0].split("_")
        json_path = os.path.join(data_path, f"{_1}_{_2}", _3, "trajectory.json")
        traj = json.load(open(json_path, mode="r", encoding="utf-8"))
        traj["meta"]["step"] = int(task[2])
        traj["meta"]["type"] = typing
        json.dump(
            traj,
            open(json_path, mode="w", encoding="utf-8"),
            ensure_ascii=False,
            indent=2
        )
