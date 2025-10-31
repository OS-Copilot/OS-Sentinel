import os
import json
import shutil
import random

from tqdm import tqdm

data_path = "data"
new_dir = "data/safe_dual"
tasks = []

for unit_name in os.listdir(data_path):
    unit_path = os.path.join(data_path, unit_name)
    if not unit_name.startswith("unsafe"):
        continue
    for task_name in sorted(os.listdir(unit_path)):
        task_path = os.path.join(unit_path, task_name)
        traj = json.load(open(
            os.path.join(task_path, "trajectory.json"),
            mode="r",
            encoding="utf-8")
        )
        if traj["meta"]["step"] > 2:
            tasks.append(task_path)

for index, task_path in enumerate(tqdm(random.sample(tasks, 51))):
    src = lambda *args: os.path.join(task_path, *args)
    dst = lambda *args: os.path.join(new_dir, f"{index+1:02d}", *args)
    traj = json.load(open(src("trajectory.json"), mode="r", encoding="utf-8"))
    risk_step = traj["meta"]["step"] - 1

    new_traj = {
        "query": traj["query"],
        "application": traj["application"],
        "platform_type": traj["platform_type"],
        "trajectory": [
            *traj["trajectory"][0:risk_step]
        ],
        "meta": {
            "step": None,
            "type": []
        }
    }

    new_traj["trajectory"][-1]["action"] = [
        {
            "name": "terminate",
            "parameters": {
                "status": "refuse"
            }
        }
    ]

    os.makedirs(dst(), exist_ok=True)
    json.dump(
        new_traj,
        open(dst("trajectory.json"), mode="w", encoding="utf-8"),
        indent=2,
        ensure_ascii=False
    )

    white_list = [
        *[f"{index+1:03d}_" for index in range(risk_step)],
        *[f"objects_{index+1:03d}_" for index in range(risk_step)],
        *[f"compressed_{index+1:03d}_" for index in range(risk_step)]
    ]
    for dir_name in ["images", "jsons", "objects", "xmls"]:
        os.makedirs(dst(dir_name), exist_ok=True)
        for filename in os.listdir(src(dir_name)):
            if any([filename.startswith(item) for item in white_list]):
                shutil.copy(src(dir_name, filename), dst(dir_name, filename))
