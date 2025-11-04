import os
import json

data_path = "data"

for unit_name in os.listdir(data_path):
    unit_path = os.path.join(data_path, unit_name)

    if not unit_name.startswith("unsafe"):
        continue

    for task_name in sorted(os.listdir(unit_path)):
        task_path = os.path.join(unit_path, task_name)
        inner_id = f"{unit_name}_{task_name}"
        sampler = unit_name.split("_")[1].replace("msb", "lzmz")

        json_path = os.path.join(task_path, "trajectory.json")
        traj = json.load(open(json_path, mode="r", encoding="utf-8"))
        inst = traj["query"]
        step = traj["meta"]["step"]

        print(f"{inner_id}\t{sampler}\t{step}\t{inst}")
