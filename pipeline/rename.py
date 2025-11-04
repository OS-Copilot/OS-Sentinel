import os

unit_path = "data/unsafe_msb"

for index, task_name in enumerate(sorted(os.listdir(unit_path))):
    task_path = os.path.join(unit_path, task_name)
    new_task_path = os.path.join(unit_path, f"{index+1:02d}")
    os.rename(task_path, new_task_path)
