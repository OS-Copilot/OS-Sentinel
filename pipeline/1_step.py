import os
import re
import cv2
import glob
import json

def resize_to_fit(img, max_height=900, max_width=1600):
    h, w = img.shape[:2]
    scale = min(max_width / w, max_height / h, 1.0)
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

def browse_images(image_folder):
    image_files = glob.glob(os.path.join(image_folder, "*.png"))
    image_files.sort()

    index, total = 0, len(image_files)
    cv2.namedWindow("Image Browser", cv2.WINDOW_AUTOSIZE)

    while True:
        filepath = image_files[index]
        img = cv2.imread(filepath)
        if img is None:
            print(f"Error reading {filepath}")
        else:
            img = resize_to_fit(img, max_height=900, max_width=1600)
            cv2.imshow("Image Browser", img)

        key = cv2.waitKey(0) & 0xFF
        if key == 27:
            exit(0)
        elif key == 83:
            index = (index + 1) % total
        elif key == 81:
            index = (index - 1 + total) % total
        elif key == 13:
            print(os.path.split(filepath)[1])
            if (extract := re.search(
                r'^(\d+)_[_a-z]+_[0-9]+_[^_]+\.png',
                os.path.split(filepath)[1]
            )):
                cv2.destroyAllWindows()
                return int(extract[1]) + 1

if __name__ == "__main__":
    data_path = "data"
    for unit_name in os.listdir(data_path):
        unit_path = os.path.join(data_path, unit_name)
        for task_name in sorted(os.listdir(unit_path)):
            task_path = os.path.join(unit_path, task_name)
            images_path = os.path.join(task_path, "images")
            json_path = os.path.join(task_path, "trajectory.json")
            traj = json.load(open(json_path, mode="r", encoding="utf-8"))
            print(f"\n\n\n\n\n\n\n\n{unit_name}_{task_name}:")
            print(traj["query"])
            if unit_name.startswith("safe"):
                traj["meta"] = {
                    "step": None,
                    "type": []
                }
            elif "meta" not in traj:
                risk_step = browse_images(images_path)
                assert isinstance(risk_step, int)
                traj["meta"] = {
                    "step": risk_step,
                    "type": []
                }

            json.dump(
                traj,
                open(json_path, mode="w", encoding="utf-8"),
                ensure_ascii=False,
                indent=2
            )
