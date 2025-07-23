import os
import cv2
import yaml
import json
import base64
import regex as re
import numpy as np

from PIL import Image
from itertools import tee


_WORK_PATH = os.environ["MOBILE_SAFETY_HOME"]


def load_config(config_path=f"{_WORK_PATH}/config/foundation_model_config.yaml"):
    configs = dict(os.environ)
    with open(config_path, "r") as file:
        yaml_data = yaml.safe_load(file)
    configs.update(yaml_data)
    return configs


def parse_response(response):
    json_pattern = re.compile(r"\{.*?\}", re.DOTALL)
    match = json_pattern.search(response)

    if match:
        json_string = match.group(0)
        try:
            response_dict = json.loads(json_string.strip())
        except:
            response_dict = {"action": None, "context": ""}
    else:
        response_dict = {"action": None, "context": ""}
    response_dict["raw_response"] = response
    return response_dict


def parse_obs(
    obs,
    height_width=None,
    skip_non_leaf=False,
    attribute_check=False,
    attribute_bbox=False,
):
    parsed_obs = []
    parsed_bbox = []
    xpath_list = []

    raw_obs, obs = tee(obs)
    for _, elem in obs:
        parsed_elem = {}

        if not "bounds" in elem.attrib:
            continue

        # xpath
        xpath_list.append(generate_xpath(elem))

        # category description
        resource_id = ""
        if "resource-id" in elem.attrib and elem.attrib["resource-id"]:
            resource_desc = elem.attrib["resource-id"]
            resource_id = resource_desc.split("/")[-1]
        if skip_non_leaf and resource_id == "":
            continue

        # class
        class_name = ""
        if "class" in elem.attrib and elem.attrib["class"]:
            class_name = f"{elem.attrib['class']}"
            class_name = class_name.split(".")[-1]

        # detailed description
        content_desc = ""
        if "content-desc" in elem.attrib and (elem.attrib["content-desc"] != ""):
            content_desc = elem.attrib["content-desc"]

        # text
        text_desc = ""
        if "text" in elem.attrib and (elem.attrib["text"] != ""):
            text_desc = elem.attrib["text"]

        # checked
        checked = "false"
        if "checked" in elem.attrib and (elem.attrib["checked"] != ""):
            if elem.attrib["checked"] == "true":
                checked = "true"
        if "selected" in elem.attrib and (elem.attrib["selected"] != ""):
            if elem.attrib["selected"] == "true":
                checked = "true"
        if "focused" in elem.attrib and (elem.attrib["focused"] != ""):
            if elem.attrib["focused"] == "true":
                checked = "true"

        # bbox location
        bounds = elem.attrib["bounds"][1:-1].split("][")
        x1, y1 = map(int, bounds[0].split(","))
        x2, y2 = map(int, bounds[1].split(","))
        # bbox corner cases
        if ("view all" in content_desc) and ("comments" in content_desc):
            x1 = 25
            x2 = 300
        parsed_bbox.append(((x1, y1), (x2, y2)))

        # parsed_elem
        parsed_elem["numeric_tag"] = len(parsed_obs)
        if resource_id != "":
            parsed_elem["resource_id"] = resource_id
        if class_name != "":
            parsed_elem["class"] = class_name
        if content_desc != "":
            parsed_elem["content_description"] = content_desc
        if text_desc != "":
            parsed_elem["text"] = text_desc
        if attribute_check:
            parsed_elem["checked"] = checked
        if attribute_bbox:
            height, width = height_width
            parsed_elem["bbox location"] = (
                f"(({x1/width:0.2f}, {y1/height:0.2f}), ({x2/width:0.2f}, {y2/height:0.2f}))"
            )
        parsed_obs.append(parsed_elem)

    return raw_obs, parsed_obs, parsed_bbox, xpath_list


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def PIL2OpenCV(pil_image):
    numpy_image = np.array(pil_image)
    opencv_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
    return opencv_image


def OpenCV2PIL(opencv_image):
    color_coverted = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(color_coverted)
    return pil_image


def generate_xpath(element):
    xpath = f"//{element.get('class')}"

    additional = []
    # for key in element.keys():
    #     if key in ['class', 'index']:
    #         continue
    #     additional.append(f'@{key}="{element.get(key)}"')
    if element.get("resource-id"):
        additional.append(f'@resource-id="{element.get("resource-id")}"')
    if element.get("text"):
        additional.append(f'@text="{element.get("text")}"')
    if element.get("content-desc"):
        additional.append(f'@content-desc="{element.get("content-desc")}"')

    if additional:
        conditions = " and ".join(additional)
        xpath = f"{xpath}[{conditions}]"

    return xpath
