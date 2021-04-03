from typing import Dict, List, Mapping
from utils import cocosplit
from detectron2.structures import BoxMode
import os
import hydra
import numpy as np
import parse
import json
from tqdm import tqdm
import cv2
from loguru import logger
import random
from omegaconf import DictConfig
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
import seaborn as sns


def extract_file_info(filename: str) -> Mapping:
    # filename: cam2_vid2_seg7.json
    parsed_info = parse.parse("cam{cam}_vid{vid}_seg{seg}.json", str(filename))
    return parsed_info


def scale_kp_xy(kp: List[float], w, h) -> List[float]:
    n = len(kp) // 2
    new_kp = [1] * (3 * n)
    new_kp[0::3] = [int(i * w) if i else i for i in kp[0::2]]
    new_kp[1::3] = [int(i * h) if i else i for i in kp[1::2]]
    new_kp[2::3] = [1 if i is not None else 0 for i in kp[0::2]]
    return new_kp


def convert_conflab_to_coco(img_root_dir: str,
                            annotation_dir: str) -> List[Dict]:
    counter_image = 0
    counter = 0
    coco_data = {"info": {}, "images": [], "annotations": [], "categories": []}

    dict_ims = {}  # all_images that have been seen so far

    for ann_file in os.listdir(annotation_dir)[5:]:
        parsed_info = extract_file_info(ann_file)

        img_ann_dir = f"cam{parsed_info['cam']}/vid{parsed_info['vid']}-seg{parsed_info['seg']}-scaled-denoised"
        img_dir = os.path.join(img_root_dir, img_ann_dir)
        if not os.path.exists(img_dir):
            logger.warning(f"Directory {img_dir} does not exist")
            continue

        with open(os.path.join(annotation_dir, ann_file), 'r') as fp:
            full_data = json.load(fp)

        coco_data["info"] = full_data["info"]
        coco_data["categories"] = full_data["categories"]

        data_kp = full_data['annotations']['skeletons']

        for _, v in tqdm(enumerate(data_kp)):
            # v contain info for each image
            annotations_for_image = list(v.items())
            record = {}
            filename = os.path.join(
                img_dir,
                f"{annotations_for_image[0][1]['image_id']+1:06d}.jpg")
            if not os.path.exists(filename):
                logger.warning(f"{filename} does not exist")
                continue

            height, width = cv2.imread(filename).shape[:2]

            if filename not in dict_ims:
                # this is a new image file, add it to dict

                counter_image += 1
                record = dict()
                record["file_name"] = os.path.join(
                    img_ann_dir,
                    f"{annotations_for_image[0][1]['image_id']+1:06d}.jpg")
                record["id"] = counter_image
                record["height"] = height
                record["width"] = width
                dict_ims[filename] = record

                coco_data["images"].append(record)

            for _, anno in annotations_for_image:
                counter += 1
                record_ann = {}
                record_ann["id"] = counter
                record_ann["image_id"] = dict_ims[filename]["id"]
                record_ann["category_id"] = 1  # NOTE: person category

                has_none = any(x is None for x in anno["keypoints"])
                if has_none:
                    # logger.warning(f"has none in {filename}")
                    continue

                anno["keypoints"] = scale_kp_xy(anno["keypoints"], width,
                                                height)
                # utilities for bbox and segm
                px = anno["keypoints"][0::3]
                py = anno["keypoints"][1::3]

                fn_none = lambda x: [i for i in x if i is not None]
                x1, y1, x2, y2 = [
                    min(fn_none(px)),
                    min(fn_none(py)),
                    max(fn_none(px)),
                    max(fn_none(py))
                ]

                bbox = [x1, y1, x2 - x1, y2 - y1]
                record_ann["bbox"] = bbox
                record_ann["segmentation"] = []
                record_ann["keypoints"] = anno["keypoints"]
                record_ann["area"] = int(
                    (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
                record_ann["iscrowd"] = 0

                coco_data["annotations"].append(record_ann)

            # if counter_image > 100:
            #     return coco_data
    return coco_data


def register_conflab_dataset(args: DictConfig):
    if args.create_coco:
        # convert to coco
        coco_info = convert_conflab_to_coco(img_root_dir=args.img_root_dir,
                                            annotation_dir=args.ann_dir)
        with open(args.coco_json_path, "w") as fp:
            json.dump(coco_info, fp)

        cocosplit.split(args.coco_json_path,
                        args.coco_json_path_train,
                        args.coco_json_path_test,
                        split=args.train_split)

    keypoints, keypoint_connection_rules, keypoint_flip_map = get_kp_names()

    def _register(dataset, ann_path):
        register_coco_instances(dataset, {}, ann_path, args.img_root_dir)
        # set meta data catalog
        MetadataCatalog.get(dataset).keypoint_names = keypoints
        MetadataCatalog.get(
            dataset).keypoint_connection_rules = keypoint_connection_rules
        MetadataCatalog.get(dataset).keypoint_flip_map = keypoint_flip_map

    _register(args.dataset, args.coco_json_path)
    _register(args.train_dataset, args.coco_json_path_train)
    _register(args.test_dataset, args.coco_json_path_test)


def get_kp_names():
    keypoints = [
        "head", "nose", "neck", "rightShoulder", "rightElbow", "rightWrist",
        "leftShoulder", "leftElbow", "leftWrist", "rightHip", "rightKnee",
        "rightAnkle", "leftHip", "leftKnee", "leftAnkle", "rightFoot",
        "leftFoot"
    ]
    connections = [[0, 1], [0, 2], [2, 3], [2, 6], [3, 4], [4, 5], [6, 7],
                   [7, 8], [2, 9], [9, 10], [10, 11], [11, 15], [2, 12],
                   [12, 13], [13, 14], [14, 16]]
    colors = [(int(r * 255), int(g * 255), int(b * 255))
              for r, g, b in sns.color_palette(n_colors=len(connections))]
    keypoint_connection_rules = []
    for i, (a, b) in enumerate(connections):
        keypoint_connection_rules.append(
            (keypoints[a], keypoints[b], colors[i]))
    keypoint_flip_map = (('leftFoot', 'rightFoot'),
                         ('leftShoulder', 'rightShoulder'), ('leftElbow',
                                                             'rightElbow'),
                         ('leftWrist', 'rightWrist'), ('leftHip', 'rightHip'),
                         ('leftKnee', 'rightKnee'), ('leftAnkle',
                                                     'rightAnkle'))

    return keypoints, keypoint_connection_rules, keypoint_flip_map


@hydra.main(config_name='config', config_path='../conf')
def main(args):
    from detectron2.utils.visualizer import Visualizer
    args.coco_json_path = os.path.join("..", args.coco_json_path)
    register_conflab_dataset(args)

    dataset_dicts: List[Dict] = DatasetCatalog.get(args.test_dataset)
    metadata = MetadataCatalog.get(args.test_dataset)
    for d in random.sample(dataset_dicts, 5):
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=metadata, scale=0.8)
        out = visualizer.draw_dataset_dict(d)
        cv2_im = out.get_image()[:, :, ::-1]
        cv2.imshow(d["file_name"], cv2_im)
        cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
