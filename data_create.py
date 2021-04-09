from typing import Dict, List, Mapping, Optional

from PIL import Image
from utils import cocosplit
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
from data_loading.conflab_dataset import *


@hydra.main(config_name='config', config_path='conf')
def main(args):
    from detectron2.utils.visualizer import Visualizer
    args.coco_json_path = os.path.join(".", args.coco_json_path)
    register_conflab_dataset(args)

    dataset_dicts: List[Dict] = DatasetCatalog.get(args.test_dataset)
    metadata = MetadataCatalog.get(args.test_dataset)
    for d in random.sample(dataset_dicts, args.data_create_num_vis):
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=metadata, scale=0.8)
        out = visualizer.draw_dataset_dict(d)
        cv2_im = out.get_image()[:, :, ::-1]
        cv2.imshow(d["file_name"], cv2_im)
        cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
