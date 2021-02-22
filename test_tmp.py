import torch, torchvision
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
import numpy as np
import os, json, cv2, random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

import matplotlib.pyplot as plt
import cv2
from data_loading import conflab_dataset

conf_data = DatasetCatalog.get("conflab-dataset")

from detectron2.utils.visualizer import Visualizer

cfg = get_cfg()

cfg.merge_from_file(
    model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
    "COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")
predictor = DefaultPredictor(cfg)

for d in random.sample(conf_data, 10):
    img = cv2.imread(d["file_name"])

    outputs = predictor(img)
    v = Visualizer(img[:, :, ::-1],
                   MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
                   scale=0.5)

    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2.imshow("", out.get_image()[..., ::-1])
    cv2.waitKey(0)
