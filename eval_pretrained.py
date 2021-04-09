"""
Evaluate coco pretrained models
"""
import os
from typing import Dict, List
from detectron2.data.catalog import MetadataCatalog
from loguru import logger
import hydra
from detectron2 import model_zoo
from omegaconf import OmegaConf, DictConfig
from typing import List, Optional
import random
import numpy as np
import cv2
from detectron2.utils.visualizer import Visualizer
from detectron2.engine import DefaultPredictor
from detectron2.data.catalog import Metadata
from detectron2.data import transforms as T

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg

from detectron2.evaluation import COCOEvaluator
from detectron2.data import build_detection_test_loader
from detectron2.data import (DatasetCatalog, DatasetMapper,
                             build_detection_train_loader,
                             build_detection_test_loader)

from detectron2.engine import DefaultTrainer, launch, default_setup, DefaultPredictor

from data_loading import conflab_dataset
from utils import visualize_det2, create_train_augmentation, create_test_augmentation


class Trainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        mapper = DatasetMapper(cfg,
                               is_train=True,
                               augmentations=create_train_augmentation(cfg))
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        mapper = DatasetMapper(cfg,
                               is_train=False,
                               augmentations=create_test_augmentation(cfg))
        return build_detection_test_loader(cfg, dataset_name, mapper=mapper)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, ("bbox", "keypoints"),
                             False,
                             output_dir=output_folder)


class Predictor(DefaultPredictor):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.aug = T.Resize((cfg.image_h, cfg.image_w))


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()

    cfg.merge_from_file(model_zoo.get_config_file(args.model_zoo))
    cfg.DATASETS.TRAIN = (args.train_dataset, )
    cfg.DATASETS.TEST = (args.test_dataset, )
    cfg.DATALOADER.NUM_WORKERS = args.num_workers
    cfg.OUTPUT_DIR = args.output_dir
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    cfg.image_w = args.size[0]
    cfg.image_h = args.size[1]

    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(args.model_zoo)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.roi_thresh  # set a custom testing threshold

    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    if args.visualize is False:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
        res = Trainer.test(cfg, model)
        logger.info(res)
        return res
    else:
        test_dataset: List[Dict] = DatasetCatalog.get(args.test_dataset)
        # metadata = MetadataCatalog.get(args.test_dataset)
        metadata = MetadataCatalog.get("coco_2014_val")
        predictor = Predictor(cfg)
        visualize_det2(test_dataset,
                       predictor,
                       metadata=metadata,
                       vis_conf=args.vis)


@hydra.main(config_name='config', config_path='conf')
def hydra_main(args: DictConfig):
    logger.info("Command Line Args:\n{}".format(
        OmegaConf.to_yaml(args, resolve=True)))
    conflab_dataset.register_conflab_dataset(args)

    launch(main, args.num_gpus, args=(args, ))


if __name__ == "__main__":
    hydra_main()