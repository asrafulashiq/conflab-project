from datetime import datetime
import os
from typing import List, Optional
from detectron2.utils.env import seed_all_rng
import numpy as np
import cv2
from detectron2.utils.visualizer import Visualizer
from detectron2.engine import DefaultPredictor
from detectron2.data.catalog import Metadata
from detectron2.data import transforms as T
from omegaconf.dictconfig import DictConfig
import torch
from tqdm import tqdm
from detectron2.utils.logger import setup_logger
from detectron2.utils import comm


def configure_logger(args):
    now = datetime.now()
    os.makedirs(args.log_dir, exist_ok=True)
    logfile = os.path.join(
        args.log_dir,
        f"{args.name}_{args.log_prefix}{now.month:02d}_{now.day:02d}_{now.hour:03d}.txt"
    )

    rank = comm.get_rank()
    logger = setup_logger(output=logfile, distributed_rank=rank)
    return logger


def custom_setup(args):
    rank = comm.get_rank()
    logger = configure_logger(args)
    logger.info("Rank of current process: {}. World size: {}".format(
        rank, comm.get_world_size()))
    seed_all_rng(None if args.seed < 0 else args.seed + rank)

    # cudnn benchmark has large overhead. It shouldn't be used considering the small size of
    # typical validation set.
    if not (hasattr(args, "eval_only") and args.eval_only):
        torch.backends.cudnn.benchmark = args.benchmark


def visualize_det2(dataset_dicts: List[dict],
                   predictor: DefaultPredictor,
                   metadata: Optional[Metadata] = None,
                   vis_conf: DictConfig = None) -> None:
    rng = np.random.RandomState(seed=vis_conf.seed)
    samples = rng.choice(dataset_dicts, size=vis_conf.count)
    for i, d in tqdm(enumerate(samples)):
        im = cv2.imread(d["file_name"])
        outputs = predictor(im)

        # filter only `person` class
        instances = outputs["instances"]
        instances = instances[instances.pred_classes == 0]

        v = Visualizer(im[:, :, ::-1], metadata=metadata)
        out = v.draw_instance_predictions(instances.to("cpu"))
        if vis_conf.save:
            os.makedirs(vis_conf.save_folder, exist_ok=True)
            filename = os.path.join(vis_conf.save_folder, f"{i:06d}.jpg")
            cv2.imwrite(filename, out.get_image()[:, :, ::-1])
        else:
            cv2.imshow(d["file_name"], out.get_image()[:, :, ::-1])
            cv2.waitKey(0)

    cv2.destroyAllWindows()


def create_train_augmentation(cfg):
    augs = [
        T.Resize((cfg.image_h, cfg.image_w)),
        T.RandomBrightness(0.9, 1.1),
        T.RandomFlip(prob=0.3)
        # T.RandomCrop("absolute", (640, 640))
    ]
    return augs


def create_test_augmentation(cfg):
    augs = [
        T.Resize((cfg.image_h, cfg.image_w)),
    ]
    return augs