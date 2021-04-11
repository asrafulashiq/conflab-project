from datetime import datetime
import os
from typing import List, Optional
import numpy as np
import cv2
from detectron2.utils.visualizer import Visualizer
from detectron2.engine import DefaultPredictor
from detectron2.data.catalog import Metadata
from detectron2.data import transforms as T
from omegaconf.dictconfig import DictConfig
from tqdm import tqdm


def configure_logger(args):
    from loguru import logger
    logger.configure(handlers=[
        dict(sink=lambda msg: tqdm.write(msg, end=''),
             level='DEBUG',
             colorize=True,
             format=
             "<green>{time: MM-DD at HH:mm}</green> <level>{message}</level>",
             enqueue=True),
    ])
    now = datetime.now()
    os.makedirs(args.log_dir, exist_ok=True)
    logfile = os.path.join(
        args.log_dir,
        f"{args.log_prefix}_{now.month:02d}_{now.day:02d}_{now.hour:03d}.txt")
    logger.add(sink=logfile,
               mode='w',
               format="{time: MM-DD at HH:mm} | {message}",
               level="DEBUG",
               enqueue=True)


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