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