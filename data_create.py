from typing import Dict, List
import os
from utils.utils_det import configure_logger
import hydra
import cv2

import random
from detectron2.data import DatasetCatalog, MetadataCatalog
from data_loading.conflab_dataset import *
from tqdm import tqdm
from detectron2.utils.visualizer import Visualizer


@hydra.main(config_name='config', config_path='conf')
def main(args):
    configure_logger(args)
    # args.coco_json_path = os.path.join(".", args.coco_json_path)
    register_conflab_dataset(args)

    if args.data_plot:
        dataset_dicts: List[Dict] = DatasetCatalog.get(args.test_dataset)
        metadata = MetadataCatalog.get(args.test_dataset)
        samples = random.sample(
            dataset_dicts, min(args.data_create_num_vis, len(dataset_dicts)))
        for d in tqdm(samples):
            img = cv2.imread(d["file_name"])
            visualizer = Visualizer(img[:, :, ::-1],
                                    metadata=metadata,
                                    scale=1.2)
            out = visualizer.draw_dataset_dict(d)
            cv2_im = out.get_image()[:, :, ::-1]
            cv2.imshow(d["file_name"], cv2_im)
            cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
