"""
Training or evaluation on different models for
    - keypoint estimation task
    - person detection task

Available models:
    - all models in detection2 model_zoo
    - our custom models
"""

import hydra
import omegaconf


@hydra.main(config_name="config", config_path="conf")
def main(cfg: omegaconf.DictConfig):
    pass


if __name__ == '__main__':
    main()