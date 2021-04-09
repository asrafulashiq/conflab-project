from collections import defaultdict
from loguru import logger


class AnnStat(object):
    def __init__(self) -> None:
        self.reset()

    def reset(self):
        self.info_im = defaultdict(lambda: defaultdict(int))

        self.total_image = 0
        self.total_bb = 0
        self.total_kp = 0
        self.nonnull_bb = 0
        self.nonnull_kp = 0
        self.total_pt = 0
        self.null_pt = 0

    def update_ann(self, filename=""):
        self.info_im[filename]["total_bb"] += 1
        self.info_im[filename]["total_kp"] += 1

    def update_pt(self, num: int = 1, null_num: int = 0, filename=""):
        self.info_im[filename]["total_pt"] += num
        self.info_im[filename]["null_pt"] += null_num

    def update_nonnull_bb(self, filename=""):
        self.info_im[filename]["nonnull_bb"] += 1

    def update_nonnull_kp(self, filename=""):
        self.info_im[filename]["nonnull_kp"] += 1

    def stat(self):
        stats = defaultdict(int)

        for values in self.info_im.values():
            for k, v in values.items():
                stats[k] += v

        logger.info(f"total input annotation : {stats['total_bb']}")
        logger.info(
            f"total non-null person with keypoint : {stats['nonnull_kp']} ({stats['nonnull_kp']/stats['total_kp']*100:.1f}%)"
        )
        logger.info(
            f"total non-null person with bb : {stats['nonnull_bb']} ({stats['nonnull_bb']/stats['total_bb']*100:.1f}%)"
        )

        logger.info(
            f"Percentage null kp: {stats['null_pt'] / stats['total_pt'] * 100 : 1f}"
        )

    def null_kp(self, filename=""):
        if self.info_im[filename]["total_pt"] == 0:
            return 1.0
        return self.info_im[filename]["null_pt"] / self.info_im[filename][
            "total_pt"]
