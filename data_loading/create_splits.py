"""
create train-test split and save to ./splits directory 
"""
import itertools
import os
import json

if __name__ == "__main__":
    cameras = ["cam2", "cam4", "cam6", "cam8", "cam10"]

    save_dir = "./splits"
    os.makedirs(save_dir, exist_ok=True)

    for r in range(1, 4):
        # r vs rest
        list_test_combs = list(sorted((itertools.combinations(cameras, r=r))))
        for i, test_combs in enumerate(list_test_combs):
            train_combs = (x for x in cameras if x not in test_combs)

            info = {
                "train_cam": list(train_combs),
                "test_cam": list(test_combs),
            }
            filepath = os.path.join(save_dir, f"split_{r}_{i}.txt")
            with open(filepath, 'w') as fp:
                json.dump(info, fp, indent=4, sort_keys=True)