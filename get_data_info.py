import pickle
from collections import defaultdict

filename = "store/data-info.pkl"

with open(filename, 'rb') as fp:
    data = pickle.load(fp)

print(f"Total frames: {len(data)}")

# get camera based info
all_frames = list(data.keys())
cam_keys = defaultdict(list)

# for cam in ["cam2", "cam4", "cam6", "cam8", "cam10"]:
for kk in all_frames:
    cam = kk.split('/')[-3]
    cam_keys[cam].append(kk)

cam_info = defaultdict(dict)

for cam in cam_keys:
    tot_per = 0
    tot_pt = 0
    null_pt = 0
    for files in cam_keys[cam]:
        _data = cam_keys[cam][files]
        tot_per += _data['total_person']
        tot_pt += _data['total_pt']
        null_pt += _data['null_pt']
    cam_info[cam]['person'] = tot_per
    cam_info[cam]['keypoint'] = tot_pt
    cam_info[cam]['null'] = null_pt
    print(f"Camera {cam}")
    print("*" * 20)

    print(f"\tPerson: {tot_per}")
    print(f"\KP: {tot_pt}")
    print(f"\tNull: {null_pt / tot_pt: .4f}")

total_per = sum([v['total_person'] for v in cam_info.values()])
total_kp = sum([v['total_pt'] for v in cam_info.values()])
total_null = sum([v['null'] for v in cam_info.values()])

print(total_per, total_kp, total_null)