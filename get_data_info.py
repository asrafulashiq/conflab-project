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
        _data = data[files]
        tot_per += _data['total_person']
        tot_pt += _data['total_pt']
        null_pt += _data['null_pt']
    tot_im = len(cam_keys[cam])

    cam_info[cam]['person'] = tot_per
    cam_info[cam]['keypoint'] = tot_pt - null_pt
    # cam_info[cam]['null'] = null_pt
    cam_info[cam]['im'] = tot_im

    print(f"Camera {cam}")
    print("*" * 20)

    print(f"\tImgaes: {tot_im}")
    print(f"\tPerson: {tot_per}")
    print(f"\tKP: {tot_pt}")
    print(f"\tNull: {null_pt / tot_pt: .4f}")

total_im = sum([v['im'] for v in cam_info.values()])
total_per = sum([v['person'] for v in cam_info.values()])
total_kp = sum([v['keypoint'] for v in cam_info.values()])
total_null = sum([v['null'] for v in cam_info.values()])

print(total_im, total_per, total_kp, total_null)