# Import SixDRepNet
from sixdrepnet import SixDRepNet
import cv2
import glob
import os
import argparse
import numpy as np

# CUDA_VISIBLE_DEVICES=6 python fix_pose.py --file_path /data/haiyang/wan22_datas/HuMo/Animations/spectre/tmp_ravdess/Actor_01/

parser = argparse.ArgumentParser(description='fix pose')
parser.add_argument('--file_path', default='/data/haiyang/wan22_datas/HuMo/Animations/spectre/tmp_ravdess/Actor_01/', type=str,)
args = parser.parse_args()
# Create model
# Weights are automatically downloaded
model = SixDRepNet()

# video_files = sorted(glob.glob("/data/haiyang/wan22_datas/HuMo/Animations/spectre/tmp/prompt_17_3000/" + "/*/"))
video_files = sorted(glob.glob(args.file_path + "/*/"))[::-1]
for input_file in video_files:
    fixedpose_path = input_file.replace("tmp_ravdess", "RAVDESS_v4/fixed_pose/")
    fixedpose_save = fixedpose_path[:-1] + ".npy"
    os.makedirs(fixedpose_path[:-1].rsplit('/', 1)[0], exist_ok=True)
    fixed_return = []
    import pdb; pdb.set_trace()
    for img_path in sorted(glob.glob(input_file + "/*.jpg")):
        img = cv2.imread(img_path)
        pitch, yaw, roll = model.predict(img)
        fixed_pose = np.array([pitch[0], yaw[0], roll[0]])
        fixed_return.append(fixed_pose)
    npy_array = np.array(fixed_return)
    np.save(fixedpose_save, npy_array)
    print("[%s]"%input_file)