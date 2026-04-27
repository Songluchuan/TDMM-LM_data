import time
import math
import re
import sys
import os
import argparse
import glob
from skimage import io

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import numpy as np
from numpy.lib.function_base import _quantile_unchecked
import cv2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.backends import cudnn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from face_detection import RetinaFace
import matplotlib
from matplotlib import pyplot as plt
from PIL import Image
from pathlib import Path
# matplotlib.use('TkAgg')

from model import SixDRepNet
import utils
from tqdm import tqdm 

fmt = lambda s: str(Path(*Path(s).parts[:10]) / "#".join(Path(s).parts[10:]))

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description='Head pose estimation using the 6DRepNet.')
    parser.add_argument('--gpu',
                        dest='gpu_id', help='GPU device id to use [0], set -1 to use CPU',
                        default=0, type=int)
    parser.add_argument('--cam',
                        dest='cam_id', help='Camera device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--snapshot',
                        dest='snapshot', help='Name of model snapshot.',
                        default='', type=str)
    parser.add_argument('--input_dirt',
                        dest='input_dirt', help='Name of model snapshot.',
                        default='/data/haiyang/wan22_datas/HuMo/Animations/spectre/tmp/prompt_1_3000', type=str)
    parser.add_argument('--save_viz',
                        dest='save_viz', help='Save images with pose cube.',
                        default=False, type=bool)

    args = parser.parse_args()
    return args

# CUDA_VISIBLE_DEVICES=3 python ./sixdrepnet/demo.py  --snapshot 6DRepNet_300W_LP_AFLW2000.pth --gpu 0 --input_dirt /data/haiyang/wan22_datas/HuMo/Animations/spectre/tmp_ravdess/Actor_01/

transformations = transforms.Compose([transforms.Resize(224),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

if __name__ == '__main__':
    args = parse_args()
    cudnn.enabled = True
    gpu = args.gpu_id
    if (gpu < 0):
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:%d' % gpu)
    # cam = args.cam_id
    snapshot_path = args.snapshot
    model = SixDRepNet(backbone_name='RepVGG-B1g2',
                       backbone_file='',
                       deploy=True,
                       pretrained=False)

    print('Loading data.')

    detector = RetinaFace(gpu_id=gpu)

    # Load snapshot
    saved_state_dict = torch.load(os.path.join(
        snapshot_path), map_location='cpu')

    if 'model_state_dict' in saved_state_dict:
        model.load_state_dict(saved_state_dict['model_state_dict'])
    else:
        model.load_state_dict(saved_state_dict)
    model.to(device)

    # Test the Model
    model.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
    
    video_files = sorted(glob.glob(args.input_dirt + "/*/*/*/"))
    
    # /data/haiyang/wan22_datas/HuMo/Animations/spectre/tmp_MEAD/M003/video/front/angry/level_1/001
    with torch.no_grad():
        # while True:

        for input_file in tqdm(video_files):
            fixedpose_path = input_file.replace("tmp_MEAD", "MEAD_v4/fixed_pose_6dof")
            fixedpose_path = fixedpose_path.replace("video/", "")
            
            
            fixedpose_path = fmt(fixedpose_path[:-1])
            fixedpose_save = fixedpose_path + ".npy"
            os.makedirs(fixedpose_path[:-1].rsplit('/', 1)[0], exist_ok=True)
            # import pdb; pdb.set_trace()
            fixed_return = []
            try:
                for img_path in sorted(glob.glob(input_file + "/*.jpg")):
                    # ret, frame = cap.read()
                    frame = io.imread(img_path)
                    faces = detector(frame)

                    box = faces[:1][0][0]
                    landmarks = faces[:1][0][1]
                    score = faces[:1][0][2]

                    # # Print the location of each face in this image
                    if score < .85:
                        print(score)
                        break
                    x_min = int(box[0])
                    y_min = int(box[1])
                    x_max = int(box[2])
                    y_max = int(box[3])
                    bbox_width = abs(x_max - x_min)
                    bbox_height = abs(y_max - y_min)

                    x_min = max(0, x_min-int(0.2*bbox_height))
                    y_min = max(0, y_min-int(0.2*bbox_width))
                    x_max = x_max+int(0.2*bbox_height)
                    y_max = y_max+int(0.2*bbox_width)

                    img = frame[y_min:y_max, x_min:x_max]
                    img = Image.fromarray(img)
                    img = img.convert('RGB')
                    img = transformations(img)

                    img = torch.Tensor(img[None, :]).to(device)

                    R_pred = model(img)
                        
                    fixed_return.append(R_pred.cpu().numpy())
                npy_array = np.array(fixed_return)
                np.save(fixedpose_save, npy_array)
                print("[%s]"%input_file)
            except: continue
