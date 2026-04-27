import torch
import cv2
import glob
import numpy as np
from skimage.transform import estimate_transform, warp
from src.smirk_encoder import SmirkEncoder
from src.FLAME.FLAME import FLAME
from src.renderer.renderer import Renderer
import argparse
import os
import re
import src.utils.masking as masking_utils
from utils.mediapipe_utils import run_mediapipe
from datasets.base_dataset import create_mask
import torch.nn.functional as F
from typing import List, Union
import pickle
import time
from pathlib import Path

# python datawheel_sample.py --checkpoint pretrained_models/SMIRK_em1.pt --crop --render_orig --Track_Dirt /data/haiyang/wan22_datawheel/out_exp_11_prp1000/

def crop_face(frame, landmarks, scale=1.0, image_size=224):
    left = np.min(landmarks[:, 0])
    right = np.max(landmarks[:, 0])
    top = np.min(landmarks[:, 1])
    bottom = np.max(landmarks[:, 1])

    h, w, _ = frame.shape
    old_size = (right - left + bottom - top) / 2
    center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])

    size = int(old_size * scale)

    # crop image
    src_pts = np.array([[center[0] - size / 2, center[1] - size / 2], [center[0] - size / 2, center[1] + size / 2],
                        [center[0] + size / 2, center[1] - size / 2]])
    DST_PTS = np.array([[0, 0], [0, image_size - 1], [image_size - 1, 0]])
    tform = estimate_transform('similarity', src_pts, DST_PTS)
    return tform


def save_coeffs_as_pickle(
    params: dict,
    out_path: str,
) -> str:
    
    with open(out_path, "wb") as f:
            pickle.dump(params, f, protocol=pickle.HIGHEST_PROTOCOL)
    return out_path

def inference_one(args, video_path, debug_path, params_path):
    start_time_total = time.time()
    cap = cv2.VideoCapture(video_path)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if not cap.isOpened():
        print('Error opening video file')
        exit()

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # calculate size of output video
    if args.render_orig:
        out_width = video_width
        out_height = video_height
    else:
        out_width = input_image_size
        out_height = input_image_size

    if args.use_smirk_generator:
        out_width *= 3
    else:
        out_width *= 2


    cap_out = cv2.VideoWriter(f"{debug_path}", cv2.VideoWriter_fourcc(*'mp4v'), video_fps, (out_width, out_height))
    
    exp_params = []
    cam_params = []
    idn_params = []
    jaw_params = []
    eyelid_params = []
    pose_params = []
    while True:
        ret, image = cap.read()

        if not ret:
            break
    
        kpt_mediapipe = run_mediapipe(image)

        # crop face if needed
        if args.crop:
            if (kpt_mediapipe is None):
                print('Could not find landmarks for the image using mediapipe and cannot crop the face. Exiting...')
                exit()
            
            kpt_mediapipe = kpt_mediapipe[..., :2]

            tform = crop_face(image,kpt_mediapipe,scale=1.4,image_size=input_image_size)
            
            cropped_image = warp(image, tform.inverse, output_shape=(224, 224), preserve_range=True).astype(np.uint8)

            cropped_kpt_mediapipe = np.dot(tform.params, np.hstack([kpt_mediapipe, np.ones([kpt_mediapipe.shape[0],1])]).T).T
            cropped_kpt_mediapipe = cropped_kpt_mediapipe[:,:2]
        else:
            cropped_image = image
            cropped_kpt_mediapipe = kpt_mediapipe

        
        cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
        cropped_image = cv2.resize(cropped_image, (224,224))
        cropped_image = torch.tensor(cropped_image).permute(2,0,1).unsqueeze(0).float()/255.0
        cropped_image = cropped_image.to(args.device)

        outputs = smirk_encoder(cropped_image)
        flame_output = flame.forward(outputs)
        renderer_output = renderer.forward(flame_output['vertices'], outputs['cam'],
                                            landmarks_fan=flame_output['landmarks_fan'], landmarks_mp=flame_output['landmarks_mp'])
        
        rendered_img = renderer_output['rendered_img']

        # exp_params.append(list(outputs['expression_params'][0].detach().cpu().numpy()))
        # cam_params.append(list(outputs['cam'][0].detach().cpu().numpy()))
        # idn_params.append(list(outputs['shape_params'][0].detach().cpu().numpy()))
        # jaw_params.append(list(outputs['jaw_params'][0].detach().cpu().numpy()))
        # eyelid_params.append(list(outputs['eyelid_params'][0].detach().cpu().numpy()))
        # pose_params.append(list(outputs['pose_params'][0].detach().cpu().numpy()))

        exp_params.append(outputs['expression_params'][0].detach().cpu().numpy())
        cam_params.append(outputs['cam'][0].detach().cpu().numpy())
        idn_params.append(outputs['shape_params'][0].detach().cpu().numpy())
        jaw_params.append(outputs['jaw_params'][0].detach().cpu().numpy())
        eyelid_params.append(outputs['eyelid_params'][0].detach().cpu().numpy())
        pose_params.append(outputs['pose_params'][0].detach().cpu().numpy())

        ## expression = exp_params + jaw_params + eyelid_params
        ## head_pose  = pose_params

        if args.render_orig:
            if args.crop:
                rendered_img_numpy = (rendered_img.squeeze(0).permute(1,2,0).detach().cpu().numpy()*255.0).astype(np.uint8)               
                rendered_img_orig = warp(rendered_img_numpy, tform, output_shape=(video_height, video_width), preserve_range=True).astype(np.uint8)
                # back to pytorch to concatenate with full_image
                rendered_img_orig = torch.Tensor(rendered_img_orig).permute(2,0,1).unsqueeze(0).float()/255.0
            else:
                rendered_img_orig = F.interpolate(rendered_img, (video_height, video_width), mode='bilinear').cpu()

            full_image = torch.Tensor(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).permute(2,0,1).unsqueeze(0).float()/255.0
            grid = torch.cat([full_image, rendered_img_orig], dim=3)
        else:
            grid = torch.cat([cropped_image, rendered_img], dim=3)


        grid_numpy = grid.squeeze(0).permute(1,2,0).detach().cpu().numpy()*255.0
        grid_numpy = grid_numpy.astype(np.uint8)
        grid_numpy = cv2.cvtColor(grid_numpy, cv2.COLOR_BGR2RGB)
        cap_out.write(grid_numpy)

    cap.release()
    cap_out.release()
    
    ret_dirt = {'expression': exp_params, 
                'pose': pose_params,
                'eyelid': eyelid_params, 
                'jaw': jaw_params, 
                'identity': idn_params, 
                'camera_pose': cam_params,
                }
    # outputs
    save_coeffs_as_pickle(ret_dirt, params_path)
    end_time_total = time.time()
    total_time = end_time_total - start_time_total
    print(f"{video_path} running time: {total_time:.4f} seconds")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_path', type=str, default='samples/mead_90.png', help='Path to the input image/video')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run the model on')
    parser.add_argument('--checkpoint', type=str, default='trained_models/SMIRK_em1.pt', help='Path to the checkpoint')
    parser.add_argument('--crop', action='store_true', help='Crop the face using mediapipe')
    parser.add_argument('--out_path', type=str, default='output', help='Path to save the output (will be created if not exists)')
    parser.add_argument('--use_smirk_generator', action='store_true', help='Use SMIRK neural image to image translator to reconstruct the image')
    parser.add_argument('--render_orig', action='store_true', help='Present the result w.r.t. the original image/video size')
    parser.add_argument('--Track_Dirt', type=str, default='/data/haiyang/wan22_datawheel/out_exp_11_prp1000/', help='')

    args = parser.parse_args()

    input_image_size = 224
    
    # ----------------------- initialize configuration ----------------------- #
    smirk_encoder = SmirkEncoder().to(args.device)
    checkpoint = torch.load(args.checkpoint)
    checkpoint_encoder = {k.replace('smirk_encoder.', ''): v for k, v in checkpoint.items() if 'smirk_encoder' in k} # checkpoint includes both smirk_encoder and smirk_generator

    smirk_encoder.load_state_dict(checkpoint_encoder)
    smirk_encoder.eval()

    # ---- visualize the results ---- #

    flame = FLAME().to(args.device)
    renderer = Renderer(render_full_head=False).to(args.device)

    video_list = sorted(glob.glob(args.Track_Dirt + "/*.mp4"))

    for video_path in video_list:
        try:
            # m = re.search(r'/out_(exp_\d+)_prp\d+/(\d+)_[^/]*\.mp4$', video_path)
            # exp_tag, clip_id = m.groups()   # exp_tag='exp_25', clip_id='00000'
            p = Path(video_path)
            dir_name  = p.parent.name          # 'prompt_1_3000'
            file_stub = p.stem                  # '0000_happy_subtle'
            
            path_params = '/data/haiyang/wan22_datas/HuMo/T3D_v3/coeff/%s/%s.pkl'%(dir_name, file_stub)
            path_debug = '/data/haiyang/wan22_datas/HuMo/T3D_v3/debug/%s/%s.mp4'%(dir_name, file_stub)

            os.makedirs('/data/haiyang/wan22_datas/HuMo/T3D_v3/coeff/%s/'%dir_name,exist_ok=True)
            os.makedirs('/data/haiyang/wan22_datas/HuMo/T3D_v3/debug/%s/'%dir_name,exist_ok=True)
            save_frames = []
            save_params = []

            inference_one(args, video_path, path_debug, path_params)
        except:
            continue


