import numpy as np
import torch
import PIL.Image
import os
from run import estimate
from basicsr.utils.flow_util import flowwrite
from run_reds import read_image
from IPython import embed


# data_root = '/home/thor/projects/data/videosr/vimeo90k'
data_root = '/cluster/work/cvl/videosr/vimeo90k'
meta_info = os.path.expanduser('~/projects/video_transformer/basicsr/data/meta_info/meta_info_Vimeo90K_train_GT.txt')
input_folder = 'vimeo_septuplet_matlabLRx4/sequences'
output_folder = 'vimeo_septuplet_matlabLRx4_flow2/sequences'
num_frame = 7
num_half_frame = num_frame // 2
center_frame_idx = num_half_frame + 1


if __name__ == '__main__':

    with open(meta_info, 'r') as fin:
        keys = [line.split(' ')[0] for line in fin]

    for key in keys:
        if not os.path.exists(os.path.join(data_root, output_folder, key)):
            os.makedirs(os.path.join(data_root, output_folder, key))
        neighbor_idx = []
        neighbor_names = []
        center_frame_path = os.path.join(data_root, input_folder, key, f'im{center_frame_idx}.png')
        for idx in range(num_half_frame, 0, -1):
            neighbor_idx.append(center_frame_idx - idx)
            neighbor_names.append(f'p{idx}')
        for idx in range(1, num_half_frame + 1):
            neighbor_idx.append(center_frame_idx + idx)
            neighbor_names.append(f'n{idx}')

        center_frame = read_image(center_frame_path)
        for idx, name in zip(neighbor_idx, neighbor_names):
            neighbor_frame_path = os.path.join(data_root, input_folder, key, f'im{idx}.png')
            output_path = os.path.join(data_root, output_folder, key, f'im{center_frame_idx}_{name}.flo')
            neighbor_frame = read_image(neighbor_frame_path)
            flow = estimate(neighbor_frame, center_frame)

            # import cv2
            # import matplotlib.pyplot as plt
            #
            # flow = flow.numpy()
            # mag, ang = cv2.cartToPolar(flow[0], flow[1])
            # hsv = np.zeros([flow.shape[1], flow.shape[2], 3], dtype=np.float32)
            # hsv[..., 2] = 255
            # hsv[..., 0] = ang * 180 / np.pi / 2
            # hsv[..., 1] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            # hsv = hsv.astype(np.uint8)
            # rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
            #
            # plt.imshow(rgb)
            # plt.show()

            flowwrite(flow.permute(1, 2, 0).numpy(), output_path)
            # embed(); exit()
        #
        # print(center_frame_path)
        # print(neighbor_idx)
        # print(neighbor_names)
        # embed()
        # exit()
