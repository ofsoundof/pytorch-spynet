import numpy as np
import torch
import PIL.Image
import os
from run import estimate
from basicsr.utils.flow_util import flowwrite
# from IPython import embed


# data_root = '/home/thor/projects/data/videosr/REDS'
data_root = '/cluster/work/cvl/videosr/REDS/'
input_folder = 'train_sharp_bicubic/X4'
output_folder = 'train_sharp_bicubic_flow/X4'
num_frame = 5
num_half_frame = num_frame // 2


def read_image(path):
    img = np.array(PIL.Image.open(path))[:, :, ::-1]
    img = img.transpose(2, 0, 1).astype(np.float32) * (1.0 / 255.0)
    return torch.FloatTensor(np.ascontiguousarray(img))


if __name__ == '__main__':

    for sequence_idx in range(270):
        if not os.path.exists(os.path.join(data_root, output_folder, f'{sequence_idx:03d}')):
            os.makedirs(os.path.join(data_root, output_folder, f'{sequence_idx:03d}'))
        for frame_idx in range(100):
            neighbor_idx = []
            neighbor_names = []
            center_frame_path = os.path.join(data_root, input_folder, f'{sequence_idx:03d}/{frame_idx:08d}.png')
            for idx in range(num_half_frame, 0, -1):
                if frame_idx - idx >= 0:
                    neighbor_idx.append(frame_idx - idx)
                    neighbor_names.append(f'p{idx}')
            for idx in range(1, num_half_frame + 1):
                if frame_idx + idx <= 99:
                    neighbor_idx.append(frame_idx + idx)
                    neighbor_names.append(f'n{idx}')

            center_frame = read_image(center_frame_path)
            for idx, name in zip(neighbor_idx, neighbor_names):
                neighbor_frame_path = os.path.join(data_root, input_folder, f'{sequence_idx:03d}/{idx:08d}.png')
                output_path = os.path.join(data_root, output_folder, f'{sequence_idx:03d}/{frame_idx:08d}_{name}')
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

        #     print(center_frame_path)
        #     print(neighbor_idx)
        #     print(neighbor_names)
        # embed()
        # exit()
