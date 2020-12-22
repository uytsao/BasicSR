import argparse
import cv2
import glob
import numpy as np
import os
import torch

from basicsr.models.archs.edsr_arch import EDSR


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_path',
        type=str,
        default=  # noqa: E251
        'experiments/pretrained_models/EDSR/EDSR_Lx4_f256b32_DIV2K_300k_B16G1.pth'  # noqa: E501
    )
    parser.add_argument(
        '--folder',
        type=str,
        default='datasets/dain_frames',
        help='input test image folder')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # set up model
    model = EDSR(
        num_in_ch=3, num_out_ch=3, num_feat=256, num_block=32, upscale=4, res_scale=0.1, img_range=255., rgb_mean=(0.4488, 0.4371, 0.4040))
    model.load_state_dict(torch.load(args.model_path)['params'], strict=True)
    model.eval()
    model = model.to(device)

    os.makedirs('results/edsr_frames', exist_ok=True)
    for idx, path in enumerate(
            sorted(glob.glob(os.path.join(args.folder, '*')))):
        imgname = os.path.splitext(os.path.basename(path))[0]
        print('Testing', idx, imgname)
        # read image
        img = cv2.imread(path, cv2.IMREAD_COLOR).astype(np.float32) / 255.
        img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]],
                                            (2, 0, 1))).float()
        img = img.unsqueeze(0).to(device)
        # inference
        with torch.no_grad():
            output = model(img)
        # save image
        output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
        output = (output * 255.0).round().astype(np.uint8)
        cv2.imwrite(f'results/edsr_frames/{imgname}.png', output)


if __name__ == '__main__':
    main()
