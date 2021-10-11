import argparse
import math
import glob
import numpy as np
import sys
import os
import torchvision

import torch
from torchvision.utils import save_image
from tqdm import tqdm
from renderer import Renderer
import curriculums
import copy


def show(tensor_img):
    if len(tensor_img.shape) > 3:
        tensor_img = tensor_img.squeeze(0)
    tensor_img = tensor_img.permute(1, 2, 0).squeeze().cpu().numpy()
    plt.imshow(tensor_img)
    plt.show()


def generate_img(gen, z, **kwargs):
    with torch.no_grad():
        img, depth_map, position = generator.forward(z, render_depth=True, render_cannon=True, **kwargs)
        tensor_img = img.detach()

        img_min = img.min()
        img_max = img.max()
        img = (img + 1) / 2
        img = img.permute(0, 2, 3, 1).squeeze().cpu().numpy()
    return img, tensor_img, depth_map, position


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('--seeds', nargs='+', default=[0, 1, 2])
    parser.add_argument('--output_dir', type=str, default='imgs')
    parser.add_argument('--max_batch_size', type=int, default=2400000)
    parser.add_argument('--lock_view_dependence', action='store_true')
    parser.add_argument('--image_size', type=int, default=128)
    parser.add_argument('--ray_step_multiplier', type=int, default=2)
    parser.add_argument('--curriculum', type=str, default='CelebA')
    parser.add_argument('--gpu_ids', type=str, default='1')
    opt = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_ids
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    curriculum = getattr(curriculums, opt.curriculum)
    curriculum['num_steps'] = curriculum[0]['num_steps'] * opt.ray_step_multiplier
    curriculum['img_size'] = opt.image_size
    curriculum['psi'] = 0.7
    curriculum['lock_view_dependence'] = opt.lock_view_dependence
    curriculum['last_back'] = curriculum.get('eval_last_back', False)
    curriculum['nerf_noise'] = 0
    curriculum['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    curriculum = {key: value for key, value in curriculum.items() if type(key) is str}

    renderer = Renderer(curriculum)

    os.makedirs(os.path.join(opt.output_dir, 'concat'), exist_ok=True)
    os.makedirs(os.path.join(opt.output_dir, 'depth'), exist_ok=True)
    os.makedirs(os.path.join(opt.output_dir, 'depth_ours'), exist_ok=True)
    os.makedirs(os.path.join(opt.output_dir, 'warping_images'), exist_ok=True)
    os.makedirs(os.path.join(opt.output_dir, 'warping_images_ours'), exist_ok=True)

    # os.makedirs(os.path.join(opt.output_dir, 'warping_unsup_maxr90_depth'), exist_ok=True)
    # os.makedirs(os.path.join(opt.output_dir, 'warping_pigan_pidiv2'), exist_ok=True)
    # os.makedirs(os.path.join(opt.output_dir, 'warping_pigan_pidiv2_depth'), exist_ok=True)


    generator = torch.load(opt.path, map_location=torch.device(device))
    ema_file = opt.path.split('generator')[0] + 'ema.pth'
    ema = torch.load(ema_file)
    ema.copy_to(generator.parameters())
    generator.set_device(device)
    generator.eval()

    h_mean = copy.copy(curriculum['h_mean'])

    b = 2

    for seed in tqdm(opt.seeds):
        torch.manual_seed(seed)
        z = torch.randn((b, 256), device=device)

        curriculum['h_mean'] = 0 + h_mean
        with torch.no_grad():
            img, depth_map, depth_ours, position = generator.forward(z, render_depth=True, render_cannon=True, **curriculum)
        warping_pos = -(position[:b] - position[b:])
        warped_images, warped_depth = renderer.render_h_v(img[b:], depth_map[b:], warping_pos)
        warped_images_ours, warped_depth_ours = renderer.render_h_v(img[b:], depth_ours[b:], warping_pos)

        save_image((img[:b] + 1) / 2,
                   os.path.join(opt.output_dir, 'concat', f'grid_{seed}.png'))
        save_image((depth_map.unsqueeze(1) - 0.88) / (1.12 - 0.88),
                   os.path.join(opt.output_dir, 'depth', f'grid_{seed}.png'))
        save_image((depth_ours.unsqueeze(1) - 0.88) / (1.12 - 0.88),
                   os.path.join(opt.output_dir, 'depth_ours', f'grid_{seed}.png'))
        save_image((warped_images + 1) / 2,
                   os.path.join(opt.output_dir, 'warping_images', f'grid_{seed}.png'))
        save_image((warped_images_ours + 1) / 2,
                   os.path.join(opt.output_dir, 'warping_images_ours', f'grid_{seed}.png'))

        # canon_im_rotate = canon_im_rotate.clamp(-1, 1).detach().cpu() / 2 + 0.5
        # canon_depth_rotate = ((canon_depth_rotate - 0.88) / (1.12 - 0.88)).clamp(0,1)
        # save_image(torchvision.utils.make_grid(canon_im_rotate[0], nrow=7),
        #            os.path.join(opt.output_dir, 'warping_unsup_maxr90', f'grid_{seed}.png'))
        # save_image(torchvision.utils.make_grid(canon_depth_rotate[0], nrow=7),
        #            os.path.join(opt.output_dir, 'warping_unsup_maxr90_depth', f'grid_{seed}.png'))

