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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def show(tensor_img):
    if len(tensor_img.shape) > 3:
        tensor_img = tensor_img.squeeze(0)
    tensor_img = tensor_img.permute(1, 2, 0).squeeze().cpu().numpy()
    plt.imshow(tensor_img)
    plt.show()


def generate_img(gen, z, **kwargs):
    with torch.no_grad():
        img, depth_map = generator.staged_forward(z, **kwargs)
        tensor_img = img.detach()

        img_min = img.min()
        img_max = img.max()
        img = (img - img_min) / (img_max - img_min)
        img = img.permute(0, 2, 3, 1).squeeze().cpu().numpy()
    return img, tensor_img, depth_map, depth_ours


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('--seeds', nargs='+', default=[0, 1, 2])
    parser.add_argument('--output_dir', type=str, default='test_results')
    parser.add_argument('--max_batch_size', type=int, default=2400000)
    parser.add_argument('--lock_view_dependence', action='store_true')
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--ray_step_multiplier', type=int, default=2)
    parser.add_argument('--curriculum', type=str, default='CelebA')
    opt = parser.parse_args()

    curriculum = getattr(curriculums, opt.curriculum)
    curriculum['num_steps'] = curriculum[0]['num_steps'] * opt.ray_step_multiplier
    curriculum['img_size'] = opt.image_size
    curriculum['psi'] = 0.7
    curriculum['v_stddev'] = 0
    curriculum['h_stddev'] = 0
    curriculum['lock_view_dependence'] = opt.lock_view_dependence
    curriculum['last_back'] = curriculum.get('eval_last_back', False)
    curriculum['nerf_noise'] = 0
    curriculum['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    curriculum = {key: value for key, value in curriculum.items() if type(key) is str}

    renderer = Renderer(curriculum)

    os.makedirs(os.path.join(opt.output_dir, 'warping_unsup'), exist_ok=True)
    os.makedirs(os.path.join(opt.output_dir, 'warping_unsup_depth'), exist_ok=True)
    os.makedirs(os.path.join(opt.output_dir, 'warping_unsup_ours'), exist_ok=True)
    os.makedirs(os.path.join(opt.output_dir, 'warping_unsup_ours_depth'), exist_ok=True)
    os.makedirs(os.path.join(opt.output_dir, 'warping_pigan'), exist_ok=True)
    os.makedirs(os.path.join(opt.output_dir, 'warping_pigan_depth'), exist_ok=True)
    os.makedirs(os.path.join(opt.output_dir, 'warping_pigan_ours_depth'), exist_ok=True)



    generator = torch.load(opt.path, map_location=torch.device(device))
    ema_file = opt.path.split('generator')[0] + 'ema.pth'
    ema = torch.load(ema_file)
    ema.copy_to(generator.parameters())
    generator.set_device(device)
    generator.eval()

    #face_angles = [-0.5, -0.25, 0., 0.25, 0.5]
    #face_angles = torch.linspace(1,-1,7)
    face_angles = torch.linspace(math.pi / 2,-math.pi /2,7)

    h_mean = copy.copy(curriculum['h_mean'])
    face_angles = [a + curriculum['h_mean'] for a in face_angles]

    b = 1

    for seed in tqdm(opt.seeds):
        torch.manual_seed(seed)
        z = torch.randn((1, 256), device=device)

        curriculum['h_mean'] = 0 + h_mean
        img, tensor_img, depth_map, depth_ours = generate_img(generator, z, **curriculum)
        v0 = torch.FloatTensor([0 * math.pi / 180 * 60, 0, 0, 0, 0, 0]).to(device).repeat(b, 1)
        canon_im_rotate, canon_depth_rotate = renderer.render_yaw(tensor_img.to(device), depth_map.to(device), v_before=v0, maxr=90,
                                              nsample=7)  # (B,T,C,H,W)
        canon_im_rotate = canon_im_rotate.clamp(-1, 1).detach().cpu() / 2 + 0.5
        canon_depth_rotate = ((canon_depth_rotate - 0.88) / (1.12 - 0.88)).clamp(0,1)
        save_image(torchvision.utils.make_grid(canon_im_rotate[0], nrow=7),
                   os.path.join(opt.output_dir, 'warping_unsup', f'grid_{seed}.png'))
        save_image(torchvision.utils.make_grid(canon_depth_rotate[0], nrow=7),
                   os.path.join(opt.output_dir, 'warping_unsup_depth', f'grid_{seed}.png'))


        canon_im_rotate_ours, canon_depth_rotate_ours = renderer.render_yaw(tensor_img.to(device), depth_ours.to(device),
                                                                  v_before=v0, maxr=90,
                                                                  nsample=7)  # (B,T,C,H,W)
        canon_im_rotate_ours = canon_im_rotate_ours.clamp(-1, 1).detach().cpu() / 2 + 0.5
        canon_depth_rotate_ours = ((canon_depth_rotate_ours - 0.88) / (1.12 - 0.88)).clamp(0, 1)
        save_image(torchvision.utils.make_grid(canon_im_rotate_ours[0], nrow=7),
                   os.path.join(opt.output_dir, 'warping_unsup_ours', f'grid_{seed}.png'))
        save_image(torchvision.utils.make_grid(canon_depth_rotate_ours[0], nrow=7),
                   os.path.join(opt.output_dir, 'warping_unsup_ours_depth', f'grid_{seed}.png'))

        images = []
        depths = []
        depths_ours = []
        for i, yaw in enumerate(face_angles):
            curriculum['h_mean'] = yaw
            torch.manual_seed(seed)
            z = torch.randn((1, 256), device=device)
            img, tensor_img, depth_map, depth_ours = generate_img(generator, z, **curriculum)
            images.append((tensor_img /2 + 0.5).clamp(0,1))
            depths.append(((depth_map - 0.88)/(1.12 - 0.88)).clamp(0,1).unsqueeze(1))
            depths_ours.append(((depth_ours - 0.88) / (1.12 - 0.88)).clamp(0, 1).unsqueeze(1))
        save_image(torch.cat(images), os.path.join(opt.output_dir, 'warping_pigan', f'grid_{seed}.png'), normalize=False)
        save_image(torch.cat(depths), os.path.join(opt.output_dir, 'warping_pigan_depth', f'grid_{seed}.png'), normalize=False)
        save_image(torch.cat(depths_ours), os.path.join(opt.output_dir, 'warping_pigan_ours_depth', f'grid_{seed}.png'), normalize=False)

