"""Train pi-GAN. Supports distributed training."""

import argparse
import os
import numpy as np
import math

from collections import deque

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.utils import save_image

from generators import generators
from discriminators import discriminators
from renderer import Renderer
from siren import siren
import fid_evaluation
import extract_shapes
import mrcfile

import datasets
import curriculums
from tqdm import tqdm
from datetime import datetime
import copy
from glob import glob

from torch_ema import ExponentialMovingAverage

def cleanup():
    dist.destroy_process_group()

def load_images(images, curriculum, device):
    return_images = []
    head = 0
    for stage in curriculum['stages']:
        stage_images = images[head:head + stage['batch_size']]
        stage_images = F.interpolate(stage_images, size=stage['img_size'],  mode='bilinear', align_corners=True)
        return_images.append(stage_images)
        head += stage['batch_size']
    return return_images


def z_sampler(shape, device, dist):
    if dist == 'gaussian':
        z = torch.randn(shape, device=device)
    elif dist == 'uniform':
        z = torch.rand(shape, device=device) * 2 - 1
    return z

def denorm(img, is_depth=False, metadata=None):
    if is_depth:
        img = ((img - metadata['ray_start']) / (metadata['ray_end'] - metadata['ray_start'])).clamp(0,1)
        return img
    else:
        img = (img /2 + 0.5).clamp(0,1)
        return img

def save_output_image(results, output_dir, step, name, metadata):
    gen_imgs = results['final_pixels']
    albedo_pixel = results['albedo_pixels']
    albedo = results['albedo']
    depth = results['depth']
    pred_normal = results['pred_normal']
    specular = results['specular']
    shading = results['shading']
    gen_imgs_e_light = results['elight']
    pretrained_rbg = results['pretrained_rbg']
    light = results['light']
    specular_0 = results['specular_0']
    specular_1 = results['specular_1']

    save_image(denorm(gen_imgs[:25]),
               os.path.join(output_dir, f"{step}_{name}.png"), nrow=5)
    save_image(denorm(albedo_pixel[:25]),
               os.path.join(output_dir, f"{step}_albedopixel_{name}.png"), nrow=5)
    save_image(denorm(albedo[:25]),
               os.path.join(output_dir, f"{step}_albedo_{name}.png"), nrow=5)
    save_image(specular[:25],
               os.path.join(output_dir, f"{step}_specular_{name}.png"), nrow=5)
    save_image(denorm(depth.type(torch.float32)[:25], is_depth=True, metadata=metadata),
               os.path.join(output_dir, f"{step}_depth_{name}.png"), nrow=5)
    save_image(denorm(pred_normal.type(torch.float32)[:25]),
               os.path.join(output_dir, f"{step}_prednormal_{name}.png"), nrow=5)
    save_image(shading, os.path.join(output_dir, f"{step}_shading_{name}.png"),
               nrow=5, normalize=True)
    save_image(gen_imgs_e_light,
               os.path.join(output_dir, f"{step}_elight_{name}.png"), nrow=5,
               normalize=True)
    save_image(denorm(pretrained_rbg[:25]),
               os.path.join(output_dir, f"{step}_pretrainedrbg_{name}.png"), nrow=5)
    save_image(specular_0[:25],
               os.path.join(output_dir, f"{step}_specular_0_{name}.png"), nrow=5)
    save_image(specular_1[:25],
               os.path.join(output_dir, f"{step}_specular_1_{name}.png"), nrow=5, normalize=True)

def save_mesh(z, generator, output_dir, step):
    voxel_grid = extract_shapes.sample_generator(generator, z, cube_length=0.3, voxel_resolution=256)
    with mrcfile.new_mmap(os.path.join(output_dir, f'{step}.mrc'), overwrite=True, shape=voxel_grid.shape,
                          mrc_mode=2) as mrc:
        mrc.data[:] = voxel_grid




def eval(opt):

    device = 'cuda'

    curriculum = getattr(curriculums, opt.curriculum)
    metadata = curriculums.extract_metadata(curriculum, 0)
    opt.step_list = [30000, 40000, 50000, 'last']
    weight_list = []
    for weight_dir in glob(os.path.join(opt.load_dir, '*generator.pth')):
        weight = torch.load(weight_dir)
        if weight.step in opt.step_list:
            weight_list.append(weight_dir)
    if 'last' in opt.step_list:
        weight_list.append(os.path.join(opt.load_dir, 'generator.pth'))
    m1, s1 = None, None
    for load_dir in weight_list:
        if opt.load_dir != '':
            generator = torch.load(load_dir, map_location=device)
            # discriminator = torch.load(os.path.join(opt.load_dir, 'discriminator.pth'), map_location=device)
            ema = torch.load(load_dir.replace('generator', 'ema'))
            # ema2 = torch.load(os.path.join(opt.load_dir, 'ema2.pth'))

        else:
            raise NotImplementedError()

        metadata = curriculums.extract_metadata(curriculum, generator.step)
        metadata['nerf_noise'] = 0

        generated_dir = os.path.join(opt.output_dir, f'evaluation/generated/{generator.step}/ema/image')
        depth_dir = generated_dir.replace('image', 'depth')
        os.makedirs(depth_dir, exist_ok=True)
        if opt.nobg:
            os.makedirs(generated_dir.replace('image', 'image_nobg'), exist_ok=True)

        fid_evaluation.setup_evaluation(metadata['dataset'], metadata['dataset_path'], generated_dir, target_size=128,
                                        bg_remove=opt.nobg)

        ema.store(generator.parameters())
        ema.copy_to(generator.parameters())
        generator.eval()
        fid_evaluation.output_images_noddp(generator, metadata, generated_dir, depth_dir, bg_remove=opt.nobg, num_imgs=opt.num_img)
        ema.restore(generator.parameters())

        fid, fid_nobg, m1, s1 = fid_evaluation.calculate_fid(metadata['dataset'], generated_dir, target_size=128, bg_remove=opt.nobg, m1=m1, s1=s1)
        with open(os.path.join(opt.output_dir, f'fid_ema.txt'), 'a') as f:
            f.write(f'\n{generator.step}:{fid}')
        if opt.nobg:
            with open(os.path.join(opt.output_dir, f'fid_ema_nobg.txt'), 'a') as f:
                f.write(f'\n{generator.step}:{fid_nobg}')


        # generated_dir = os.path.join(opt.output_dir, 'evaluation/generated/ema_2/image')
        # depth_dir = generated_dir.replace('image', 'depth')
        # os.makedirs(depth_dir, exist_ok=True)
        # if opt.nobg:
        #     os.makedirs(generated_dir.replace('image', 'image_nobg'), exist_ok=True)
        #
        # fid_evaluation.setup_evaluation(metadata['dataset'], metadata['dataset_path'], generated_dir, target_size=128)

        # ema2.store(generator.parameters())
        # ema2.copy_to(generator.parameters())
        # generator.eval()
        # fid_evaluation.output_images_noddp(generator, metadata, generated_dir, depth_dir, bg_remove=opt.nobg)
        # ema2.restore(generator.parameters())
        #
        # fid, fid_nobg = fid_evaluation.calculate_fid(metadata['dataset'], generated_dir, target_size=128, bg_remove=opt.nobg)
        # with open(os.path.join(opt.output_dir, f'fid_ema2.txt'), 'a') as f:
        #     f.write(f'\n{discriminator.step}:{fid}')
        # if opt.nobg:
        #     with open(os.path.join(opt.output_dir, f'fid_ema2_nobg.txt'), 'a') as f:
        #         f.write(f'\n{discriminator.step}:{fid_nobg}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='eval')
    parser.add_argument('--load_dir', type=str, default='')
    parser.add_argument('--curriculum', type=str, required=True)
    parser.add_argument('--gpu_ids', type=str, default='5')
    parser.add_argument('--num_img', type=int, default=20)
    parser.add_argument('--nobg', type=bool, default=False)


    opt = parser.parse_args()
    print(opt)
    os.makedirs(opt.output_dir, exist_ok=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_ids
    num_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    if num_gpus > 1:
        raise NotImplementedError()
    eval(opt)
