"""
Contains code for logging approximate FID scores during training.
If you want to output ground-truth images from the training dataset, you can
run this file as a script.
"""

import os
import shutil
import torch
import copy
import argparse

from torchvision.utils import save_image
#from pytorch_fid import fid_score
import fid_score_2 as fid_score
from tqdm import tqdm

import datasets
import curriculums

try:
    from face_parsing.remove import BG_remover
except:
    pass

def output_real_images(dataloader, num_imgs, real_dir, bg_remove):
    img_counter = 0
    batch_size = dataloader.batch_size
    dataloader = iter(dataloader)
    if bg_remove:
        remover = BG_remover()
    for i in range(num_imgs//batch_size):
        real_imgs, _ = next(dataloader)

        for img in real_imgs:
            save_image(img, os.path.join(real_dir, f'{img_counter:0>5}.jpg'), normalize=True, range=(-1, 1))
            if bg_remove:
                img_nobg = remover.remove_bg(img / 2 + 0.5)
                save_image(img_nobg, os.path.join(real_dir.replace('_real_images_', '_real_images_nobg_'), f'{img_counter:0>5}.jpg'))
            img_counter += 1

def setup_evaluation(dataset_name, dataset_path, generated_dir, target_size=128, num_imgs=8000, bg_remove=False):
    # Only make real images if they haven't been made yet
    real_dir = os.path.join('EvalImages', dataset_name + '_real_images_' + str(target_size))
    if bg_remove:
        real_nobg_dir = real_dir.replace('_real_images_', '_real_images_nobg_')
    if not os.path.exists(real_dir):
        os.makedirs(real_dir)
        if bg_remove:
            os.makedirs(real_dir.replace('_real_images_', '_real_images_nobg_'), exist_ok=True)
        dataloader, CHANNELS = datasets.get_dataset(dataset_name, dataset_path=dataset_path, img_size=target_size)
        print('outputting real images...')
        output_real_images(dataloader, num_imgs, real_dir, bg_remove)
        print('...done')

    if generated_dir is not None:
        os.makedirs(generated_dir, exist_ok=True)
    return real_dir

def output_images(generator, input_metadata, rank, world_size, output_dir, num_imgs=2048):
    metadata = copy.deepcopy(input_metadata)
    metadata['img_size'] = 128
    metadata['batch_size'] = 4

    metadata['h_stddev'] = metadata.get('h_stddev_eval', metadata['h_stddev'])
    metadata['v_stddev'] = metadata.get('v_stddev_eval', metadata['v_stddev'])
    metadata['sample_dist'] = metadata.get('sample_dist_eval', metadata['sample_dist'])
    metadata['psi'] = 1

    img_counter = rank
    generator.eval()
    img_counter = rank

    if rank == 0: pbar = tqdm("generating images", total = num_imgs)
    with torch.no_grad():
        while img_counter < num_imgs:
            z = torch.randn((metadata['batch_size'], generator.module.z_dim), device=generator.module.device)
            generated_imgs, generated_depth = generator.module.staged_forward(z, use_fixed_light=False, **metadata)
            for img in generated_imgs:
                save_image(img, os.path.join(output_dir, f'{img_counter:0>5}.jpg'), normalize=True, range=(-1, 1))
                img_counter += world_size
                if rank == 0: pbar.update(world_size)
    if rank == 0: pbar.close()

def output_images_noddp(generator, input_metadata, output_dir, depth_dir, num_imgs=2048, bg_remove=False):
    if bg_remove:
        remover = BG_remover()
    metadata = copy.deepcopy(input_metadata)
    metadata['img_size'] = 128
    metadata['batch_size'] = 4

    metadata['h_stddev'] = metadata.get('h_stddev_eval', metadata['h_stddev'])
    metadata['v_stddev'] = metadata.get('v_stddev_eval', metadata['v_stddev'])
    metadata['sample_dist'] = metadata.get('sample_dist_eval', metadata['sample_dist'])
    metadata['psi'] = 1

    generator.eval()


    pbar = tqdm("generating images", total = num_imgs)
    with torch.no_grad():
        img_counter=0
        while img_counter < num_imgs:
            z = torch.randn((metadata['batch_size'], generator.z_dim), device=generator.device)
            generated_imgs, generated_depth = generator.staged_forward(z, use_fixed_light=False, **metadata)
            for i, img in enumerate(generated_imgs):
                save_image(img, os.path.join(output_dir, f'{img_counter:0>5}.jpg'), normalize=True, range=(-1, 1))
                save_image(generated_depth[i], os.path.join(depth_dir, f'{img_counter:0>5}.jpg'), normalize=True, range=(metadata['ray_start'], metadata['ray_end']))
                if bg_remove:
                    img_nobg = remover.remove_bg(img / 2 + 0.5)
                    save_image(img_nobg, os.path.join(output_dir.replace('image', 'image_nobg'), f'{img_counter:0>5}.jpg'))
                img_counter += 1
                pbar.update(1)
    pbar.close()

def calculate_fid(dataset_name, generated_dir, target_size=256, bg_remove=False, m1=None, s1=None):
    real_dir = os.path.join('EvalImages', dataset_name + '_real_images_' + str(target_size))
    fid, m1, s1 = fid_score.calculate_fid_given_paths([real_dir, generated_dir], 32, 'cuda', 2048, m1, s1)
    fid_nobg = None
    if bg_remove:
        fid_nobg = fid_score.calculate_fid_given_paths([real_dir.replace('_real_images_', '_real_images_nobg_'),
                                                        generated_dir.replace('image','image_nobg')], 32, 'cuda', 2048)

    torch.cuda.empty_cache()

    return fid, fid_nobg, m1, s1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='CelebA')
    parser.add_argument('--img_size', type=int, default=128)
    parser.add_argument('--num_imgs', type=int, default=8000)

    opt = parser.parse_args()

    real_images_dir = setup_evaluation(opt.dataset, None, target_size=opt.img_size, num_imgs=opt.num_imgs)