#!/usr/bin/python
# -*- encoding: utf-8 -*-

from .model import BiSeNet

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.distributed as dist
from torchvision.utils import save_image

import os
import os.path as osp
import argparse
import logging
import time
import numpy as np
from tqdm import tqdm
import math
from PIL import Image
import torchvision.transforms as transforms
import cv2

class BG_remover():
    def __init__(self, weight_pth='79999_iter.pth'):
        n_classes = 19
        self.net = BiSeNet(n_classes=n_classes)
        self.net.cuda()
        weight_pth = osp.join('./face_parsing/res/cp', weight_pth)
        self.net.load_state_dict(torch.load(weight_pth))
        self.net.eval()

    def vis_parsing_maps(self, im, parsing_anno, stride, save_im=False, save_path='vis_results/parsing_map_on_im.jpg'):
        # Colors for all 20 parts
        part_colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0],
                       [255, 0, 85], [255, 0, 170],
                       [0, 255, 0], [85, 255, 0], [170, 255, 0],
                       [0, 255, 85], [0, 255, 170],
                       [0, 0, 255], [85, 0, 255], [170, 0, 255],
                       [0, 85, 255], [0, 170, 255],
                       [255, 255, 0], [255, 255, 85], [255, 255, 170],
                       [255, 0, 255], [255, 85, 255], [255, 170, 255],
                       [0, 255, 255], [85, 255, 255], [170, 255, 255]]

        im = np.array(im)
        vis_im = im.copy().astype(np.uint8)
        vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
        vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
        vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255

        num_of_class = np.max(vis_parsing_anno)

        for pi in range(1, num_of_class + 1):
            index = np.where(vis_parsing_anno == pi)
            vis_parsing_anno_color[index[0], index[1], :] = part_colors[pi]

        vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
        # print(vis_parsing_anno_color.shape, vis_im.shape)
        vis_im = cv2.addWeighted(cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR), 0.4, vis_parsing_anno_color, 0.6, 0)

        # Save result or not
        if save_im:
            cv2.imwrite(save_path, vis_im, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

        # return vis_im

    def remove_bg_with_open(self, image_dir, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        to_tensor_nonorm = transforms.Compose([
            transforms.ToTensor()
        ])
        with torch.no_grad():
            for image_path in os.listdir(image_dir):
                img = Image.open(osp.join(image_dir, image_path))
                ori_img = to_tensor_nonorm(img).clone()
                c, w, h = ori_img.shape
                image = img.resize((512, 512), Image.BILINEAR)
                img = to_tensor(image)
                img = torch.unsqueeze(img, 0)
                img = img.cuda()
                out = self.net(img)[0]
                parsing = out.squeeze(0).cpu().numpy().argmax(0)
                bg = torch.Tensor((parsing == 0).astype('int8')).unsqueeze(0)
                bg = F.interpolate(bg.unsqueeze(0),(w,h))[0]
                fore_ground = ori_img * (1 - bg)
                save_image(fore_ground, os.path.join(save_dir, image_path))

    def remove_bg(self, image):
        normalize = transforms.Compose([
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        to_tensor_nonorm = transforms.Compose([
            transforms.ToTensor()
        ])
        with torch.no_grad():
            ori_img = image.clone()
            c, w, h = ori_img.shape
            image = normalize(image)
            img = F.interpolate(image.unsqueeze(0), (512,512), mode='bilinear')
            img = img.cuda()
            out = self.net(img)[0]
            parsing = out.squeeze(0).cpu().numpy().argmax(0)
            bg = torch.Tensor((parsing == 0).astype('int8')).unsqueeze(0)
            bg = F.interpolate(bg.unsqueeze(0),(w,h))[0]
            fore_ground = ori_img * (1 - bg)
            return fore_ground

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, default='/home/nas1_temp/minsoolee/3Dmining/pigan-feature-field/eval/pretrained_recon_spec_freeze/20_10_0_1_0_0_nofr_light2/30000/evaluation/generated/test')
    parser.add_argument('--save_dir', type=str, default='./res/test_res')
    opt = parser.parse_args()
    remover = BG_remover()
    remover.remove_bg_with_open(opt.image_dir, opt.save_dir)
