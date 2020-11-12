# -*- coding: utf-8 -*-
"""
SRNet data generator.
Copyright (c) 2019 Netease Youdao Information Technology Co.,Ltd.
Licensed under the GPL License 
Written by Yu Qian
"""

import os
import cv2
import math
import glob
import numpy as np
import pygame
from pygame import freetype
import random
import multiprocessing
import queue
import Augmentor

from . import render_text_mask
from . import colorize
from . import skeletonization
from . import render_standard_text
from . import data_cfg
import pickle as cp

class datagen():
    def __init__(self):
        
        freetype.init()
        cur_file_path = os.path.dirname(__file__)
        
        font_dir = os.path.join(cur_file_path, data_cfg.font_dir)
        self.font_list = os.listdir(font_dir)
        self.font_list = [os.path.join(font_dir, font_name) for font_name in self.font_list]
        self.standard_font_path = os.path.join(cur_file_path, data_cfg.standard_font_path)
        
        text_filepath = os.path.join(cur_file_path, data_cfg.text_filepath)
        self.text_list = open(text_filepath, 'r').readlines()
        self.text_list = [text.strip() for text in self.text_list]
        
        network_input_filepath = os.path.join(cur_file_path,data_cfg.input_text_source_filepath)
        
        if os.path.exists(network_input_filepath):
            self.source_image_paths = [os.path.join(network_input_filepath,path) for path in os.listdir(network_input_filepath)]
            print(self.source_image_paths)

        
        self.surf_augmentor = Augmentor.DataPipeline(None)
        self.surf_augmentor.random_distortion(probability = data_cfg.elastic_rate,
            grid_width = data_cfg.elastic_grid_size, grid_height = data_cfg.elastic_grid_size,
            magnitude = data_cfg.elastic_magnitude)
        

    def gen_srnet_data_with_background(self):
        while True:
            # choose font, text and bg
            font = np.random.choice(self.font_list)
            text1, text2 = np.random.choice(self.text_list), np.random.choice(self.text_list)
            
            upper_rand = np.random.rand()

            if upper_rand < data_cfg.capitalize_rate + data_cfg.uppercase_rate:
                text1, text2 = text1.capitalize(), text2.capitalize()
            if upper_rand < data_cfg.uppercase_rate:
                text1, text2 = text1.upper(), text2.upper()

            # init font
            font = freetype.Font(font)
            font.antialiased = True
            font.origin = True

            # choose font style
            font.size = np.random.randint(data_cfg.font_size[0], data_cfg.font_size[1] + 1)
            font.underline = np.random.rand() < data_cfg.underline_rate
            font.strong = np.random.rand() < data_cfg.strong_rate
            font.oblique = np.random.rand() < data_cfg.oblique_rate

            # render text to surf
            param = {
                        'is_curve': np.random.rand() < data_cfg.is_curve_rate,
                        'curve_rate': data_cfg.curve_rate_param[0] * np.random.randn() 
                                      + data_cfg.curve_rate_param[1],
                        'curve_center': np.random.randint(0, len(text1))
                    }
            surf1, bbs1 = render_text_mask.render_text(font, text1, param)
            param['curve_center'] = int(param['curve_center'] / len(text1) * len(text2))
            surf2, bbs2 = render_text_mask.render_text(font, text2, param)

            # get padding
            padding_ud = np.random.randint(data_cfg.padding_ud[0], data_cfg.padding_ud[1] + 1, 2)
            padding_lr = np.random.randint(data_cfg.padding_lr[0], data_cfg.padding_lr[1] + 1, 2)
            padding = np.hstack((padding_ud, padding_lr))

            # perspect the surf
            rotate = data_cfg.rotate_param[0] * np.random.randn() + data_cfg.rotate_param[1]
            zoom = data_cfg.zoom_param[0] * np.random.randn(2) + data_cfg.zoom_param[1]
            shear = data_cfg.shear_param[0] * np.random.randn(2) + data_cfg.shear_param[1]
            perspect = data_cfg.perspect_param[0] * np.random.randn(2) +data_cfg.perspect_param[1]
            surf1 = render_text_mask.perspective(surf1, rotate, zoom, shear, perspect, padding) # w first
            surf2 = render_text_mask.perspective(surf2, rotate, zoom, shear, perspect, padding) # w first

            # choose a background
            surf1_h, surf1_w = surf1.shape[:2]
            surf2_h, surf2_w = surf2.shape[:2]
            surf_h = max(surf1_h, surf2_h)
            surf_w = max(surf1_w, surf2_w)
            surf1 = render_text_mask.center2size(surf1, (surf_h, surf_w))
            surf2 = render_text_mask.center2size(surf2, (surf_h, surf_w))
            
            # augment surf
            surfs = [[surf1, surf2]]
            self.surf_augmentor.augmentor_images = surfs
            surf1, surf2 = self.surf_augmentor.sample(1)[0]

            # render standard text
            text2 ="sample"
            print("surf_h:{}".format(surf_h))
            print("surf_w:{}".format(surf_w))
            
            i_t = render_standard_text.make_standard_text(self.standard_font_path, text2, (surf_h, surf_w))
            break
   
        return i_t

    def run_generate(self,output_dir):
        text = "MOMO"
        for idx, img_path in enumerate(self.source_image_paths):
            print ("Generating step {:>6d} / {:>6d}".format(idx + 1, len(self.source_image_paths)))
            im = cv2.imread(img_path)
            surf_h, surf_w = im.shape[0],im.shape[1]
            
            i_t = render_standard_text.make_standard_text(self.standard_font_path,text,(surf_h,surf_w))

            basename = os.path.splitext(os.path.basename(img_path))[0]
            basename = basename.split('_')[0]

            i_t_path = os.path.join(output_dir, basename + '_i_t.png')
            i_s_path = os.path.join(output_dir, basename + '_i_s.png')
            cv2.imwrite(i_t_path, i_t, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
            cv2.imwrite(i_s_path, im, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])


def enqueue_data(queue, capacity):  
    
    np.random.seed()
    gen = datagen()
    while True:
        try:
            data = gen.gen_srnet_data_with_background()
        except Exception as e:
            pass
        if queue.qsize() < capacity:
            queue.put(data)

class multiprocess_datagen():
    
    def __init__(self, process_num, data_capacity):
        
        self.process_num = process_num
        self.data_capacity = data_capacity
            
    def multiprocess_runningqueue(self):
        
        manager = multiprocessing.Manager()
        self.queue = manager.Queue()
        self.pool = multiprocessing.Pool(processes = self.process_num)
        self.processes = []
        for _ in range(self.process_num):
            p = self.pool.apply_async(enqueue_data, args = (self.queue, self.data_capacity))
            self.processes.append(p)
        self.pool.close()
        
    def dequeue_data(self):
        
        while self.queue.empty():
            pass
        data = self.queue.get()
        return data
        '''
        data = None
        if not self.queue.empty():
            data = self.queue.get()
        return data
        '''

    def dequeue_batch(self, batch_size, data_shape):
        
        while self.queue.qsize() < batch_size:
            pass

        i_t_batch = []
        for i in range(batch_size):
            i_t = self.dequeue_data()
            i_t_batch.append(i_t)
            
        w_sum = 0
        for t_b in t_b_batch:
            h, w = t_b.shape[:2]
            scale_ratio = data_shape[0] / h
            w_sum += int(w * scale_ratio)
        
        to_h = data_shape[0]
        to_w = w_sum // batch_size
        to_w = int(round(to_w / 8)) * 8
        to_size = (to_w, to_h) # w first for cv2
        for i in range(batch_size): 
            i_t_batch[i] = cv2.resize(i_t_batch[i], to_size)
            i_s_batch[i] = cv2.resize(i_s_batch[i], to_size)
            t_sk_batch[i] = cv2.resize(t_sk_batch[i], to_size, interpolation=cv2.INTER_NEAREST)
            t_t_batch[i] = cv2.resize(t_t_batch[i], to_size)
            t_b_batch[i] = cv2.resize(t_b_batch[i], to_size)
            t_f_batch[i] = cv2.resize(t_f_batch[i], to_size)
            mask_t_batch[i] = cv2.resize(mask_t_batch[i], to_size, interpolation=cv2.INTER_NEAREST)
            # eliminate the effect of resize on t_sk
            t_sk_batch[i] = skeletonization.skeletonization(mask_t_batch[i], 127)

        i_t_batch = np.stack(i_t_batch)
        
        i_t_batch = i_t_batch.astype(np.float32) / 127.5 - 1. 
        
        return [i_t_batch]
    
    def get_queue_size(self):
        
        return self.queue.qsize()
    
    def terminate_pool(self):
        
        self.pool.terminate()
