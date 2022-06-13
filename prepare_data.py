#!/usr/bin/env python3
"""
script for metaseg input preparation
"""

import os
import numpy as np
from PIL import Image

from global_defs import CONFIG


class Cityscapes():  

    def __init__(self, **kwargs):
        """
        Dataset loader that processes all images from one specified root directory
        Also searches for images in every subdirectory in root directory
        """
        
        self.images = []    # where to load input images - absolute paths
        self.targets = []   # where to load ground truth if available - absolute paths
        self.depth = []     # where to load depth maps - absolute paths
        self.sem_seg = []   # where to load semantic segmentation predictions - absolute paths
        self.depth_seg = [] # where to load depth segmentation predictions - absolute paths
        self.name = []      # image name

        for city in sorted(os.listdir(CONFIG.IMG_DIR)):
            for img in sorted(os.listdir(os.path.join(CONFIG.IMG_DIR,city))):

                self.images.append(os.path.join(CONFIG.IMG_DIR, city, img)) 
                self.targets.append(os.path.join(CONFIG.GT_DIR, city, img.replace('leftImg8bit','gtFine_labelTrainIds'))) 
                self.depth.append(os.path.join(CONFIG.DEPTH_DIR, CONFIG.DEPTH_MODEL_NAME, CONFIG.DATASET, 'val', city, img.replace('leftImg8bit','leftImg8bit'))) 
                self.sem_seg.append(os.path.join(CONFIG.PRED_SEM_SEG_DIR, CONFIG.SEG_MODEL_NAME, CONFIG.DATASET, city, img.replace('leftImg8bit.png','labelTrainIds.npy'))) 
                self.depth_seg.append(os.path.join(CONFIG.PRED_DEPTH_SEG_DIR, CONFIG.DEPTH_MODEL_NAME, CONFIG.DATASET, img.replace('.png','.npy'))) 
                self.name.append(img.split('_left')[0])

    def __getitem__(self, index):
        """Generate one sample of data"""
        image = np.asarray(Image.open(self.images[index]).convert('RGB'))
        target = np.asarray(Image.open(self.targets[index])) 
        depth = np.asarray(Image.open(self.depth[index])) 
        sem_seg = np.load(self.sem_seg[index])
        depth_seg = np.load(self.depth_seg[index])  
        return image, target, depth, sem_seg, depth_seg, self.name[index]

    def __len__(self):
        """Denote the total number of samples"""
        return len(self.images)


class Lost_and_found():  

    def __init__(self, **kwargs):
        """
        Dataset loader that processes all images from one specified root directory
        Also searches for images in every subdirectory in root directory
        """

        self.images = []    # where to load input images - absolute paths
        self.targets = []   # where to load ground truth if available - absolute paths
        self.depth = []     # where to load depth maps - absolute paths
        self.sem_seg = []   # where to load semantic segmentation predictions - absolute paths
        self.depth_seg = [] # where to load depth segmentation predictions - absolute paths
        self.name = []      # image name

        for city in sorted(os.listdir(CONFIG.IMG_DIR)):
            for img in sorted(os.listdir(os.path.join(CONFIG.IMG_DIR,city))):

                self.images.append(os.path.join(CONFIG.IMG_DIR, city, img)) 
                self.targets.append(os.path.join(CONFIG.GT_DIR, city, img.replace('leftImg8bit','gtCoarse_labelTrainIds'))) 
                self.depth.append(os.path.join(CONFIG.DEPTH_DIR, CONFIG.DEPTH_MODEL_NAME, CONFIG.DATASET, 'val', city, img.replace('leftImg8bit','leftImg8bit'))) 
                self.sem_seg.append(os.path.join(CONFIG.PRED_SEM_SEG_DIR, CONFIG.SEG_MODEL_NAME, CONFIG.DATASET, city, img.replace('leftImg8bit.png','labelTrainIds.npy'))) 
                self.depth_seg.append(os.path.join(CONFIG.PRED_DEPTH_SEG_DIR, CONFIG.DEPTH_MODEL_NAME, CONFIG.DATASET, img.replace('.png','.npy'))) 
                self.name.append(img.split('_left')[0])

    def __getitem__(self, index):
        """Generate one sample of data"""
        image = np.asarray(Image.open(self.images[index]).convert('RGB'))
        target = np.asarray(Image.open(self.targets[index])) 
        target[target==1] = 0
        target[target==2] = 254  
        depth = np.asarray(Image.open(self.depth[index])) 
        sem_seg = np.load(self.sem_seg[index])
        depth_seg = np.load(self.depth_seg[index])  
        return image, target, depth, sem_seg, depth_seg, self.name[index]

    def __len__(self):
        """Denote the total number of samples"""
        return len(self.images)


class A2d2():  

    def __init__(self, **kwargs):
        """
        Dataset loader that processes all images from one specified root directory
        Also searches for images in every subdirectory in root directory
        """
        
        self.images = []    # where to load input images - absolute paths
        self.targets = []   # where to load ground truth if available - absolute paths
        self.depth = []     # where to load depth maps - absolute paths
        self.sem_seg = []   # where to load semantic segmentation predictions - absolute paths
        self.depth_seg = [] # where to load depth segmentation predictions - absolute paths
        self.name = []      # image name

        for city in sorted(os.listdir(CONFIG.IMG_DIR)):
            for img in sorted(os.listdir(os.path.join(CONFIG.IMG_DIR,city))):

                self.images.append(os.path.join(CONFIG.IMG_DIR, city, img)) 
                self.targets.append(os.path.join(CONFIG.GT_DIR, city, img.replace('leftImg8bit','gtFine_labelTrainIds'))) 
                self.depth.append(os.path.join(CONFIG.DEPTH_DIR, CONFIG.DEPTH_MODEL_NAME, CONFIG.DATASET, 'val', city, img.replace('leftImg8bit','leftImg8bit'))) 
                self.sem_seg.append(os.path.join(CONFIG.PRED_SEM_SEG_DIR, CONFIG.SEG_MODEL_NAME, CONFIG.DATASET, city, img.replace('leftImg8bit.png','labelTrainIds.npy'))) 
                self.depth_seg.append(os.path.join(CONFIG.PRED_DEPTH_SEG_DIR, CONFIG.DEPTH_MODEL_NAME, CONFIG.DATASET, img.replace('.png','.npy'))) 
                self.name.append(img.split('_left')[0])

    def __getitem__(self, index):
        """Generate one sample of data"""
        image = np.asarray(Image.open(self.images[index]).convert('RGB'))
        target = np.asarray(Image.open(self.targets[index])) 
        target[target==254] = 255
        depth = np.asarray(Image.open(self.depth[index])) 
        sem_seg = np.load(self.sem_seg[index])
        depth_seg = np.load(self.depth_seg[index])  
        return image, target, depth, sem_seg, depth_seg, self.name[index]

    def __len__(self):
        """Denote the total number of samples"""
        return len(self.images)


class Idd():  

    def __init__(self, **kwargs):
        """
        Dataset loader that processes all images from one specified root directory
        Also searches for images in every subdirectory in root directory
        """
        
        self.images = []    # where to load input images - absolute paths
        self.targets = []   # where to load ground truth if available - absolute paths
        self.depth = []     # where to load depth maps - absolute paths
        self.sem_seg = []   # where to load semantic segmentation predictions - absolute paths
        self.depth_seg = [] # where to load depth segmentation predictions - absolute paths
        self.name = []      # image name

        for city in sorted(os.listdir(CONFIG.IMG_DIR)):
            for img in sorted(os.listdir(os.path.join(CONFIG.IMG_DIR,city))):

                self.images.append(os.path.join(CONFIG.IMG_DIR, city, img)) 
                self.targets.append(os.path.join(CONFIG.GT_DIR, city, img.replace('leftImg8bit','gtFine_labelTrainIds'))) 
                self.depth.append(os.path.join(CONFIG.DEPTH_DIR, CONFIG.DEPTH_MODEL_NAME, CONFIG.DATASET, 'val', city, img.replace('leftImg8bit','leftImg8bit'))) 
                self.sem_seg.append(os.path.join(CONFIG.PRED_SEM_SEG_DIR, CONFIG.SEG_MODEL_NAME, CONFIG.DATASET, city, img.replace('leftImg8bit.png','labelTrainIds.npy'))) 
                self.depth_seg.append(os.path.join(CONFIG.PRED_DEPTH_SEG_DIR, CONFIG.DEPTH_MODEL_NAME, CONFIG.DATASET, img.replace('.png','.npy'))) 
                self.name.append(img.split('_left')[0])

    def __getitem__(self, index):
        """Generate one sample of data"""
        image = np.asarray(Image.open(self.images[index]).convert('RGB'))
        target = np.asarray(Image.open(self.targets[index]))
        target[target==9] = 11  # animal -> person
        target[target==16] = 13 # autorickshaw -> car
        depth = np.asarray(Image.open(self.depth[index])) 
        sem_seg = np.load(self.sem_seg[index])
        depth_seg = np.load(self.depth_seg[index])  
        return image, target, depth, sem_seg, depth_seg, self.name[index]

    def __len__(self):
        """Denote the total number of samples"""
        return len(self.images)


