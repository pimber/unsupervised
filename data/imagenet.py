"""
Author: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import os
import torch
import torchvision.datasets as datasets
import torch.utils.data as data
from PIL import Image
from utils.mypath import MyPath
from torchvision import transforms as tf
from glob import glob


class ImageNet(datasets.ImageFolder):
    def __init__(self, root=MyPath.db_root_dir('imagenet'), split='train', transform=None):
        super(ImageNet, self).__init__(root=os.path.join(root, 'ILSVRC2012_img_%s' %(split)),
                                         transform=None)
        self.transform = transform 
        self.split = split
        self.resize = tf.Resize(256)
    
    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        path, target = self.imgs[index]
        with open(path, 'rb') as f:
            img = Image.open(f).convert('RGB')
        im_size = img.size
        img = self.resize(img)

        if self.transform is not None:
            img = self.transform(img)

        out = {'image': img, 'target': target, 'meta': {'im_size': im_size, 'index': index}}

        return out

    def get_image(self, index):
        path, target = self.imgs[index]
        with open(path, 'rb') as f:
            img = Image.open(f).convert('RGB')
        img = self.resize(img) 
        return img


class ImageNetSubset(data.Dataset):
    def __init__(self, subset_file, root='E:\imagenet', split='train',
                    transform=None):
        super(ImageNetSubset, self).__init__()

        self.root = os.path.join(root, 'ILSVRC2012_img_%s' %(split))
        self.transform = transform
        self.split = split
        print(split)

        if split == "train":
            # Read the subset of classes to include (sorted)
            with open(subset_file, 'r') as f:
                result = f.read().splitlines()
            subdirs, class_names = [], []
            for line in result:
                subdir, class_name = line.split(' ', 1)
                subdirs.append(subdir)
                class_names.append(class_name)

            # Gather the files (sorted)
            imgs = []
            for i, subdir in enumerate(subdirs):
                subdir_path = os.path.join(self.root, subdir)
                files = sorted(glob(os.path.join(self.root, subdir, '*.JPEG')))
                for f in files:
                    imgs.append((f, i))
            self.imgs = imgs
            self.classes = class_names
        else:
            # Get class names
            class_dir, class_names = {}, []
            with open(subset_file, 'r') as f:
                result = f.read().splitlines()
            for line in result:
                subdir, class_name = line.split(' ', 1)
                class_dir[subdir] = class_name
                class_names.append(class_name)

            # Load validation classes from files
            imgs, img_labels = [], []
            with open('./data/imagenet_subsets/imagenet_validation_labels.txt') as f:
                img_labels = f.read().splitlines()
            files = sorted(glob(os.path.join(self.root, '*.JPEG')))
            for i, f in enumerate(files):
                label = img_labels[i]
                if label not in class_dir.keys():
                    continue
                class_name_index = class_names.index(class_dir[label])
                imgs.append((f, class_name_index))
            self.imgs = imgs
            self.classes = class_names
    
	# Resize
        self.resize = tf.Resize(256)

    def get_image(self, index):
        path, target = self.imgs[index]
        with open(path, 'rb') as f:
            img = Image.open(f).convert('RGB')
        img = self.resize(img) 
        return img

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        path, target = self.imgs[index]
        with open(path, 'rb') as f:
            img = Image.open(f).convert('RGB')
        im_size = img.size
        img = self.resize(img) 
        class_name = self.classes[target]

        if self.transform is not None:
            img = self.transform(img)

        out = {'image': img, 'target': target, 'meta': {'im_size': im_size, 'index': index, 'class_name': class_name}}

        return out
