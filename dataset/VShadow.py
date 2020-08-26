import os
import os.path

import torch.utils.data as data
from PIL import Image
import random
import torch


# return video clips
class Video_shadow(data.Dataset):
    def __init__(self, root, joint_transform=None, img_transform=None, target_transform=None, temporal_dilation=None):
        self.root = root
        self.temporal_dilation = temporal_dilation
        if self.root[1] != 'video' or isinstance(root, list):
            raise TypeError('Please make sure correct input type')
        self.joint_transform = joint_transform
        self.img_transform = img_transform
        self.target_transform = target_transform
        self.window_size = 3
        self.window_stride = 1
        self.input_folder = 'images'
        self.label_folder = 'labels'
        self.img_ext = '.jpg'
        self.label_ext = '.png'
        self.num_video_frame = 0
        self.clip_market = []
        self.generateClipFromVideo()
        print('Total video frames is {}.'.format(self.num_video_frame))
        print('Total clip number is {}.'.format(len(self.clip_market)))

    def __getitem__(self, index):
        clip_img = []
        clip_target = []
        manual_random = random.random()
        for pair in self.clip_market[index]:
            img_path, gt_path = pair
            img = Image.open(img_path).convert('RGB')
            target = Image.open(gt_path).convert('L')
            if self.joint_transform is not None:
                img, target = self.joint_transform(img, target, manual_random)
            if self.img_transform is not None:
                img = self.img_transform(img)
            if self.target_transform is not None:
                target = self.target_transform(target)
            clip_img.append(img)
            clip_target.append(target)
        clip_img = torch.stack(clip_img, dim=0)  # (C, H, W) -> (T, C, H , W)
        clip_target = torch.stack(clip_target, dim=0)
        return clip_img, clip_target

    def generateClipFromVideo(self):
        video_list = os.listdir(os.path.join(self.root[0], self.input_folder))
        for video in video_list:
            img_list = [os.path.splitext(f)[0] for f in os.listdir(os.path.join(self.root[0], self.input_folder, video)) if f.endswith(self.img_ext)] # no ext
            img_list = self.sortImg(img_list)
            self.num_video_frame += len(img_list)
            # add sample like {3,4,5}... and {2,4,6}..., and {1,4,7}...
            if self.temporal_dilation is None:
                raise TypeError('please choose a proper group of training dilations')
            for rate in self.temporal_dilation:
                self.generateClipFromVideoWithDilation(video_name=video, img_list=img_list, dilation_rate=rate)

    def generateClipFromVideoWithDilation(self, video_name, img_list, dilation_rate=1):
        idx = 0
        while idx + self.window_size * dilation_rate < len(img_list):
            clip = []
            for j in range(0, self.window_size * dilation_rate, dilation_rate):
                pair = (os.path.join(self.root[0], self.input_folder, video_name, img_list[idx + j] + self.img_ext),
                        os.path.join(self.root[0], self.label_folder, video_name, img_list[idx + j] + self.label_ext))
                clip.append(pair)
            self.clip_market.append(clip)
            idx = idx + 1

    def sortImg(self, img_list):
        img_int_list = [int(f) for f in img_list]
        sort_index = [i for i, v in sorted(enumerate(img_int_list), key=lambda x: x[1])]  # sort img to 001,002,003...
        return [img_list[i] for i in sort_index]

    def __len__(self):
        return len(self.clip_market)


# return image pairs
class Image_shadow(data.Dataset):
    def __init__(self, root, joint_transform=None, img_transform=None, target_transform=None):
        # self.root = root
        self.joint_transform = joint_transform
        self.img_transform = img_transform
        self.target_transform = target_transform
        self.input_folder = 'images'
        self.label_folder = 'labels'
        self.img_ext = '.jpg'
        self.label_ext = '.png'
        self.imgs = self.combineMultiCenterData(root)
        print('Total image number is {}.'.format(len(self.imgs)))

    def __getitem__(self, index):
        img_path, gt_path = self.imgs[index]
        img = Image.open(img_path).convert('RGB')
        target = Image.open(gt_path).convert('L')
        if self.joint_transform is not None:
            img, target = self.joint_transform(img, target)
        if self.img_transform is not None:
            img = self.img_transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def combineMultiCenterData(self, root):
        # root can be str or list(one dataset or more)
        if isinstance(root, tuple):
            imgs = self.generateImagePair(root[0])
            print('Image number of ImageSet {} is {}.'.format(root[2], len(imgs)))
        elif isinstance(root, list):
            imgs = []
            for sub_root in root:
                if sub_root[1] == 'image':
                    tmp = self.generateImagePair(sub_root[0])
                    imgs.extend(tmp)  # deal with image case
                    print('Image number of ImageSet {} is {}.'.format(sub_root[2], len(tmp)))
                elif sub_root[1] == 'video':
                    tmp = self.generateImagePairFromVideo(sub_root[0])
                    imgs.extend(tmp)  # transform video to images
                    print('Image number of VideoSet {} is {}.'.format(sub_root[2], len(tmp)))
        else:
            raise TypeError('root is tuple or list')
        return imgs

    def generateImagePairFromVideo(self, root):
        video_list = os.listdir(os.path.join(root, self.input_folder))
        if len(video_list) == 0:
            raise IOError('make sure the sequence path is correct')
        seq_imgs = []
        for video in video_list:
            seq_root = os.path.join(root, self.input_folder, video)
            label_root = os.path.join(root, self.label_folder, video)
            img_list = [os.path.splitext(f)[0] for f in os.listdir(seq_root) if f.endswith(self.img_ext)]
            img_pair = [(os.path.join(seq_root, img_name + self.img_ext),
              os.path.join(label_root, img_name + self.label_ext)) for img_name in img_list]
            seq_imgs.extend(img_pair)
        return seq_imgs


    def generateImagePair(self, root):
        img_list = [os.path.splitext(f)[0] for f in os.listdir(os.path.join(root, self.input_folder)) if f.endswith(self.img_ext)]
        if len(img_list) == 0:
            raise IOError('make sure the dataset path is correct')
        return [(os.path.join(root, self.input_folder, img_name + self.img_ext), os.path.join(root, self.label_folder, img_name + self.label_ext))
            for img_name in img_list]

    def __len__(self):
        return len(self.imgs)

