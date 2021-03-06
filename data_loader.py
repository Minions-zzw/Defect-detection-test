# data loader
import os
from torch.utils.data import Dataset, DataLoader
import cv2
import torch
from skimage import io, transform, color
import numpy as np
import math
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

class RescaleT(object):

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        if len(sample)==3:
            image, label0, label1 = sample
        else:
            image, label0 = sample
        h, w = image.shape[:2]

        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        # #resize the image to new_h x new_w and convert image from range [0,255] to [0,1]
        # img = transform.resize(image,(new_h,new_w),mode='constant')
        # lbl = transform.resize(label,(new_h,new_w),mode='constant', order=0, preserve_range=True)

        img = transform.resize(image, (self.output_size, self.output_size), mode='constant')
        lbl0 = transform.resize(label0, (self.output_size, self.output_size), mode='constant', order=0, preserve_range=True)
        if len(sample)==3:
           lbl1 = transform.resize(label1, (self.output_size, self.output_size), mode='constant', order=0, preserve_range=True)
           return img, lbl0, lbl1
        else:
            return img,lbl0

class RandomCrop(object):

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):

        if len(sample) == 3:
            image, label0, label1 = sample
        else:
            image, label0 = sample

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h, left: left + new_w]
        label0 = label0[top: top + new_h, left: left + new_w]
        if len(sample) == 3:
            label1 = label1[top: top + new_h, left: left + new_w]
            return image, label0, label1
        else:
            return image,label0

class ToTensorLab(object):
	"""Convert ndarrays in sample to Tensors."""
	def __init__(self,flag=0):
		self.flag = flag

	def __call__(self, sample):

            if len(sample )== 3:
                image, label0, label1 = sample
            else:
                image, label0 = sample


            # change the color space
            if self.flag == 2: # with rgb and Lab colors
                        tmpImg = np.zeros((image.shape[0],image.shape[1],6))
                        tmpImgt = np.zeros((image.shape[0],image.shape[1],3))
                        if image.shape[2]==1:
                            tmpImgt[:,:,0] = image[:,:,0]
                            tmpImgt[:,:,1] = image[:,:,0]
                            tmpImgt[:,:,2] = image[:,:,0]
                        else:
                            tmpImgt = image
                        tmpImgtl = color.rgb2lab(tmpImgt)

                        # nomalize image to range [0,1]
                        tmpImg[:,:,0] = (tmpImgt[:,:,0]-np.min(tmpImgt[:,:,0]))/(np.max(tmpImgt[:,:,0])-np.min(tmpImgt[:,:,0]))
                        tmpImg[:,:,1] = (tmpImgt[:,:,1]-np.min(tmpImgt[:,:,1]))/(np.max(tmpImgt[:,:,1])-np.min(tmpImgt[:,:,1]))
                        tmpImg[:,:,2] = (tmpImgt[:,:,2]-np.min(tmpImgt[:,:,2]))/(np.max(tmpImgt[:,:,2])-np.min(tmpImgt[:,:,2]))
                        tmpImg[:,:,3] = (tmpImgtl[:,:,0]-np.min(tmpImgtl[:,:,0]))/(np.max(tmpImgtl[:,:,0])-np.min(tmpImgtl[:,:,0]))
                        tmpImg[:,:,4] = (tmpImgtl[:,:,1]-np.min(tmpImgtl[:,:,1]))/(np.max(tmpImgtl[:,:,1])-np.min(tmpImgtl[:,:,1]))
                        tmpImg[:,:,5] = (tmpImgtl[:,:,2]-np.min(tmpImgtl[:,:,2]))/(np.max(tmpImgtl[:,:,2])-np.min(tmpImgtl[:,:,2]))

                        # tmpImg = tmpImg/(np.max(tmpImg)-np.min(tmpImg))

                        tmpImg[:,:,0] = (tmpImg[:,:,0]-np.mean(tmpImg[:,:,0]))/np.std(tmpImg[:,:,0])
                        tmpImg[:,:,1] = (tmpImg[:,:,1]-np.mean(tmpImg[:,:,1]))/np.std(tmpImg[:,:,1])
                        tmpImg[:,:,2] = (tmpImg[:,:,2]-np.mean(tmpImg[:,:,2]))/np.std(tmpImg[:,:,2])
                        tmpImg[:,:,3] = (tmpImg[:,:,3]-np.mean(tmpImg[:,:,3]))/np.std(tmpImg[:,:,3])
                        tmpImg[:,:,4] = (tmpImg[:,:,4]-np.mean(tmpImg[:,:,4]))/np.std(tmpImg[:,:,4])
                        tmpImg[:,:,5] = (tmpImg[:,:,5]-np.mean(tmpImg[:,:,5]))/np.std(tmpImg[:,:,5])

            elif self.flag == 1: #with Lab color
                        tmpImg = np.zeros((image.shape[0],image.shape[1],3))

                        if image.shape[2]==1:
                            tmpImg[:,:,0] = image[:,:,0]
                            tmpImg[:,:,1] = image[:,:,0]
                            tmpImg[:,:,2] = image[:,:,0]

                        tmpImg = color.rgb2lab(tmpImg)

                        # tmpImg = tmpImg/(np.max(tmpImg)-np.min(tmpImg))

                        tmpImg[:,:,0] = (tmpImg[:,:,0]-np.min(tmpImg[:,:,0]))/(np.max(tmpImg[:,:,0])-np.min(tmpImg[:,:,0]))
                        tmpImg[:,:,1] = (tmpImg[:,:,1]-np.min(tmpImg[:,:,1]))/(np.max(tmpImg[:,:,1])-np.min(tmpImg[:,:,1]))
                        tmpImg[:,:,2] = (tmpImg[:,:,2]-np.min(tmpImg[:,:,2]))/(np.max(tmpImg[:,:,2])-np.min(tmpImg[:,:,2]))

                        tmpImg[:,:,0] = (tmpImg[:,:,0]-np.mean(tmpImg[:,:,0]))/np.std(tmpImg[:,:,0])
                        tmpImg[:,:,1] = (tmpImg[:,:,1]-np.mean(tmpImg[:,:,1]))/np.std(tmpImg[:,:,1])
                        tmpImg[:,:,2] = (tmpImg[:,:,2]-np.mean(tmpImg[:,:,2]))/np.std(tmpImg[:,:,2])

            else: # with rgb color
                        tmpImg = np.zeros((image.shape[0],image.shape[1],3))
                        image = image/np.max(image)
                        if image.shape[2]==1:
                            tmpImg[:,:,0] = (image[:,:,0]-0.485)/0.229
                            tmpImg[:,:,1] = (image[:,:,0]-0.485)/0.229
                            tmpImg[:,:,2] = (image[:,:,0]-0.485)/0.229
                        else:
                            tmpImg[:,:,0] = (image[:,:,0]-0.485)/0.229
                            tmpImg[:,:,1] = (image[:,:,1]-0.456)/0.224
                            tmpImg[:,:,2] = (image[:,:,2]-0.406)/0.225



            tmpImg = tmpImg.transpose((2, 0, 1))
            label0 = label0.transpose((2, 0, 1))
            if len(sample) == 3:
                label1 = label1.transpose((2, 0, 1))
                return  torch.from_numpy(tmpImg),torch.from_numpy(label0),torch.from_numpy(label1)
            else:
                return torch.from_numpy(tmpImg), torch.from_numpy(label0)


class SalObjDataset(Dataset):
    def __init__(self, image_root, gt_root,transform):
        self.images = [image_root + f for f in os.listdir(image_root)]
        self.gts = [gt_root + f for f in os.listdir(gt_root)]
        # self.edges = [egde_root + f for f in os.listdir(egde_root)]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        # self.edges = sorted(self.edges)
        self.transform = transform



    def __getitem__(self, index):


        image = io.imread(self.images[index])
        label = io.imread(self.gts[index])
        # label_1=  io.imread(self.edges[index])

        label0 = np.zeros(label.shape[0:2])
        if (3 == len(label.shape)):
            label0 = label[:, :, 0]
        elif (2 == len(label.shape)):
            label0 = label

        if (np.max(label0) < 1e-6):
            label0 = label0
        else:
            label0 = label0 / np.max(label0)


        # label1 = np.zeros(label_1.shape[0:2])
        # if (3 == len(label1.shape)):
        #     label1 = label1[:, :, 0]
        # elif (2 == len(label0.shape)):
        #     label1= label1
        # label1 = label1 / 255

        if (3 == len(image.shape) and 2 == len(label0.shape)):
            label0 = label0[:, :, np.newaxis]
            # label1 = label1[:, :, np.newaxis]
        elif (2 == len(image.shape) and 2 == len(label0.shape) ):
            image = image[:, :, np.newaxis]
            label0 = label0[:, :, np.newaxis]
            # label1 = label1[:, :, np.newaxis]

        sample = [image,label0]
        sample = self.transform(sample)


        return sample


    def __len__(self):
        return len(self.images)
