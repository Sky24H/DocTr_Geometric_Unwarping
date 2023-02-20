import os
from os.path import join as pjoin
import collections
import json
import torch
import numpy as np
import scipy.misc as m
import scipy.io as io
import matplotlib.pyplot as plt
import glob
import cv2
import random
import hdf5storage as h5

from tqdm import tqdm
from torch.utils import data

from .augmentationsk import data_aug, tight_crop

class doctr_geo_loader(data.Dataset):
    """
    Loader for world coordinate regression and RGB images
    """
    def __init__(self, root, split='train', is_transform=False,
                 img_size=512, augmentations=None):
        self.root = os.path.expanduser(root)
        self.split = split
        self.is_transform = is_transform
        # no augmentations for now?
        self.augmentations = None#augmentations
        self.n_classes = 3
        self.files = collections.defaultdict(list)
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)

        for split in ['train', 'val']:
            path = pjoin(self.root, split + '.txt')
            file_list = tuple(open(path, 'r'))
            file_list = [id_.rstrip() for id_ in file_list]
            self.files[split] = file_list
        #self.setup_annotations()
        if self.augmentations:
            self.txpths=[]
            with open(os.path.join(self.root[:-7],'augtexnames.txt'),'r') as f:
                for line in f:
                    txpth=line.strip()
                    self.txpths.append(txpth)


    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        im_name = self.files[self.split][index]                # 1/824_8-cp_Page_0503-7Nw0001
        im_path = pjoin(self.root, 'img',  im_name + '.png')  
        wc_path = pjoin(self.root, 'wc', im_name + '.exr')
        im = m.imread(im_path,mode='RGB')
        im = np.array(im, dtype=np.uint8)
        lbl = cv2.imread(wc_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        lbl = np.array(lbl, dtype=np.float)
        if 'val' in self.split:
            im, lbl=tight_crop(im/255.0,lbl)
        if self.augmentations:          #this is for training, default false for validation\
            tex_id=random.randint(0,len(self.txpths)-1)
            txpth=self.txpths[tex_id] 
            tex=cv2.imread(os.path.join(self.root[:-7],txpth)).astype(np.uint8)
            bg=cv2.resize(tex,self.img_size,interpolation=cv2.INTER_NEAREST)
            im,lbl=data_aug(im,lbl,bg)
        if self.is_transform:
            im1, lbl1 = self.transform_wc(im, lbl)

        recon_foldr='chess48'
        bm_path = pjoin(self.root, 'bm' , im_name + '.mat')
        alb_path = pjoin(self.root,'recon', im_name[:-4]+recon_foldr+'0001.png')

        wc = cv2.imread(wc_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        bm = h5.loadmat(bm_path)['bm']
        alb = m.imread(alb_path,mode='RGB')
        if self.is_transform:
            im2, lbl2 = self.transform_bm(wc,bm,alb)

        # return (im1, lbl1), (im2, lbl2)
        return im1, lbl2


    def transform_wc(self, img, lbl):
        img = m.imresize(img, self.img_size) # uint8 with RGB mode
        if img.shape[-1] == 4:
            img=img[:,:,:3]   # Discard the alpha channel  
        img = img[:, :, ::-1] # RGB -> BGR
        # plt.imshow(img)
        # plt.show()
        img = img.astype(float) / 255.0
        img = img.transpose(2, 0, 1) # NHWC -> NCHW
        lbl = lbl.astype(float)

        #normalize label
        msk=((lbl[:,:,0]!=0)&(lbl[:,:,1]!=0)&(lbl[:,:,2]!=0)).astype(np.uint8)*255
        xmx, xmn, ymx, ymn,zmx, zmn= 1.2539363, -1.2442188, 1.2396319, -1.2289206, 0.6436657, -0.67492497   # calculate from all the wcs
        lbl[:,:,0]= (lbl[:,:,0]-zmn)/(zmx-zmn)
        lbl[:,:,1]= (lbl[:,:,1]-ymn)/(ymx-ymn)
        lbl[:,:,2]= (lbl[:,:,2]-xmn)/(xmx-xmn)
        lbl=cv2.bitwise_and(lbl,lbl,mask=msk)
        lbl = cv2.resize(lbl, self.img_size, interpolation=cv2.INTER_NEAREST)
        lbl = lbl.transpose(2, 0, 1)   # NHWC -> NCHW
        lbl = np.array(lbl, dtype=np.float)

        # to torch
        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).float()

        return img, lbl



    def tight_crop(self, wc, alb):
        msk=((wc[:,:,0]!=0)&(wc[:,:,1]!=0)&(wc[:,:,2]!=0)).astype(np.uint8)
        size=msk.shape
        [y, x] = (msk).nonzero()
        minx = min(x)
        maxx = max(x)
        miny = min(y)
        maxy = max(y)
        wc = wc[miny : maxy + 1, minx : maxx + 1, :]
        alb = alb[miny : maxy + 1, minx : maxx + 1, :]
        
        s = 20
        wc = np.pad(wc, ((s, s), (s, s), (0, 0)), 'constant')
        alb = np.pad(alb, ((s, s), (s, s), (0, 0)), 'constant')
        cx1 = random.randint(0, s - 5)
        cx2 = random.randint(0, s - 5) + 1
        cy1 = random.randint(0, s - 5)
        cy2 = random.randint(0, s - 5) + 1

        wc = wc[cy1 : -cy2, cx1 : -cx2, :]
        alb = alb[cy1 : -cy2, cx1 : -cx2, :]
        t=miny-s+cy1
        b=size[0]-maxy-s+cy2
        l=minx-s+cx1
        r=size[1]-maxx-s+cx2

        return wc,alb,t,b,l,r

    def transform_bm(self, wc, bm, alb):
        wc,alb,t,b,l,r=self.tight_crop(wc,alb)               #t,b,l,r = is pixels cropped on top, bottom, left, right
        alb = m.imresize(alb, self.img_size) 
        alb = alb[:, :, ::-1] # RGB -> BGR
        alb = alb.astype(np.float64)
        if alb.shape[2] == 4:
            alb=alb[:,:,:3]
        alb = alb.astype(float) / 255.0
        alb = alb.transpose(2, 0, 1) # NHWC -> NCHW
       
        msk=((wc[:,:,0]!=0)&(wc[:,:,1]!=0)&(wc[:,:,2]!=0)).astype(np.uint8)*255
        #normalize label
        xmx, xmn, ymx, ymn,zmx, zmn= 1.2539363, -1.2442188, 1.2396319, -1.2289206, 0.6436657, -0.67492497
        wc[:,:,0]= (wc[:,:,0]-zmn)/(zmx-zmn)
        wc[:,:,1]= (wc[:,:,1]-ymn)/(ymx-ymn)
        wc[:,:,2]= (wc[:,:,2]-xmn)/(xmx-xmn)
        wc=cv2.bitwise_and(wc,wc,mask=msk)
        
        wc = m.imresize(wc, self.img_size) 
        wc = wc.astype(float) / 255.0
        wc = wc.transpose(2, 0, 1) # NHWC -> NCHW

        bm = bm.astype(float)
        #normalize label [-1,1]
        bm[:,:,1]=bm[:,:,1]-t
        bm[:,:,0]=bm[:,:,0]-l
        bm=bm/np.array([448.0-l-r, 448.0-t-b])
        bm=(bm-0.5)*2

        bm0=cv2.resize(bm[:,:,0],(self.img_size[0],self.img_size[1]))
        bm1=cv2.resize(bm[:,:,1],(self.img_size[0],self.img_size[1]))
        
        img=np.concatenate([alb,wc],axis=0)
        lbl=np.stack([bm0,bm1],axis=-1)

        # match the size of output
        lbl = lbl.transpose(2, 0, 1)   # NHWC -> NCHW
        lbl = np.array(lbl, dtype=np.float)

        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).float()
        return img, lbl

# #Leave code for debugging purposes
# if __name__ == '__main__':
#     local_path = '../datasets/doc3D'
#     bs = 4
#     dst = doctr_geo_loader(root=local_path, split='train', is_transform=True, augmentations=True)
#     trainloader = data.DataLoader(dst, batch_size=bs)
#     for i, data in enumerate(trainloader):
#         (imgs, labels), (imgs2, labels2) = data
#         imgs = imgs.numpy()
#         lbls = labels.numpy()
#         imgs = np.transpose(imgs, [0,2,3,1])
#         lbls = np.transpose(lbls, [0,2,3,1])
#         print(imgs.shape, lbls.shape)
#         # f, axarr = plt.subplots(bs, 2)
#         # for j in range(bs):
#         #     # print imgs[j].shape
#         #     axarr[j][0].imshow(imgs[j])
#         #     axarr[j][1].imshow(lbls[j])
#         # plt.show()
#         # a = raw_input()
#         # if a == 'ex':
#         #     break
#         # else:
#         #     plt.close()
