from typing import Any, Tuple, Dict
import torchvision.transforms as transforms
import torch
import os
import PIL
from PIL import Image, ImageOps,ImageFilter
from collections import defaultdict
import numpy as np
from itertools import product
from generate_augmentations import Augmentation
import tqdm
import cv2
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
# dicts ={}
# dicts['n01749939']='green mamba'
# dicts['n02086910']='papillon'
# dicts['n03017168']='chime'
# dicts['n03947888']='pirate'
# dicts['n07836838']='chocolate sauce'
class ImageNetDataset():

    def __init__(self,  
                 synthetic_probability = 0.5,
                 max_classes = 100,
                 strength= 0.1,
                 guidance=1,
                 image_size= (256, 256)):
        self.class_names=[] 
        self.class_id_to_name={}
        self.class_name_to_id={}
        self.augmentations = Augmentation(strength=strength,guidance=guidance)
        self.batch_size=2
        t=0
        self.samples=[]
        self.target=[]
        self.classes={}
        # transforms.Normalize(
        #                                     [0.48145466, 0.4578275, 0.40821073],
        #                                     [0.26862954, 0.26130258, 0.27577711]),
        self.transforms = transforms.Compose([
                                            transforms.ToTensor(),
                                            transforms.Resize((224,224),
                                            interpolation=transforms.InterpolationMode.BICUBIC,
                                            antialias=False,),
                                        ])
        class_list = os.listdir('./ImageNet100/train')
        # os.listdir('./ImageNet5/train')
        for x in class_list:
            self.class_names.append(x)
            self.class_id_to_name[x]=t
            self.class_name_to_id[t]=x
            # self.classes[t] = dicts[x]
            t+=1
        
        for j in self.class_names:
            for k in os.listdir('./ImageNet100/train/'+j):
                self.samples.append('./ImageNet100/train/'+j+'/'+k)
                os.makedirs('./diffimage_new/quickdraw/'+j+'/'+k, exist_ok=True)
                self.target.append(self.class_id_to_name[j])
        print(self.samples)       
        train_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15.0),
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float),
            transforms.Lambda(lambda x: x.expand(3, *image_size)),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], 
                                  std=[0.5, 0.5, 0.5])
        ])

        val_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float),
            transforms.Lambda(lambda x: x.expand(3, *image_size)),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], 
                                  std=[0.5, 0.5, 0.5])
        ])

        self.transform = train_transform

    def __len__(self):
        
        return len(self.samples)

    def get_image_by_idx(self, idx): #x(self, idxl,idxu):

        # L= []
        # # self.transforms(Image.open(self.samples[idxl]).convert('RGB'))
        # for i  in self.samples[idxl:idxu]:
        #     L.append(self.transforms(Image.open(i).convert('RGB')).unsqueeze(0))
        image = Image.open(self.samples[idx])
        # image = PIL.ImageOps.exif_transpose(image)
        image = image.convert("RGB").resize((224, 224))
        # res = cv2.edgePreservingFilter(np.array(image) , sigma_s=60, sigma_r=0.6)
        # image = image.filter(ImageFilter.FIND_EDGES)
        # image = ImageOps.invert(image)
        # img = np.array(image).astype(np.uint8)
        # img = np.expand_dims(img, axis=2)
        # img = np.repeat(img, repeats = 3, axis=2)
        # image = Image.fromarray(res)
        # image = image.filter(ImageFilter.GaussianBlur(radius = 5))
        return image,self.samples[idx] #torch.cat(L,dim=0),self.samples[idxl:idxu]

    def get_label_by_idx(self, idx):

        return self.target[idx]
    def generate_augmentation(self):
        o=0
        for idx in tqdm.tqdm(range(0,len(self.samples)), desc="Generating Augmentations"):
            image,path = self.get_image_by_idx(idx)
            # print(path)
            if o==0:
                print(path)
            o+=1
            label = self.get_label_by_idx(idx)
            # print(image)
            # print(label)
            # print(path)
            # filename=path.split("/")[-1]
            class_names=[self.class_name_to_id[label]]
            # for i in label:
            #     class_names.append(self.class_name_to_id[i])
            # print(class_names)
            # class_names = 
            aug_img = self.augmentations(image)
            t=0
            r=0
            for i in range(4):
                r=0
                for j in class_names:
                        # print(path[r])
                        filename=path.split("/")[-1]
                        # print('../diffimage/'+j+'/'+filename+'/img'+str(i)+'.jpg')
                        aug_img[t].save('./diffimage_new/quickdraw/'+j+'/'+filename+'/img'+str(i)+'.jpg')
                        t+=1
                        r+=1
                    # aug_img[t][i].save('../diffimage/'+class_names+'/'+filename+'/img'+str(i)+'.jpg')
        
    
#     def get_metadata_by_idx(self, idx: int) -> Dict:

#         return dict(name=self.class_names[self.all_labels[idx]])
