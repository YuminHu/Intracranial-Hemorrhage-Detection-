import numpy as np 
import pandas as pd 
import cv2
import random
import pydicom
import torch
from torch.utils.data.dataset import Dataset



########
#read csv dataset containing ID and labels
########

def read_testset(filename = DATA_DIR + "stage_2_sample_submission.csv"):
    df = pd.read_csv(filename)
    df["Image"] = df["ID"].str.slice(stop=12)
    df["Diagnosis"] = df["ID"].str.slice(start=13)
    df = df.loc[:, ["Label", "Diagnosis", "Image"]]
    df = df.set_index(['Image', 'Diagnosis']).unstack(level=-1)
    return df

def read_trainset(filename = DATA_DIR + "stage_2_train.csv"):
    df = pd.read_csv(filename)
    df["Image"] = df["ID"].str.slice(stop=12)
    df["Diagnosis"] = df["ID"].str.slice(start=13)
    duplicates_to_remove = [
        56346,   56347,   56348,   56349,   56350,   56351,
        1171830, 1171831, 1171832, 1171833, 1171834, 1171835,
        3705312, 3705313, 3705314, 3705315, 3705316, 3705317,
        3842478, 3842479, 3842480, 3842481, 3842482, 3842483
    ]
    df = df.drop(index = duplicates_to_remove)
    df = df.reset_index(drop = True)    
    df = df.loc[:, ["Label", "Diagnosis", "Image"]]
    df = df.set_index(['Image', 'Diagnosis']).unstack(level=-1)
    
    return df



########
#create image from dicom
########
def window_image(dicom, window_center, window_width):
    image_min = window_center - window_width // 2
    image_max = window_center + window_width // 2
    image = np.clip(dicom, image_min, image_max)
    image = (image-image_min)/(image_max-image_min)
    return image

def load_dicom(dir,image_id):
    dicom = pydicom.dcmread(dir + '%s.dcm'%image_id)
    dico = dicom.pixel_array * dicom.RescaleSlope + dicom.RescaleIntercept

    image = dico
    if image.shape[:2]!=(512,512):
        image = cv2.resize(image, dsize=(512,512),interpolation=cv2.INTER_LINEAR)

    return image

def load_ssb_dicom(dir,image_id):
    dico = load_dicom(dir,image_id)
    brain       = window_image(dico, 40,  80)
    subdural    = window_image(dico, 80, 200)
    soft_tissue = window_image(dico, 40, 380)

    image = np.dstack([soft_tissue,subdural,brain])

    return image



########
#image augmentation
########


def randomHorizontalFlip(image, p=0.5):
    if np.random.random() < p:
        image = cv2.flip(image, 1)
    return image

def random_cropping(image, ratio=0.8, is_random = True):
    height, width, _ = image.shape
    target_h = int(height*ratio)
    target_w = int(width*ratio)

    if is_random:
        start_x = random.randint(0, width - target_w)
        start_y = random.randint(0, height - target_h)
    else:
        start_x = ( width - target_w ) // 2
        start_y = ( height - target_h ) // 2

    zeros = image[start_y:start_y+target_h,start_x:start_x+target_w,:]
    zeros = cv2.resize(zeros ,(width,height))
    return zeros

def random_noise(image, noise=0.1, p=0.5):
    if np.random.random() < p:
        H,W = image.shape[:2]
        image = image + np.random.uniform(-1,1,(H,W,1))*noise
        image = np.clip(image,0,1)
    return image

def aug_image(image):

    image = randomHorizontalFlip(image)
    image = random_noise(image)
    image = random_cropping(image, ratio=random.uniform(0.6,0.99), is_random=True)
    return image



########
#dataset for training
########

class IntraDataset(Dataset):
    def __init__(self, df, load_image_function, augment=None,mode="train",img_dir = TRAIN_IMAGES_DIR):
        self.augment = augment
        self.df = df
        self.uid = df.index
        self.load_image_function = load_image_function
        self.label = df.Label.values
        self.mode = mode
        self.img_dir = img_dir

    def __len__(self):
        return len(self.uid)
    
    def __getitem__(self,index):
        image_id = self.uid[index]
        image = self.load_image_function(self.img_dir , image_id)

        label = self.label[index]
        if self.mode != "train":
            return image.transpose(2,0,1)
        if self.augment is None:
            return image.transpose(2,0,1), label
        else:
            return self.augment(image).transpose(2,0,1),label


def run_check_train_dataset(train_df):
    
    dataset = IntraDataset(
        df = train_df,
        load_image_function=load_ssb_dicom,
        augment = aug_image, 
    )
    return dataset

if __name__ == '__main__':
    DATA_DIR = 'rsna-intracranial-hemorrhage-detection/'
    TRAIN_IMAGES_DIR = DATA_DIR + 'stage_2_train/'
    train_df = read_trainset(DATA_DIR + "stage_2_train.csv")
    run_check_train_dataset(train_df)
