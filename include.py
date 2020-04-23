import time
import numpy as np
import pandas as pd


import torch
from tqdm.notebook import tqdm
import torch.optim as optim
from sklearn.model_selection import KFold, train_test_split
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet
from torch.utils.data import DataLoader

##path
DATA_DIR = 'rsna-intracranial-hemorrhage-detection/'
TRAIN_IMAGES_DIR = DATA_DIR + 'stage_2_train/'
TEST_IMAGES_DIR = DATA_DIR + 'stage_2_test/'

##constant
BATCH_SIZE = 32
N_CLASSES = 6