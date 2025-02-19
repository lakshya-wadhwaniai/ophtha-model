import random
from os import makedirs, listdir
from os.path import join
from imblearn.under_sampling import RandomUnderSampler
import pandas as pd
import numpy as np


#integrate in pipeline
DATA_ROOT_DIR = "/scratchk/ehealth/optha/"
APTOS_ROOT_DIR = join(DATA_ROOT_DIR, "aptos")
EYEPACS_ROOT_DIR = join(DATA_ROOT_DIR, "eyepacs")
ODIR_ROOT_DIR = join(DATA_ROOT_DIR, "odir")

def create_splits(ftrain=0.8, ftest=0.1, fval=0.1, split_name='test_1024', ROOT_PATH=APTOS_ROOT_DIR):

    split_dir = join(ROOT_PATH, f'data/splits/{split_name}_{ftrain}_{ftest}_{fval}/')
    df_labels = pd.read_csv(join(ROOT_PATH, 'train.csv'))

    df_labels['image'] = ROOT_PATH + '/data/processed_images_cv2/' + df_labels['image'] + '.png'

    images = list(df_labels['image'])
    images_shuffle = random.sample(images, int(len(images)))

    train_names = pd.DataFrame()
    val_names = pd.DataFrame()
    test_names = pd.DataFrame()

    train_names['image'] = images_shuffle[:int(ftrain*len(images_shuffle))]
    val_names['image'] = images_shuffle[int(ftrain*len(images_shuffle)):int((ftrain+fval)*len(images_shuffle))]
    test_names['image'] = images_shuffle[int((ftrain+fval)*len(images_shuffle)):]

    train_names = train_names.set_index('image').join(df_labels.set_index('image'))
    val_names = val_names.set_index('image').join(df_labels.set_index('image'))
    test_names = test_names.set_index('image').join(df_labels.set_index('image'))

    makedirs(split_dir, exist_ok = True)

    train_names.to_csv(join(split_dir, 'train.csv'))
    val_names.to_csv(join(split_dir, 'val.csv'))
    test_names.to_csv(join(split_dir, 'test.csv'))


if __name__ == '__main__':
    create_splits()