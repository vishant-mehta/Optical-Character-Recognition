import cv2
import string
import multiprocessing
from tqdm import tqdm
from functools import partial
import numpy as np


def augment(db):

    img = db[0]
    label = db[1]
    img = cv2.resize(img, (128,1024))
    ne1 = cv2.resize(img[62:962, 33:95], (128,1024)) 
    ne2 = cv2.resize(img[62:962, 0:128], (128,1024)) 
    ne3 = cv2.resize(img[0:1024, 33:95], (128,1024)) 

    return img, ne1, ne2, ne3, label


def data_augmentation(dataset):
    
    dataset_aug = dict()
    for i in ['train','test','valid']:
        dataset_aug[i] = {"dt": [], "gt": []}
    
    for y in ['train','test','valid']:
        results = []
        db_list = []
        for j in range(len(dataset[y]['dt'])):
            tup = (dataset[y]['dt'][j],dataset[y]['gt'][j])
            db_list.append(tup)
        with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
            print(f"Partition: {y}")
            for result in tqdm(pool.imap(augment, db_list),total=len(dataset[y]['dt']), position=0, leave=True):
                for r in range(0,4):
                    dataset_aug[y]['dt'].append(result[r])
                    dataset_aug[y]['gt'].append(result[4])
            pool.close()
            pool.join()           
    
    return dataset_aug
