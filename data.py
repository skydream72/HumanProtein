import cv2
import glob
import os
import numpy as np
from keras.utils.np_utils import to_categorical
import scipy.misc
import matplotlib.pyplot as plt
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tqdm import tqdm
import pickle
from skimage import io
import tables
import numpy as np

LABEL_PATH = '/mnt/DataStorage/NIH/Data_Entry_2017.csv'
LABEL_BBOX_PATH = '/mnt/DataStorage/NIH/BBox_List_2017.csv'

def load_hp_data_hd5_tf(data_path, image_size, label_path, labels_size, hd5_name):
    df_train = pd.read_csv(label_path)

    targets_count = np.zeros(labels_size)
    
    data_model = tables.open_file(hd5_name, mode='w')
    atom = tables.Float64Atom()
    X = data_model.create_earray(data_model.root, 'data', atom, (0, image_size, image_size, 4))
    y = data_model.create_earray(data_model.root, 'tag', atom, (0, labels_size))
    idx = 0
    for row in tqdm(df_train.values[:]):
        f = row[0]
        tags = row[1]
        
        tag_files = [data_path+'/{}'.format(f)+'_blue.png', data_path+'/{}'.format(f)+'_green.png', data_path+'/{}'.format(f)+'_blue.png', data_path+'/{}'.format(f)+'_yellow.png']

        if not os.path.exists(tag_files[0]):
            continue
            
        new_image = np.zeros((0, image_size, image_size))
        #print(file_name)
        for file in tag_files:
            idx += 1
                
            img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
            #print('print img shape = ', img.shape)
            #img = cv2.resize(img, (image_size, image_size))

                #print('show origin image')
                #io.imshow(img)
                #io.show()
                #print('show flip image')
                #io.imshow(img_f)
                #io.show()

            img = img / 255.
            img = np.expand_dims(img, axis=0)
            new_image = np.vstack((new_image, img))
        targets = np.zeros(labels_size)     
        targets = targets.reshape(1, len(targets))
        for t in tags.split(' '):
            targets[0][int(t)] = 1
            targets_count[int(t)] += 1
        #print('new_image shape = ', new_image.shape)
        new_image = np.transpose(new_image, (1,2,0))
        new_image = np.expand_dims(new_image, axis=0)
        X.append(new_image)
        y.append(targets)
                  
    data_model.close()        
    print('tags count = ', targets_count)

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return p