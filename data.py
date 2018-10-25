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

FLAS_NIH_IMAGE_SIZE = 224
    
def load_one_nih_data(file_name):
    #flatten = lambda l: [item for sublist in l for item in sublist]
    #labels = list(set(flatten([l.split('|') for l in df_train['Finding Labels'].values])))
    
    #get label if in bbox list
    df_train = pd.read_csv(LABEL_BBOX_PATH)
    bbox = dict()
    tags = dict()
    for row in df_train.values:
        f = row[0]
        if f == file_name:
            if row[1] == 'Infiltrate':
                tags['bbox'] = 'Infiltration'
            else:
                tags['bbox'] = row[1]
            
            bbox['x'] = int(row[2])
            bbox['y'] = int(row[3])
            bbox['w'] = int(row[4])
            bbox['h'] = int(row[5])
    
    #get label if in full data list, the tag is more correctly in Data Entry table
    df_train = pd.read_csv(LABEL_PATH)
    for row in df_train.values:
        f = row[0]
        if f == file_name:
            tags['ground_true'] = row[1]
            tags['follow_up'] = row[2]
            tags['age'] = row[4]
            tags['gender'] = row[5]
            tags['view_position'] = row[6]
    return tags, bbox
        
def load_all_nih_bbox_data():
    #get label if in bbox list
    df_train = pd.read_csv(LABEL_BBOX_PATH)

    return df_train

def load_nih_test_data_hd5(data_path, hd5_name):
    labels = []
    for c in sorted(os.listdir(data_path)):
        labels.append(c)
    print("labels = ", labels)
    label_map = {l: i for i, l in enumerate(labels)}
    #one hot encording
    labels_size = len(labels)
    targets_count = np.zeros(labels_size)
    
    data_model = tables.open_file(hd5_name, mode='w')
    atom = tables.Float64Atom()
    X = data_model.create_earray(data_model.root, 'data', atom, (0, 3, FLAS_NIH_IMAGE_SIZE, FLAS_NIH_IMAGE_SIZE))
    y = data_model.create_earray(data_model.root, 'tag', atom, (0, labels_size))
    

    for label in tqdm(labels):
        data_label_path = os.path.join(data_path,label)
        for file in os.listdir(data_label_path):
            f = file

            #print(data_label_path+'/{}'.format(f))
            if os.path.exists(data_label_path+'/{}'.format(f)):

                img = cv2.imread(data_label_path+'/{}'.format(f), cv2.IMREAD_COLOR)
                img = cv2.resize(img, (FLAS_NIH_IMAGE_SIZE, FLAS_NIH_IMAGE_SIZE))
                
                img = np.transpose(img, (2, 0, 1)) / 255.
                targets = np.zeros(labels_size)
                targets[label_map[label]] = 1
                targets_count[label_map[label]] += 1
                #print('f = ', f, 'label = ', targets)
                img = img.reshape(1,3,FLAS_NIH_IMAGE_SIZE,FLAS_NIH_IMAGE_SIZE)
                X.append(img)
                targets = targets.reshape(1,labels_size)
                y.append(targets)
        
    data_model.close()        
    print('tags count = ', targets_count)
    
def load_nih_gray_data_hd5(data_path, label_path, hd5_name):
    
    df_train = pd.read_csv(label_path)

    flatten = lambda l: [item for sublist in l for item in sublist]
    labels = list(set(flatten([l.split('|') for l in df_train['Finding Labels'].values])))
    labels = labels[1:]
    print("labels = ", labels)
    label_map = {l: i for i, l in enumerate(labels)}
    inv_label_map = {i: l for l, i in label_map.items()}

    #one hot encording
    labels_size = len(labels)
    targets_count = np.zeros(labels_size + 1)
    
    data_model = tables.open_file(hd5_name, mode='w')
    atom = tables.Float64Atom()
    X = data_model.create_earray(data_model.root, 'data', atom, (0, 1, FLAS_NIH_IMAGE_SIZE, FLAS_NIH_IMAGE_SIZE))
    y = data_model.create_earray(data_model.root, 'tag', atom, (0, labels_size))
    idx = 0
    temp_X = []
    temp_y = []
    for row in tqdm(df_train.values):
        f = row[0]
        tags = row[1]
        #print(data_path+'/{}'.format(f))
        if os.path.exists(data_path+'/{}'.format(f)):
            idx += 1
            #print(data_path+'/{}'.format(f))
            #print("file exist")
            img = cv2.imread(data_path+'/{}'.format(f), cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (FLAS_NIH_IMAGE_SIZE, FLAS_NIH_IMAGE_SIZE))
            img = img.reshape(FLAS_NIH_IMAGE_SIZE,FLAS_NIH_IMAGE_SIZE,1)
            #print('print img shape = ', img.shape)
            
            #flip samples
            img_f = cv2.flip(img,1)
            img_f = img_f.reshape(FLAS_NIH_IMAGE_SIZE,FLAS_NIH_IMAGE_SIZE,1)
            #print('print img f shape = ', img_f.shape)
            
            #print('show origin image')
            #io.imshow(img)
            #io.show()
            #print('show flip image')
            #io.imshow(img_f)
            #io.show()
       
            img = np.transpose(img, (2, 0, 1)) / 255.
            img_f = np.transpose(img_f, (2, 0, 1)) / 255.
            targets = np.zeros(labels_size)

            if tags != 'No Finding':
                for t in tags.split('|'):
                    targets[label_map[t]] = 1
                    targets_count[label_map[t]] += 1

                    #oversampling twice
                    for i in range(8):
                        temp_X.append(img)
                        temp_y.append(targets)
                        #print('y0 = ', y[-1], 'x0 =', X[-1])
                        temp_X.append(img_f)
                        temp_y.append(targets)
                        #print('y1 = ', y[-1], 'x1 =', X[-1])
                        #print('label_map[t] = ', label_map[t], 'tags = ', tags, 'taget = ', targets)
            else:
                if idx%2==0:
                    targets_count[-1] += 1
                    temp_X.append(img)
                    temp_y.append(targets)
            
            if idx%1000==0:
                #print('list size = ', len(temp_X))
                a_X = np.array(temp_X, np.float16)
                a_y = np.array(temp_y, np.uint8)
                p = unison_shuffled_copies(a_X, a_y)
                X.append(a_X[p])
                y.append(a_y[p])
                temp_X = []
                temp_y = []
    #append last data
    a_X = np.array(temp_X, np.float16)
    a_y = np.array(temp_y, np.uint8)
    p = unison_shuffled_copies(a_X, a_y)
    X.append(a_X[p])
    y.append(a_y[p])           
    data_model.close()        
    print('tags count = ', targets_count)

def load_nih_data_hd5_no_finding_in(data_path, label_path, csv_index, hd5_name):
    
    df_train = pd.read_csv(label_path)

    flatten = lambda l: [item for sublist in l for item in sublist]
    labels = list(set(flatten([l.split('|') for l in df_train['Finding Labels'].values])))
    labels = labels[0:]
    print("labels = ", labels)
    label_map = {l: i for i, l in enumerate(labels)}
    inv_label_map = {i: l for l, i in label_map.items()}

    #one hot encording
    labels_size = len(labels)
    targets_count = np.zeros(labels_size)
    
    data_model = tables.open_file(hd5_name, mode='w')
    atom = tables.Float64Atom()
    X = data_model.create_earray(data_model.root, 'data', atom, (0, 3, FLAS_NIH_IMAGE_SIZE, FLAS_NIH_IMAGE_SIZE))
    y = data_model.create_earray(data_model.root, 'tag', atom, (0, labels_size))
    idx = 0
    temp_X = []
    temp_y = []
    for row in tqdm(df_train.values):
        f = row[0]
        tags = row[1]
        #print(data_path+'/{}'.format(f))
        if os.path.exists(data_path+'/{}'.format(f)):
            idx += 1
            #print(data_path+'/{}'.format(f))
            #print("file exist")
            img = cv2.imread(data_path+'/{}'.format(f), cv2.IMREAD_COLOR)
            img = cv2.resize(img, (FLAS_NIH_IMAGE_SIZE, FLAS_NIH_IMAGE_SIZE))
            #img = img.reshape(FLAS_NIH_IMAGE_SIZE,FLAS_NIH_IMAGE_SIZE,1)
            #print('print img shape = ', img.shape)
            
            #flip samples
            img_f = cv2.flip(img,1)
            #img_f = img_f.reshape(FLAS_NIH_IMAGE_SIZE,FLAS_NIH_IMAGE_SIZE,1)
            #print('print img f shape = ', img_f.shape)
            
            #print('show origin image')
            #io.imshow(img)
            #io.show()
            #print('show flip image')
            #io.imshow(img_f)
            #io.show()
       
            img = np.transpose(img, (2, 0, 1)) / 255.
            img_f = np.transpose(img_f, (2, 0, 1)) / 255.
            targets = np.zeros(labels_size)

            
            for t in tags.split('|'):
                targets[label_map[t]] = 1
                targets_count[label_map[t]] += 1

            temp_X.append(img)
            temp_y.append(targets)


            
            if idx%1000==0:
                #print('list size = ', len(temp_X))
                a_X = np.array(temp_X, np.float16)
                a_y = np.array(temp_y, np.uint8)
                p = unison_shuffled_copies(a_X, a_y)
                X.append(a_X[p])
                y.append(a_y[p])
                temp_X = []
                temp_y = []
    #append last data
    a_X = np.array(temp_X, np.float16)
    a_y = np.array(temp_y, np.uint8)
    p = unison_shuffled_copies(a_X, a_y)
    X.append(a_X[p])
    y.append(a_y[p])           
    data_model.close()        
    print('tags count = ', targets_count)

def load_nih_data_hd5_tf(data_path, label_path, csv_index, hd5_name):
    print('csv_index = ', csv_index)
    df_train = pd.read_csv(label_path)

    flatten = lambda l: [item for sublist in l for item in sublist]
    labels = list(set(flatten([l.split('|') for l in df_train['Finding Labels'].values])))
    #print("labels raw = ", labels)
    labels.remove('No Finding')
    print("labels = ", labels)
    label_map = {l: i for i, l in enumerate(labels)}
    #inv_label_map = {i: l for l, i in label_map.items()}

    #one hot encording
    labels_size = len(labels)
    targets_count = np.zeros(labels_size + 1)
    
    data_model = tables.open_file(hd5_name, mode='w')
    atom = tables.Float64Atom()
    X = data_model.create_earray(data_model.root, 'data', atom, (0, FLAS_NIH_IMAGE_SIZE, FLAS_NIH_IMAGE_SIZE, 3))
    y = data_model.create_earray(data_model.root, 'tag', atom, (0, labels_size))
    idx = 0
    for row in tqdm(df_train.values[csv_index:]):
        f = row[0]
        tags = row[1]
        #print('f = ', f, 'tags = ', tags)
        #print(data_path+'/{}'.format(f))
        if os.path.exists(data_path+'/{}'.format(f)):
            idx += 1
            #print(data_path+'/{}'.format(f))
            #print("file exist")
            img = cv2.imread(data_path+'/{}'.format(f), cv2.IMREAD_COLOR)
            img = cv2.resize(img, (FLAS_NIH_IMAGE_SIZE, FLAS_NIH_IMAGE_SIZE))
            
            #print('print img shape = ', img.shape)
            
            #flip samples
            img_f = cv2.flip(img,1)
            
            #print('print img f shape = ', img_f.shape)
            
            #print('show origin image')
            #io.imshow(img)
            #io.show()
            #print('show flip image')
            #io.imshow(img_f)
            #io.show()
       
            img = img / 255.
            img_f = img_f / 255.
            targets = np.zeros(labels_size)     
                        
            img = img.reshape(1, FLAS_NIH_IMAGE_SIZE,FLAS_NIH_IMAGE_SIZE, 3)
            img_f = img_f.reshape(1, FLAS_NIH_IMAGE_SIZE,FLAS_NIH_IMAGE_SIZE, 3)
            targets = targets.reshape(1, len(targets))
            if tags != 'No Finding':
                for t in tags.split('|'):
                    #only train the single disease patient
                    targets[0][label_map[t]] = 1
                    targets_count[label_map[t]] += 2

                X.append(img)
                y.append(targets)
                    
                X.append(img_f)
                y.append(targets)
            else:
                #if idx%2==0:
                X.append(img)
                y.append(targets)
                targets_count[-1] += 1
                  
    data_model.close()        
    print('tags count = ', targets_count)

def load_hp_data_hd5_tf(data_path, label_path, labels_size, hd5_name):
    df_train = pd.read_csv(label_path)

    targets_count = np.zeros(labels_size)
    
    data_model = tables.open_file(hd5_name, mode='w')
    atom = tables.Float64Atom()
    X = data_model.create_earray(data_model.root, 'data', atom, (0, FLAS_NIH_IMAGE_SIZE, FLAS_NIH_IMAGE_SIZE, 4))
    y = data_model.create_earray(data_model.root, 'tag', atom, (0, labels_size))
    idx = 0
    for row in tqdm(df_train.values[:]):
        f = row[0]
        tags = row[1]
        
        tag_files = [data_path+'/{}'.format(f)+'_blue.png', data_path+'/{}'.format(f)+'_green.png', data_path+'/{}'.format(f)+'_blue.png', data_path+'/{}'.format(f)+'_yellow.png']

        if not os.path.exists(tag_files[0]):
            continue
            
        new_image = np.zeros((0, FLAS_NIH_IMAGE_SIZE, FLAS_NIH_IMAGE_SIZE))
        #print(file_name)
        for file in tag_files:
            idx += 1
                
            img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
            #print('print img shape = ', img.shape)
            img = cv2.resize(img, (FLAS_NIH_IMAGE_SIZE, FLAS_NIH_IMAGE_SIZE))

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
    
def load_nih_data_hd5(data_path, label_path, csv_index, hd5_name):
    print('csv_index = ', csv_index)
    df_train = pd.read_csv(label_path)

    flatten = lambda l: [item for sublist in l for item in sublist]
    labels = list(set(flatten([l.split('|') for l in df_train['Finding Labels'].values])))
    labels = labels[1:]
    print("labels = ", labels)
    label_map = {l: i for i, l in enumerate(labels)}
    inv_label_map = {i: l for l, i in label_map.items()}

    #one hot encording
    labels_size = len(labels)
    targets_count = np.zeros(labels_size + 1)
    
    data_model = tables.open_file(hd5_name, mode='w')
    atom = tables.Float64Atom()
    X = data_model.create_earray(data_model.root, 'data', atom, (0, 3, FLAS_NIH_IMAGE_SIZE, FLAS_NIH_IMAGE_SIZE))
    y = data_model.create_earray(data_model.root, 'tag', atom, (0, labels_size))
    idx = 0
    for row in tqdm(df_train.values[csv_index:]):
        f = row[0]
        tags = row[1]
        #print('f = ', f, 'tags = ', tags)
        #print(data_path+'/{}'.format(f))
        if os.path.exists(data_path+'/{}'.format(f)):
            idx += 1
            #print(data_path+'/{}'.format(f))
            #print("file exist")
            img = cv2.imread(data_path+'/{}'.format(f), cv2.IMREAD_COLOR)
            img = cv2.resize(img, (FLAS_NIH_IMAGE_SIZE, FLAS_NIH_IMAGE_SIZE))
            
            #print('print img shape = ', img.shape)
            
            #flip samples
            img_f = cv2.flip(img,1)
            
            #print('print img f shape = ', img_f.shape)
            
            #print('show origin image')
            #io.imshow(img)
            #io.show()
            #print('show flip image')
            #io.imshow(img_f)
            #io.show()
       
            img = np.transpose(img, (2, 0, 1)) / 255.
            img_f = np.transpose(img_f, (2, 0, 1)) / 255.
            targets = np.zeros(labels_size)     
                        
            img = img.reshape(1, 3, FLAS_NIH_IMAGE_SIZE,FLAS_NIH_IMAGE_SIZE)
            img_f = img_f.reshape(1, 3, FLAS_NIH_IMAGE_SIZE,FLAS_NIH_IMAGE_SIZE)
            targets = targets.reshape(1, len(targets))
            if tags != 'No Finding':
                for t in tags.split('|'):
                    #only train the single disease patient
                    targets[0][label_map[t]] = 1
                    targets_count[label_map[t]] += 1

                X.append(img)
                y.append(targets)
                    
                X.append(img_f)
                y.append(targets)
                #targets_count[label_map[t]] += 1
            else:
                X.append(img)
                y.append(targets)
                targets_count[-1] += 1
                  
    data_model.close()        
    print('tags count = ', targets_count)

def load_nih_data_hd5_binary(data_path, label_path, hd5_name):
    
    df_train = pd.read_csv(label_path)

    flatten = lambda l: [item for sublist in l for item in sublist]
    labels = list(set(flatten([l.split('|') for l in df_train['Finding Labels'].values])))
    labels = labels[1:]
    print("labels = ", labels)
    label_map = {l: i for i, l in enumerate(labels)}
    inv_label_map = {i: l for l, i in label_map.items()}

    #one hot encording
    labels_size = 1
    targets_count = np.zeros(labels_size+1)
    
    data_model = tables.open_file(hd5_name, mode='w')
    atom = tables.Float64Atom()
    X = data_model.create_earray(data_model.root, 'data', atom, (0, 3, FLAS_NIH_IMAGE_SIZE, FLAS_NIH_IMAGE_SIZE))
    y = data_model.create_earray(data_model.root, 'tag', atom, (0, labels_size))
    idx = 0
    temp_X = []
    temp_y = []
    for row in tqdm(df_train.values):
        f = row[0]
        tags = row[1]
        #print(data_path+'/{}'.format(f))
        if os.path.exists(data_path+'/{}'.format(f)):
            idx += 1
            #print(data_path+'/{}'.format(f))
            #print("file exist")
            img = cv2.imread(data_path+'/{}'.format(f), cv2.IMREAD_COLOR)
            img = cv2.resize(img, (FLAS_NIH_IMAGE_SIZE, FLAS_NIH_IMAGE_SIZE))
            #img = img.reshape(FLAS_NIH_IMAGE_SIZE,FLAS_NIH_IMAGE_SIZE,1)
            #print('print img shape = ', img.shape)
            
            #flip samples
            img_f = cv2.flip(img,1)
            #img_f = img_f.reshape(FLAS_NIH_IMAGE_SIZE,FLAS_NIH_IMAGE_SIZE,1)
            #print('print img f shape = ', img_f.shape)
            
            #print('show origin image')
            #io.imshow(img)
            #io.show()
            #print('show flip image')
            #io.imshow(img_f)
            #io.show()
       
            img = np.transpose(img, (2, 0, 1)) / 255.
            img_f = np.transpose(img_f, (2, 0, 1)) / 255.
            targets = np.zeros(labels_size)

            if tags != 'No Finding':
                for t in tags.split('|'):
                    if t=='Pneumonia':
                        targets[0] = 1
                        targets_count[0] += 1
                        
                        #for i in range(2):
                        temp_X.append(img)
                        temp_y.append(targets)

                        temp_X.append(img_f)
                        temp_y.append(targets)
            else:
                targets_count[-1] += 1
                temp_X.append(img)
                temp_y.append(targets)
            
            if idx%1000==0:
                #print('list size = ', len(temp_X))
                a_X = np.array(temp_X, np.float16)
                a_y = np.array(temp_y, np.uint8)
                p = unison_shuffled_copies(a_X, a_y)
                X.append(a_X[p])
                y.append(a_y[p])
                temp_X = []
                temp_y = []
    #append last data
    a_X = np.array(temp_X, np.float16)
    a_y = np.array(temp_y, np.uint8)
    p = unison_shuffled_copies(a_X, a_y)
    X.append(a_X[p])
    y.append(a_y[p])           
    data_model.close()        
    print('tags count = ', targets_count)   

def load_nih_data(data_path, label_path, pkl_name):
    X = []
    y = []
    
    df_train = pd.read_csv(label_path)

    flatten = lambda l: [item for sublist in l for item in sublist]
    labels = list(set(flatten([l.split('|') for l in df_train['Finding Labels'].values])))
    labels = labels[1:]
    print("labels = ", labels)
    label_map = {l: i for i, l in enumerate(labels)}
    inv_label_map = {i: l for l, i in label_map.items()}

    #one hot encording
    labels_size = len(labels)
    targets_count = np.zeros(labels_size + 1)
    for row in tqdm(df_train.values):
        f = row[0]
        tags = row[1]
        #print(data_path+'/{}'.format(f))
        if os.path.exists(data_path+'/{}'.format(f)):
            #print(data_path+'/{}'.format(f))
            #print("file exist")
            img = cv2.imread(data_path+'/{}'.format(f), cv2.IMREAD_COLOR)
            img = cv2.resize(img, (FLAS_NIH_IMAGE_SIZE, FLAS_NIH_IMAGE_SIZE))
            #img = img.reshape(FLAS_NIH_IMAGE_SIZE,FLAS_NIH_IMAGE_SIZE,1)
            #print('print img shape = ', img.shape)
            
            #flip samples
            img_f = cv2.flip(img,1)
            #img_f = img_f.reshape(FLAS_NIH_IMAGE_SIZE,FLAS_NIH_IMAGE_SIZE,1)
            #print('print img f shape = ', img_f.shape)
            
            #print('show origin image')
            #io.imshow(img)
            #io.show()
            #print('show flip image')
            #io.imshow(img_f)
            #io.show()
       
            img = np.transpose(img, (2, 0, 1)) / 255.
            img_f = np.transpose(img_f, (2, 0, 1)) / 255.
            targets = np.zeros(labels_size)
            if tags != 'No Finding':
                for t in tags.split('|'):
                    targets[label_map[t]] = 1
                    targets_count[label_map[t]] += 1
                    
                    
                    #oversampling twice
                    for i in range(2):
                        X.append(img)
                        y.append(targets)
                        #print('y0 = ', y[-1], 'x0 =', X[-1])
                        X.append(img_f)
                        y.append(targets)
                        #print('y1 = ', y[-1], 'x1 =', X[-1])
                        #print('label_map[t] = ', label_map[t], 'tags = ', tags, 'taget = ', targets)
            else:
                targets_count[-1] += 1
                X.append(img)
                y.append(targets)
                #print('tags = ', tags, 'taget = ', targets)
    print('tags count = ', targets_count)
    print('convert list to array')
    y = np.array(y, np.uint8)
    X = np.array(X, np.float16)
    print('shuffled array')
    #r_X, r_y = unison_shuffled_copies(X, y)
    print('saving dataset')
    output = open(pkl_name, 'wb')
    # Pickle dictionary using protocol 0.
    pickle.dump(X, output)
    pickle.dump(y, output)
    # Pickle the list using the highest protocol available.
    output.close()
    #print('X shape = ', r_X.shape, ' y shape = ', r_y.shape, 'len tag = ', len(y[0]))
    #print(y)
    return X, y

def load_nih_data_only14disease(data_path, label_path, data_name):
    X = []
    y = []
    
    df_train = pd.read_csv(label_path)

    flatten = lambda l: [item for sublist in l for item in sublist]
    labels = list(set(flatten([l.split('|') for l in df_train['Finding Labels'].values])))
    labels = labels[1:]
    print("labels = ", labels)
    label_map = {l: i for i, l in enumerate(labels)}
    inv_label_map = {i: l for l, i in label_map.items()}

    #one hot encording
    labels_size = len(labels)
    for row in tqdm(df_train.values):
        f = row[0]
        tags = row[1]
        #print(data_path+'/{}'.format(f))
        if os.path.exists(data_path+'/{}'.format(f)):
            #print(data_path+'/{}'.format(f))
            #print("file exist")
            img = cv2.imread(data_path+'/{}'.format(f))
            #print('print img shape = ', img.shape)
            img = cv2.resize(img, (FLAS_NIH_IMAGE_SIZE, FLAS_NIH_IMAGE_SIZE))
            img = np.transpose(img, (2, 0, 1))
            targets = np.zeros(labels_size)
            if tags != 'No Finding':
                for t in tags.split('|'):
                    targets[label_map[t]] = 1 
            #print(targets)
                X.append(img)
                y.append(targets)

    y = np.array(y, np.uint8)
    X = np.array(X, np.float16) / 255.
    output = open(data_name, 'wb')
    # Pickle dictionary using protocol 0.
    pickle.dump(X, output)
    pickle.dump(y, output)
    # Pickle the list using the highest protocol available.
    output.close()
    print('X shape = ', X.shape, ' y shape = ', y.shape, 'len tag = ', len(y[0]))
    #print(y)
    return X, y

def load_nih_data_binary(data_path, label_path):
    X = []
    y = []
    
    df_train = pd.read_csv(label_path)

    flatten = lambda l: [item for sublist in l for item in sublist]
    labels = list(set(flatten([l.split('|') for l in df_train['Finding Labels'].values])))
    labels = labels[1:]
    print("labels = ", labels)
    label_map = {l: i for i, l in enumerate(labels)}
    inv_label_map = {i: l for l, i in label_map.items()}

    #one hot encording
    labels_size = len(labels)
    for row in tqdm(df_train.values):
        f = row[0]
        tags = row[1]
        #print(data_path+'/{}'.format(f))
        if os.path.exists(data_path+'/{}'.format(f)):
            #print(data_path+'/{}'.format(f))
            #print("file exist")
            img = cv2.imread(data_path+'/{}'.format(f))
            #print('print img shape = ', img.shape)
            img = cv2.resize(img, (FLAS_NIH_IMAGE_SIZE, FLAS_NIH_IMAGE_SIZE))
            img = np.transpose(img, (2, 0, 1))
            targets = np.zeros(1)
            if tags != 'No Finding':
                targets = 1
            else:
                targets = 0
            #print(targets)
            X.append(img)
            y.append(targets)

    y = np.array(y, np.uint8)
    X = np.array(X, np.float16) / 255.
    output = open(data_name, 'wb')
    # Pickle dictionary using protocol 0.
    pickle.dump(X, output)
    pickle.dump(y, output)
    # Pickle the list using the highest protocol available.
    output.close()
    print('X shape = ', X.shape, ' y shape = ', y.shape)
    #print(y)
    return X, y

def load_inria_person(path):
    front_path = os.path.join(path, "front")
    side_path = os.path.join(path, "side")
    back_path = os.path.join(path, "back")
    mask_path = os.path.join(path, "mask")
    background_path = os.path.join(path, "background")
    
    front_images = [cv2.imread(x) for x in glob.glob(front_path + "/*.png")]
    front_images = [np.transpose(img, (2, 0, 1)) for img in front_images]
    
    side_images = [cv2.imread(x) for x in glob.glob(side_path + "/*.png")]
    side_images = [np.transpose(img, (2, 0, 1)) for img in side_images]
    
    back_images = [cv2.imread(x) for x in glob.glob(back_path + "/*.png")]
    back_images = [np.transpose(img, (2, 0, 1)) for img in back_images]
    
    mask_images = [cv2.imread(x) for x in glob.glob(mask_path + "/*.png")]
    mask_images = [np.transpose(img, (2, 0, 1)) for img in mask_images]
    
    background_images = [cv2.imread(x) for x in glob.glob(background_path + "/*.png")]
    background_images = [np.transpose(img, (2, 0, 1)) for img in background_images]
    
    y = [0] * len(front_images) + [1] * len(side_images) + [2] * len(back_images) + [3] * len(mask_images) + [3] * len(background_images)
    y = to_categorical(y, 4)
    y[len(front_images)+len(side_images)+len(back_images)+len(mask_images):] = 0
    X = np.float32(front_images + side_images + back_images + mask_images + background_images)
    
    return X, y

IMAGE_HEIGHT_SCALE_SIZE = 2.5

def load_virus_data(path):
    a_path = os.path.join(path, "A")
    b_path = os.path.join(path, "B")
    neg_path = os.path.join(path, "neg")
    
    a_images = [scipy.misc.imread(x) for x in glob.glob(a_path + "/*.bmp")]
    print(a_images[0])
    for idx, img in enumerate(a_images):
        #fig = plt.figure(figsize=(10,4))
        #plt.imshow(img)
        #plt.show()
        
        height, width, chann = img.shape
        # We pick the largest center square.
        centery = height // 2
        centerx = width // 2
        radius = min((centerx, centery))
        centerx = centerx + 100
        img = img[centery-radius:centery+radius, centerx-radius:centerx+radius]
        img = scipy.misc.imresize(img, size=(int(height*IMAGE_HEIGHT_SCALE_SIZE), width), interp='bilinear')
        img = np.transpose(img, (2, 0, 1))
        a_images[idx] = img
    print(a_images[0])
    b_images = [scipy.misc.imread(x) for x in glob.glob(b_path + "/*.bmp")]
    for idx, img in enumerate(b_images):
        height, width, chann = img.shape
        # We pick the largest center square.
        centery = height // 2
        centerx = width // 2
        radius = min((centerx, centery))
        centerx = centerx + 100
        img = img[centery-radius:centery+radius, centerx-radius:centerx+radius]
        img = np.transpose(img, (2, 0, 1))
        img = scipy.misc.imresize(img, size=(int(height*IMAGE_HEIGHT_SCALE_SIZE), width), interp='bilinear')
        img = np.transpose(img, (2, 0, 1))
        b_images[idx] = img
        
    neg_images = [scipy.misc.imread(x) for x in glob.glob(neg_path + "/*.bmp")]
    for idx, img in enumerate(neg_images):
        height, width, chann = img.shape
        # We pick the largest center square.
        centery = height // 2
        centerx = width // 2
        radius = min((centerx, centery))
        centerx = centerx + 100
        img = img[centery-radius:centery+radius, centerx-radius:centerx+radius]
        img = scipy.misc.imresize(img, size=(int(height*IMAGE_HEIGHT_SCALE_SIZE), width), interp='bilinear')
        img = np.transpose(img, (2, 0, 1))
        neg_images[idx] = img
    
    y = [0] * len(a_images) + [1] * len(b_images) + [2] * len(neg_images)
    y = to_categorical(y, 3)
    X = np.float32(a_images + b_images + neg_images)
    
    return X, y

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return p