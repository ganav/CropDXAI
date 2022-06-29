
import tensorflow as tf
#from skimage import data, io, filters
import numpy as np
from numpy import array
from numpy.random import randint
#from scipy.misc import imresize
from scipy import ndimage
import os,cv2,csv,sys, glob,math
import matplotlib.pyplot as plt
import tensorflow.keras as Ker
from tensorflow.keras.preprocessing.image import img_to_array
# Display
from IPython.display import Image, display
import matplotlib.cm as cm

plt.switch_backend('agg')

def get_img_parts(img_path):
    """Given a full path to a video, return its parts."""
    parts = img_path.split('\\')

    leng=len(parts)
    filename = parts[leng-1]
    path=parts[0]+'\\'
 
    for i in range(1,leng-3):
        path=path+parts[i]+'\\'
    return path+'b\\'+parts[4]+'\\',  filename

def normalize(input_data):
    return (input_data / 255.).astype(np.float32)#(input_data.astype(np.float32) - 127.5)/127.5

    
def denormalize(input_data):
    input_data = (input_data + 1) * 127.5
    return input_data.astype(np.uint8)
    
def load_data_from_dirs(dirs, ext, n_class,ss):
    files1,files2,files3 = [],[],[]

    if dirs == 'test':

        with open('.\\data\\'+dirs+'\\a\\'+'list2.csv', 'r') as fin:
            reader = csv.reader(fin)
            data = list(reader)    

        for d in range(len(data)):
            imgs_A = cv2.imread('.\\data\\'+dirs+'\\a\\0\\'+data[d][1],0)# cv2.IMREAD_COLOR  cv2.IMREAD_GRAYSCALE , cv2.IMREAD_UNCHANGED, 1, 0 or -1
      
            #comment below 4 lines in case of color input
            imgs_A = imgs_A[np.newaxis,...]#add 1 more dimension
            imgs_A = imgs_A[np.newaxis,...]#add 1 more dimension
            imgs_A = np.moveaxis(imgs_A, 0, -1)#reorder the shape of dimension


            files1.append(imgs_A)
            files3.append(int(data[d][0]))
            
    else:
        with open('.\\data\\'+dirs+'\\a\\'+'list.csv', 'r') as fin:
            reader = csv.reader(fin)
            data = list(reader)
        for d in range(len(data)):
            imgs_A = cv2.imread('.\\data\\'+dirs+'\\a\\'+data[d][0]+'\\'+data[d][1],0)
            imgs_B = cv2.imread('.\\data\\'+dirs+'\\b\\'+data[d][0]+'\\'+data[d][1],0)
            
            imgs_B = cv2.resize(imgs_B, (ss, ss), interpolation = cv2.INTER_AREA)

            #comment below 4 lines in case of color input
            imgs_A = imgs_A[np.newaxis,...]#add 1 more dimension
            imgs_A = imgs_A[np.newaxis,...]#add 1 more dimension
            imgs_A = np.moveaxis(imgs_A, 0, -1)#reorder the shape of dimension

            imgs_B = imgs_B[np.newaxis,...]#add 1 more dimension
            imgs_B = imgs_B[np.newaxis,...]#add 1 more dimension
            imgs_B = np.moveaxis(imgs_B, 0, -1)#reorder the shape of dimension


            files1.append(imgs_A)
            files2.append(imgs_B)
            files3.append(int(data[d][0])-1)#minus 1 coz class starts from 1 but in python array it starts from 0

        files2 = normalize(array(files2))

    files1 = normalize(array(files1))

    print(len(files1))
    print(len(files3))

    return files1,files2,files3     
    
def load_training_data(directory, ext,n_class,ss):
    x_train,y_train,labels = load_data_from_dirs(directory, ext, n_class,ss)
    return x_train,y_train,labels
