from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
import numpy as np 
import os
import glob
import skimage.io as io
import skimage.transform as trans
import cv2

COLOR_DICT = {'Grass' : [0,0,255] , 'Building' :[0,255,255], 'Tree' :[0,255,0] , 'Car':[255,255,0] , 'Sand':[255,0,0],
                        'Road':[255,255,255], 'Unlabeled':[0,0,0] }
Grass =[0,0,255]
Building=[0,255,255]
Tree=[0,255,0]
Car=[255,255,0]
Sand=[255,0,0]
Road=[255,255,255]
COLOR_DICT = np.array([Grass, Building, Tree, Car, Sand,Road])
label=[29,179,149,225,76,255]
def adjustData(img,mask,flag_multi_class,num_class):
    if(flag_multi_class):
        img = img / 255             
        mask = mask[:,:,:,0] if(len(mask.shape) == 4) else mask[:,:,0]
        new_mask = np.zeros(mask.shape + (num_class,))
        for i in range(num_class):
            #for one pixel in the image, find the class in mask and convert it into one-hot vector
            index = np.where(mask == label[i])
            index_mask = (index[0],index[1],index[2],np.zeros(len(index[0]),dtype = np.int64) + i)
            new_mask[index_mask] = 1
        mask = new_mask
    return (img,mask)



def trainGenerator(batch_size,train_path,image_folder,mask_folder,aug_dict,image_color_mode = "grayscale",
                    mask_color_mode = "grayscale",image_save_prefix  = "image",mask_save_prefix  = "mask",
                    flag_multi_class = True,num_class = 6,save_to_dir = None,target_size = (256,256),seed = 1):

    image_datagen = ImageDataGenerator(None)
    mask_datagen = ImageDataGenerator(None)
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes = [image_folder],
        class_mode = None,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = image_save_prefix,
        seed = seed)
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes = [mask_folder],
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = mask_save_prefix,
        seed = seed)
    train_generator = zip(image_generator, mask_generator)
    for (img,mask) in train_generator:
        img,mask = adjustData(img,mask,flag_multi_class,num_class)
        yield (img,mask)



def testGenerator(test_path,num_image = 10,target_size = (256,256),as_gray = True):
    for i in range(num_image):
        img = io.imread(os.path.join(test_path,"%d.png"%i))
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        gray = gray / 255
        gray = np.reshape(gray,(1,)+gray.shape+(1,))
        yield gray

        
def labelVisualize(num_class,color_dict,img):
    for i in range(256):
      for j in range(256):
        tmp = img[i][j]
        new_label = [int(k==np.max(tmp)) for k in tmp]
        img[i][j] = np.array(new_label)
    mask_RGB = np.zeros((img.shape[0],img.shape[1]) + (3,))
    mask_gray = np.sum(img*label, axis=2)
    for i in range(num_class):
      idx=np.where(mask_gray==label[i])
      mask_RGB[idx]=COLOR_DICT[i]
    return mask_RGB
def saveResult(save_path,npyfile,flag_multi_class = True,num_class = 6):
    for i,item in enumerate(npyfile):
        img = labelVisualize(num_class,COLOR_DICT,item)
        io.imsave(os.path.join(save_path,"%d_predict.png"%i),img)