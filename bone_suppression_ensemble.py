#check cuda device status
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"  # specify which GPU(s) to be used

#%% clear session and check available gpus
from keras import backend as K
K.clear_session()
K.tensorflow_backend._get_available_gpus()

#%%
#import libraries
import tensorflow as tf
import cv2
import os
from os import listdir,makedirs
from os.path import isfile,join
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from keras import regularizers
from keras.layers import Input,Dense,Conv2D,MaxPooling2D,UpSampling2D,BatchNormalization, add, Activation
from keras.layers import Add, SeparableConv2D, MaxPool2D, Conv2DTranspose, Input, Lambda, Dense, Flatten, LeakyReLU, PReLU
from keras.models import Model, load_model, model_from_json, save_model
from keras.optimizers import RMSprop, Adam
from keras import backend as K
from keras.losses import mean_absolute_error, mean_squared_error
from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau, EarlyStopping
from keras.preprocessing import image
from skimage import io, img_as_ubyte
from skimage.util import img_as_float
from skimage.measure import compare_ssim
from scipy.io import savemat 
from skimage import color
import time
from tqdm import tqdm
import math
from math import log10, sqrt
import struct
import glob
import shutil
import zlib
from PIL import Image
import imageio

#%%
#get current path
print(os.getcwd())

#%% loss functions

def tf_log10(x):
    numerator = tf.log(x)
    denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator

def PSNR(y_true, y_pred):
    max_pixel = 1.0
    return 10.0 * tf_log10((max_pixel ** 2) / (K.mean(K.square(y_pred - y_true)))) 

def mse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred)

def mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)

def ssim(y_true, y_pred):
    return tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))

def ssim_loss(y_true, y_pred):
    return 1-tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))

def ssim_multi(y_true, y_pred):
    return tf.reduce_mean(tf.image.ssim_multiscale(y_true, y_pred, 1.0))

def ssim_multi_loss(y_true, y_pred):
    return 1-tf.reduce_mean(tf.image.ssim_multiscale(y_true, y_pred, 1.0))

def mix_loss(y_true, y_pred):    
    return 0.16 * mean_absolute_error(y_true, y_pred) +\
             0.84 * (1-ssim_multi(y_true, y_pred)) 

#%% load data

def load_data(no_of_images):
    
    img_size = (256,256)
    imgs_source = []
    imgs_target = []
    
    dir_source = "data/source" 
    dir_target = "data/target"
    
    i = 0
    for _, _, filenames in os.walk('data/source/'):
        for filename in filenames:
            i = i+1
            if(i > no_of_images):
                break
            #img_source = cv2.imread(os.path.join(dir_source,filename)) #for U-Net and FPN
            img_source = cv2.imread(os.path.join(dir_source,filename),cv2.IMREAD_GRAYSCALE) # for other models
            img_target = cv2.imread(os.path.join(dir_target, filename),cv2.IMREAD_GRAYSCALE)
            
            # resizing images
            img_source = cv2.resize(img_source,img_size)
            img_target = cv2.resize(img_target,img_size)
            img_source = np.array(img_source)/255
            img_target = np.array(img_target)/255            
            imgs_source.append(img_source)
            imgs_target.append(img_target)
    return imgs_source, imgs_target

source, target = load_data(1000)

#%% load test data

def load_test_data(no_of_images):
    
    img_size = (256,256)
    imgs_source = []
    imgs_target = []
    
    dir_source = "data/test_source" 
    dir_target = "data/test_target"
    
    i = 0
    for _, _, filenames in os.walk('data/test_source/'):
        for filename in filenames:
            i = i+1
            if(i > no_of_images):
                break
            # img_source = cv2.imread(os.path.join(dir_source,filename)) #for U-Net and FPN models
            img_source = cv2.imread(os.path.join(dir_source,filename),cv2.IMREAD_GRAYSCALE) # for other models
            img_target = cv2.imread(os.path.join(dir_target, filename),cv2.IMREAD_GRAYSCALE)
            
            # resizing images
            img_source = cv2.resize(img_source,img_size)
            img_target = cv2.resize(img_target,img_size)
            img_source = np.array(img_source)/255
            img_target = np.array(img_target)/255            
            imgs_source.append(img_source)
            imgs_target.append(img_target)
    return imgs_source, imgs_target

source_test, target_test = load_test_data(27)

#%% Let's see some of the source images and their corresponding targets.

plt.figure(figsize=(15,10))

for i in range(3):
    ax = plt.subplot(2, 3, i+1)
    plt.imshow(source[i])
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.title('Source')

    ax = plt.subplot(2, 3, i+4)
    plt.imshow(target[i])
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.title('Target')
plt.show()

#%% Let's see some of the test images and their corresponding targets.

plt.figure(figsize=(15,10))

for i in range(3):
    ax = plt.subplot(2, 3, i+1)
    plt.imshow(source_test[i])
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.title('Source')

    ax = plt.subplot(2, 3, i+4)
    plt.imshow(target_test[i])
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.title('Target')
plt.show()

#%%
#format image shape 
'''
check the size of the source and target train images in load data.
IF they are grayscale, use img_channels_rgb = 1 for the models that train
with grayscale images. if the input data is 3-channel, then use
img_channels_rgb = 3 for training the ImageNet models

'''
#%%
img_rows = 256
img_cols = 256
img_channels_rgb = 1 # change to 3 channels for U-Net and FPN
img_channels_gray = 1
img_shape = (img_rows, img_cols, img_channels_rgb)

source = np.array(source).reshape(-1, img_rows, img_cols, img_channels_rgb)
target = np.array(target).reshape(-1, img_rows, img_cols, img_channels_gray)

source_train, source_val, target_train, target_val = train_test_split(source, target,
                                                                        test_size=0.1,
                                                                        random_state=42) # 10% for validation
source_test = np.array(source_test).reshape(-1, img_rows, img_cols, img_channels_rgb)
target_test = np.array(target_test).reshape(-1, img_rows, img_cols, img_channels_gray)
print(source_train.shape, source_val.shape, source_test.shape,
      target_train.shape, target_val.shape, target_test.shape)

#%%
'''
Models used in this study:
Autoencoder with separable convlutions
ResNet model with separable convolutions, num_filters=128, num_res_blocks=16, res_block_scaling=0.1
U-Net variants: efficientnet-B0, ResNet-18, SE-ResNet-18, DenseNet-121
Inception-V3, and MobileNet-V2
FPN variants: efficientnet-B0, ResNet-18, SE-ResNet-18, DenseNet-121
Inception-V3, and MobileNet-V2 
'''
#%%
#Autoencoder with separable convolutions: works with grayscale inputs and targets
  
def autoencoder_sep(input_img):
    
    #encoder
    x1 = SeparableConv2D(64, (3, 3), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l1(10e-10))(input_img) 
    x2 = SeparableConv2D(64, (3, 3), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l1(10e-10))(x1) 
    x3 = MaxPool2D(padding='same')(x2)
    x4 = SeparableConv2D(128, (3, 3), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l1(10e-10))(x3) 
    x5 = SeparableConv2D(128, (3, 3), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l1(10e-10))(x4) 
    x6 = MaxPool2D(padding='same')(x5)
    x7 = SeparableConv2D(256, (3, 3), activation='relu', padding='same', 
                     kernel_regularizer=regularizers.l1(10e-10))(x6)
    x8 = SeparableConv2D(256, (3, 3), activation='relu', padding='same', 
                     kernel_regularizer=regularizers.l1(10e-10))(x7)
    x9 = MaxPool2D(padding='same')(x8)
    encoded = SeparableConv2D(512, (3, 3), activation='relu', padding='same', 
                     kernel_regularizer=regularizers.l1(10e-10))(x9)
    
    #decoder
    x10 = UpSampling2D()(encoded)
    x11 = SeparableConv2D(256, (3, 3), activation='relu', padding='same', 
                kernel_regularizer=regularizers.l1(10e-10))(x10)
    x12 = SeparableConv2D(256, (3, 3), activation='relu', padding='same', 
                kernel_regularizer=regularizers.l1(10e-10))(x11)
    x13 = Add()([x8, x12])
    
    x14 = UpSampling2D()(x13)    
    x15 = SeparableConv2D(128, (3, 3), activation='relu', padding='same', 
                 kernel_regularizer=regularizers.l1(10e-10))(x14)
    x16 = SeparableConv2D(128, (3, 3), activation='relu', padding='same', 
                 kernel_regularizer=regularizers.l1(10e-10))(x15)
    x17 = Add()([x5, x16])
    x18 = UpSampling2D()(x17) 
    x19 = SeparableConv2D(64, (3, 3), activation='relu', padding='same', 
                 kernel_regularizer=regularizers.l1(10e-10))(x18)
    x20 = SeparableConv2D(64, (3, 3), activation='relu', padding='same', 
                 kernel_regularizer=regularizers.l1(10e-10))(x19)
    x21 = Add()([x2, x20]) 
    
    decoded = SeparableConv2D(1, (3, 3), padding='same',activation='relu', 
                     kernel_regularizer=regularizers.l1(10e-10))(x21)
    return decoded

input_img = Input(shape = img_shape)
ae_sep = Model(input_img, autoencoder_sep(input_img), name='AE_Separable')
ae_sep.summary()

#%%
#ResNet model with num_filters=128, num_res_blocks=16, res_block_scaling=0.1
  
def res_block_new(x_in, filters, scaling):
    x = Conv2D(filters, 3, padding='same', activation='relu')(x_in)
    x = Conv2D(filters, 3, padding='same')(x)
    if scaling:
        x = Lambda(lambda t: t * scaling)(x)
    x = Add()([x_in, x])
    return x

def res_scale(scale, num_filters=64, num_res_blocks=16, 
         res_block_scaling=None): 
    x_in = Input(shape=(256,256,1)) #grayscale input and output
    x = b = Conv2D(num_filters, 3, padding='same')(x_in)
    for i in range(num_res_blocks):
        b = res_block_new(b, num_filters, res_block_scaling)
    b = Conv2D(num_filters, 3, padding='same')(b)
    x = Add()([x, b])
    x = Conv2D(1, 3, padding='same')(x)
    return Model(x_in, x, name="ResNet-BS")

#instantiate the model
resnet_scale = res_scale(1, num_filters=128, num_res_blocks=16, 
                      res_block_scaling=0.1)
resnet_scale.summary()

#%%
'''
We are going to use the U-Net model with various ImageNet-pretrained
classifier backbones from https://github.com/qubvel/segmentation_models
The following backbones are used:
'resnet18' 'seresnet18' 'densenet121' 'inceptionv3', 'efficientnetb0',
'mobilenetv2' 
'''
#%%
import segmentation_models as sm
print(sm.__version__)

#%%
BACKBONE = 'seresnet18' #one of 'resnet18' 'seresnet18' 'densenet121' 'inceptionv3', 'efficientnetb0', 'mobilenetv2'  

# define model
model_unet = sm.Unet(BACKBONE, input_shape=(256,256,3), 
                         encoder_weights='imagenet',
                         classes=1, activation='sigmoid') 
model_unet.summary()

#%%
'''
Next, we will use the FPN (Feature pyramid network) model with various ImageNet-pretrained
classifier backbones from https://github.com/qubvel/segmentation_models
The following backbones are used:
'resnet18' 'seresnet18' 'densenet121' 'inceptionv3', 'efficientnetb0', 'mobilenetv2'  

'''
#%%
   
BACKBONE = 'efficientnetb0' 

#define model
modelfpn_ef0 = sm.FPN(BACKBONE, input_shape=(256,256,3), 
                       classes=1, activation='sigmoid')
modelfpn_ef0.summary()

#%%
# Train each model: for instance we show training the FPN model

n_epoch = 200
n_batch = 8

filepath='weights/' + modelfpn_ef0.name +'.fpn_eb0_bs.h5' 
checkpoint = ModelCheckpoint(filepath, 
                             monitor='val_loss', 
                             verbose=1, 
                             save_weights_only=True,
                             save_best_only=True, 
                             mode='min') 
earlyStopping = EarlyStopping(monitor='val_loss', 
                               patience=10, 
                               verbose=1, 
                               mode='min')
tensor_board = TensorBoard(log_dir='logs/', 
                           histogram_freq=0, 
                           batch_size=n_batch)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', 
                              factor=0.5, 
                              patience=10,
                              verbose=1, 
                              mode='min', 
                              min_lr=0.00001)
callbacks_list = [checkpoint, tensor_board, earlyStopping, reduce_lr]
t=time.time()
modelfpn_ef0.compile(optimizer=Adam(lr=0.001), 
                  loss=mix_loss, 
                  metrics=[mse, mae, ssim_multi_loss,PSNR, ssim, ssim_multi]) 

modelfpn_ef0_train = modelfpn_ef0.fit(source_train, 
                                target_train,
                                epochs = n_epoch,
                                batch_size = n_batch,
                                verbose = 1,
                                callbacks=callbacks_list,
                                shuffle=True,
                                validation_data = (source_test, target_test))

print('Training time: %s' % (time.time()-t))

#%%
#plot performance

N = 200 #modify if early stopping
plt.style.use("ggplot")
plt.figure(figsize=(20,10), dpi=100)
plt.plot(np.arange(1, N+1), 
         modelfpn_ef0_train.history["loss"], 'orange', label="train_mix_loss")
plt.plot(np.arange(1, N+1), 
         modelfpn_ef0_train.history["val_loss"], 'red', label="val_mix_loss")
plt.plot(np.arange(1, N+1), 
         modelfpn_ef0_train.history["mse"], 'blue', label="MSE_loss")
plt.plot(np.arange(1, N+1), 
         modelfpn_ef0_train.history["ssim_multi_loss"], 'green', label="MS-SSIM_loss")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc="lower right")
plt.savefig("FPN_EB0_performance.png", dpi=600)

#%%
# predict performance on the test data 

modelfpn_ef0.load_weights("weights/model_fpn_eb0_b.h5") # Path to the saved model
print("Loaded model from disk")
modelfpn_ef0.summary()

#compile model
modelfpn_ef0.compile(optimizer=Adam(lr=0.001), 
                  loss=mix_loss, 
                  metrics=[mse, mae, ssim_multi_loss, PSNR, ssim, ssim_multi])

# evaluate model
t=time.time()
print('-'*30)
print('evaluating on test data...')
print('-'*30)
y_pred_fpn = modelfpn_ef0.evaluate(source_test, target_test, verbose=1)
print('Testing time: %s' % (time.time()-t))

#%%
'''
Predict the bone suppressed image for the test data
using each of the trained models. Repeat the process for other unseen data
to generate bone suppressed image using the best performing model
'''
#%%
# load test data
source = glob.glob('data/test_source/*.png') # _3c for Imagenet models
source.sort()

#%%
#load model
modelfpn_ef0.load_weights("weights/model_fpn_eb0_b.h5")
modelfpn_ef0.summary()

#%%
# run the loop to genrate and save bone suppressed image

for f in source:
    img = Image.open(f)
    img_name = f.split(os.sep)[-1]
    
    #preprocess the image
    img = img.resize((256,256))
    x = image.img_to_array(img)
    x = x.astype('float32') / 255 
    x1 = np.expand_dims(x, axis=0)
    
    #predict on the image
    pred = model_fpn.predict(x1)
    
    #reshape to the original size dimension
    test_img = np.reshape(pred, (256,256,1)) 

    #save the image to a mat file
    savemat("data/predictions/{}.mat".format(img_name[:-4]), 
            {'test_img': test_img}) 
    
    #write to a image file
    imageio.imwrite("data/predictions/{}.png".format(img_name[:-4]), 
                    test_img)

#%%
#save ground truth images as mat file

source = glob.glob('data/test_target/*.png')
source.sort()

for f in source:
    img = Image.open(f)
    img_name = f.split(os.sep)[-1]
    
    #preprocess the image
    x = image.img_to_array(img)
    x = x.astype('float32') / 255
    test_img = np.expand_dims(x, axis=0)
    test_img = np.reshape(x, (256,256,1))
    
    #save as mat file
    savemat("data/predictions/gt_target/{}.mat".format(img_name[:-4]), 
            {'test_img' : test_img})

#%%
'''
Compute the average value of correlation, intersection, chi-square
and bhattacharya distances for each image predicted by the models
and their respective ground truths, write the values to a CSV file and
then compute their average.
'''
#%%
def PSNR(img1, img2):
    mse = np.mean((img1-img2) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr

def mse(img1, img2):
    return np.square(np.subtract(img2.astype("float"),img1.astype("float"))).mean()

def mae(img1, img2):
    return np.absolute(np.subtract(img2, img1)).mean()

#%%
filenames1 = glob.glob("data/source_target/*.png") # ground truth
filenames1.sort()

filenames2 = glob.glob("data/predictions/model_fpn_ef0/img/*.png") 
filenames2.sort()

#%%

#open a new csv file
csv = open('data/predictions/model_fpn_ef0.csv','w') 
csv.write("filename,Correlation,Intersection,PSNR,MAE,SSIM,Chisquare,Bhattacharya\n")

for f1,f2 in zip(filenames1,filenames2):
    #read images
    img_g = cv2.imread(f1)
    img_p = cv2.imread(f2)
    img_name = f1.split(os.sep)[-1]
    
    # Convert the images to grayscale
    img_g1 = cv2.cvtColor(img_g, cv2.COLOR_BGR2GRAY)
    img_p1 = cv2.cvtColor(img_p, cv2.COLOR_BGR2GRAY)
    
    #calculate histograms    
    hist_g = cv2.calcHist([img_g],[0],None,[256],[0,256])
    hist_p = cv2.calcHist([img_p],[0],None,[256],[0,256])
    
    #normalize histograms
    hist_gn = cv2.normalize(hist_g, hist_g).flatten()
    hist_pn = cv2.normalize(hist_p, hist_p).flatten()
    
    #PSNR
    psnr = PSNR(img_g1, img_p1) 
    
    #MSE
    meanabserror = mae(img_g1, img_p1)
    
    #SSIM
    (SSIM, diff) = compare_ssim(img_g1, img_p1, full=True)
    diff = (diff * 255).astype("uint8")
    
    #measure distances
    corr = cv2.compareHist(hist_gn, hist_pn, 
                           cv2.HISTCMP_CORREL)
    intersection = cv2.compareHist(hist_gn, hist_pn, 
                                   cv2.HISTCMP_INTERSECT)
    chi_square = cv2.compareHist(hist_gn, hist_pn, 
                                 cv2.HISTCMP_CHISQR)
    bhattacharyya = cv2.compareHist(hist_gn, hist_pn, 
                                    cv2.HISTCMP_BHATTACHARYYA)
    csv.write(img_name+','+str(corr)+','+str(intersection)+','\
              +str(psnr)+','+str(meanabserror)+',' +str(SSIM)+','\
                  +str(chi_square)+','+str(bhattacharyya)+'\n')

csv.close()

#%%
# Matlab scripts
"""
The MS-SSIM measure is computed between the predicted and ground truth images
using the following matlab script 

clear
path = ["C:\\Users\\<your folder>"]; %this folder contains two separate folders one is gt with ground turth and the other with predicted data
files1 = dir([path filesep 'ae']); %prediction data folder
k=1;
for imgs = 1:length(files1)
    current = files1(imgs).name;
    if length(current)<5
        continue
    end
    if ~strcmp(current(end-2:end),'png')
        continue
    end
    imgt = imread([path filesep 'gt' filesep current]);
    impr = imread([path filesep 'ae' filesep current]);
    outputsheet{k,1} = current;
    result = multissim(imgt,impr);
    outputsheet{k,2} = result;
    k=k+1;
end

out=cell2table(outputsheet,'VariableNames',{'Image Name','MSSSIM'}); 
writetable(out,[path filesep 'MSSSIM_AE.csv']);
"""
#%%
# Majority voting script

'''
Folder structure: 
    predictions
     |_gt
     |_fpnresnet18
     |_unetresnet18
     |_fpnef0
    
Each folder will contain 27 mat files corresponding to 27 images in the test set. 
The majority voted image and its coresponding mat file will be saved in the parent folder. 
Block sizes attempted: 2, 3, 4, 8, 12, 16, 32, 64, 128, 256

% code: 
clearvars -except e1 e2 e3 gt 
% x_dir = 1:3:256; %block size 4
% y_dir = 1:3:256; %block size 4
% x_dir = 1:7:256; %block size 8
% y_dir = 1:7:256; %block size 8
% x_dir = 1:15:256; %block size 16
% y_dir = 1:15:256; %block size 16
% x_dir = 1:31:256; %block size 32
% y_dir = 1:31:256; %block size 32
% x_dir = 1:63:256; %block size 64
% y_dir = 1:63:256; %block size 64
% x_dir = 1:127:256; %block size 128
% y_dir = 1:127:256; %block size 128
% x_dir = 1:255:256; %block size 256
% y_dir = 1:255:256; %block size 256

linds={};
count=1;
for temp1 = 1:length(x_dir)-1
    for temp2 = 1:length(y_dir)-1
        [mx,my] = meshgrid(x_dir(temp1):x_dir(temp1+1),y_dir(temp2):y_dir(temp2+1));
        lind = sub2ind([256,256],mx(:),my(:));
        linds{count} = lind;
        count=count+1;
    end
end

%%
path = ["C:\\Users\\xx\\codes\\predictions'];
files1 = dir([path filesep 'gt']); 

for imgs = 3:length(files1)
    current = files1(imgs).name;
    load([path filesep 'gt' filesep current]); % ground truth
    gt = test_img;
    load([path filesep 'fpnresnet18' filesep current]); % ResNet18 based FPN model
    e1 = test_img;
    load([path filesep 'unetresnet18' filesep current]); %ResNet18 based U-Net model
    e2 = test_img;
    load([path filesep 'fpnef0' filesep current]); %EfficientNet-B0 based FPN model
    e3 = test_img;
    es1=[]; es2=[]; es3=[];
    for i=1:numel(linds)          
        

%         es1(i) = multissim(reshape(e1(linds{i}),[4 4]),reshape(gt(linds{i}),[4 4]),'NumScales',2); %4
%         es1(i) = multissim(reshape(e1(linds{i}),[8 8]),reshape(gt(linds{i}),[8 8]),'NumScales',3);  %8            
%         es1(i) = multissim(reshape(e1(linds{i}),[16 16]),reshape(gt(linds{i}),[16 16]),'NumScales',5); %16
%         es1(i) = multissim(reshape(e1(linds{i}),[32 32]),reshape(gt(linds{i}),[32 32])); %32
%         es1(i) = multissim(reshape(e1(linds{i}),[64 64]),reshape(gt(linds{i}),[64 64])); %64
%         es1(i) = multissim(reshape(e1(linds{i}),[128 128]),reshape(gt(linds{i}),[128 128])); %128
%         es1(i) = multissim(reshape(e1(linds{i}),[256 256]),reshape(gt(linds{i}),[256 256])); %256
        
 
%         es2(i) = multissim(reshape(e2(linds{i}),[4 4]),reshape(gt(linds{i}),[4 4]),'NumScales',2); %4
%         es2(i) = multissim(reshape(e2(linds{i}),[8 8]),reshape(gt(linds{i}),[8 8]),'NumScales',3);  %8        
%         es2(i) = multissim(reshape(e2(linds{i}),[16 16]),reshape(gt(linds{i}),[16 16]),'NumScales',5); %16
%         es2(i) = multissim(reshape(e2(linds{i}),[32 32]),reshape(gt(linds{i}),[32 32])); %32
%         es2(i) = multissim(reshape(e2(linds{i}),[64 64]),reshape(gt(linds{i}),[64 64])); %64
%         es2(i) = multissim(reshape(e2(linds{i}),[128 128]),reshape(gt(linds{i}),[128 128])); %128
%         es2(i) = multissim(reshape(e2(linds{i}),[256 256]),reshape(gt(linds{i}),[256 256])); %256
        
        
%         es3(i) = multissim(reshape(e3(linds{i}),[4 4]),reshape(gt(linds{i}),[4 4]),'NumScales',2); %4
%         es3(i) = multissim(reshape(e3(linds{i}),[8 8]),reshape(gt(linds{i}),[8 8]),'NumScales',3);  %8%           
%         es3(i) = multissim(reshape(e3(linds{i}),[16 16]),reshape(gt(linds{i}),[16 16]),'NumScales',5); %16
%         es3(i) = multissim(reshape(e3(linds{i}),[32 32]),reshape(gt(linds{i}),[32 32])); %32
%         es3(i) = multissim(reshape(e3(linds{i}),[64 64]),reshape(gt(linds{i}),[64 64])); %64
%         es3(i) = multissim(reshape(e3(linds{i}),[128 128]),reshape(gt(linds{i}),[128 128])); %128
%         es3(i) = multissim(reshape(e3(linds{i}),[256 256]),reshape(gt(linds{i}),[256 256])); %256
    end


    [~,majv] = max([es1;es2;es3]);

    models = ["e1","e2","e3"];

    FinalImg = zeros(256);
    for i=1:numel(linds)
        eval(['currentim = ' char(models(majv(i))) ';']); 
        FinalImg(linds{i}) = currentim(linds{i});
    end
    figure(1)
    imshow(FinalImg);

    save([path filesep current(1:end-4) '.mat'],'FinalImg')
    imwrite(FinalImg,[path filesep current(1:end-4) '.png']);
end

For the above predicted bone suppression images, compute their histograms and 
measure the distances from the ground truth as show before
'''
#%%
# classification code:
'''
We perform a two-class classification of the data (COVID-19/Normal)
We create 90/10 train/test splits and allocate 10% of the training
for validation with a fixed seed. We create bone and non bone suppressed images
and train the models individually on this data
'''
#%%
#import libraries

import time
import cv2
import pickle
import struct
import zlib
from tqdm import tqdm
import itertools
from itertools import cycle
from matplotlib import pyplot
from math import log
from scipy.stats import gaussian_kde
from scipy import special
from sklearn import metrics
from numpy import sqrt
from numpy import argmax
import numpy as np
from scipy import interp
from numpy import genfromtxt
import scikitplot as skplt
import pandas as pd
import math
from classification_models.keras import Classifiers
from keras.utils import to_categorical
from keras.activations import softmax
from tensorboard.plugins import projector
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error
from keras.preprocessing.image import load_img, img_to_array
from keras_efficientnets import EfficientNetB0
from keras.models import Model, load_model, Sequential
from keras.preprocessing.image import load_img
from keras.preprocessing import image
from keras.optimizers import SGD
from keras import backend as K
from keras import applications
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.layers import Flatten, ZeroPadding2D, Conv2D, Concatenate, MaxPooling2D, ZeroPadding2D, concatenate, Input, Reshape, GlobalAveragePooling2D, Dense, Dropout, Activation, BatchNormalization, Dropout, LSTM, ConvLSTM2D
from sklearn.metrics import roc_curve, auc,  precision_recall_curve, average_precision_score, matthews_corrcoef
from sklearn.metrics import f1_score, cohen_kappa_score, precision_score, recall_score, classification_report, log_loss, confusion_matrix, accuracy_score 
from sklearn.utils import class_weight
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import roc_auc_score, brier_score_loss
import seaborn as sns

#%% 

#get current working directory
print(os.getcwd())

#%%
# define custom function for confusion matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

#%% Load data

img_width, img_height = 256,256
train_data_dir = "data/bs_covid/train"
test_data_dir = "data/bs_covid/test"
epochs = 64 
batch_size = 16
num_classes = 2 
input_shape = (img_width, img_height, 3)
model_input = Input(shape=input_shape)
print(model_input) 

#%%
#define data generators
datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.1) #90/10 train val split

train_generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        seed=42,
        batch_size=batch_size, 
        class_mode='categorical', 
        subset = 'training')

validation_generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        seed=42,
        batch_size=batch_size, 
        class_mode='categorical', 
        subset = 'validation')

test_generator = datagen.flow_from_directory(
        test_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size, 
        class_mode='categorical', 
        shuffle = False)

#identify the number of samples
nb_train_samples = len(train_generator.filenames)
nb_validation_samples = len(validation_generator.filenames)
nb_test_samples = len(test_generator.filenames)

#check the class indices
print(train_generator.class_indices)
print(validation_generator.class_indices)
print(test_generator.class_indices)

#true labels
Y_val=validation_generator.classes
print(Y_val.shape)

Y_test=test_generator.classes
print(Y_test.shape)

Y_test1=to_categorical(Y_test, num_classes=num_classes, dtype='float32')
print(Y_test1.shape)

#%%
#compute class weights to penalize over represented classes
class_weights = dict(zip(np.unique(train_generator.classes), 
                         class_weight.compute_class_weight('balanced', 
                                                           np.unique(train_generator.classes), 
                                                           train_generator.classes))) 
print(class_weights)

#%%
'''
Load the best-performing bone suppression model,
truncate the encoder and add the classification layers.
This is performed to transfer CXR modality-specific knowledge
for a relevant CXR classification task and improve performance
'''
#%%
# best performing bone suppression model
modelfpn_ef0.load_weights("weights/model_1.fpn_ef0.h5")
modelfpn_ef0.summary()

#%%
#truncate the encoder
base_model_ef0=Model(inputs=modelfpn_ef0.input,
                        outputs=modelfpn_ef0.get_layer('block5c_add').output)
x = base_model_ef0.output 

# add classification layers
x = ZeroPadding2D()(x)
x = Conv2D(512, (3, 3), activation='relu')(x) 
x = GlobalAveragePooling2D()(x) 
logits = Dense(num_classes, 
                    activation='softmax', 
                    name='predictions')(x)
model_ef0 = Model(inputs=modelfpn_ef0.input, 
                    outputs=logits, 
                    name = 'eff0_bs_covid')

model_ef0.summary()

#%% train the model

sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)  
model_ef0.compile(optimizer=sgd, 
                    loss='categorical_crossentropy', 
                    metrics=['accuracy']) 
filepath = 'weights/classification/' + model_ef0.name + '.{epoch:02d}-{val_acc:.4f}.h5'
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', 
                             verbose=1, 
                             save_weights_only=False, 
                             save_best_only=True, 
                             mode='min', 
                             period=1)
earlyStopping = EarlyStopping(monitor='val_loss', 
                              patience=5, 
                              verbose=1, 
                              mode='min')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', 
                              factor=0.5, 
                              patience=5,
                              verbose=1,
                              mode='min', 
                              min_lr=0.00001)
callbacks_list = [checkpoint, earlyStopping, reduce_lr]
t=time.time()

#reset generators
train_generator.reset()
validation_generator.reset()

#train the model
model_ef0_history = model_ef0.fit_generator(train_generator, 
                                      steps_per_epoch=nb_train_samples // batch_size + 1,
                                      epochs=epochs, 
                                      validation_data=validation_generator,
                                      callbacks=callbacks_list, 
                                      class_weight = class_weights,
                                      validation_steps=nb_validation_samples // batch_size + 1, 
                                      verbose=1)

print('Training time: %s' % (time.time()-t))
    
#%% plot performance

N = 16 #epochs; change if early stopping
plt.style.use("ggplot")
plt.figure(figsize=(20,10), dpi=400)
plt.plot(np.arange(1, N+1), 
         model_ef0_history.history["loss"], 'orange', label="train_loss")
plt.plot(np.arange(1, N+1), 
         model_ef0_history.history["val_loss"], 'red', label="val_loss")
plt.plot(np.arange(1, N+1), 
         model_ef0_history.history["accuracy"], 'blue', label="train_acc")
plt.plot(np.arange(1, N+1), 
         model_ef0_history.history["val_accuracy"], 'green', label="val_acc")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower right")
plt.savefig("performance/fpnef0_bs_covid.png")

#%%
# Load model for evaluation: keep compile as False 
#since the model is used only for inference

model = load_model('weights/classification/eff0_bs_covid.h5', 
                          compile=False)
model.summary()

#%%
#Generate predictions on the test data
test_generator.reset() 
custom_y_pred = model.predict_generator(test_generator,
                                    nb_test_samples // batch_size + 1, 
                                    verbose=1)
custom_y_pred1_label = custom_y_pred.argmax(axis=-1)

#%%
#save predictions to a CSV file

predicted_class_indices=np.argmax(custom_y_pred,axis=1)
print(predicted_class_indices)

'''
map the predicted labels with their unique ids such 
as filenames to find out what you predicted for which image.
'''

labels = (test_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]

#save the results to a CSV file
filenames=test_generator.filenames
results=pd.DataFrame({"Filename":filenames,
                      "Predictions":predicted_class_indices,
                      "Labels":predictions})
results.to_csv("weights/classification/bs_covid_ef0.csv",index=False)

#%%
#evaluate classification performance
accuracy = accuracy_score(Y_test1.argmax(axis=-1),
                          custom_y_pred.argmax(axis=-1))
print('The accuracy of the model is: ', 
      accuracy)

prec = precision_score(Y_test1.argmax(axis=-1),
                       custom_y_pred.argmax(axis=-1), 
                       average='weighted') #options: micro, macro, weighted
print('The precision of the model is: ', prec)

rec = recall_score(Y_test1.argmax(axis=-1),
                   custom_y_pred.argmax(axis=-1), 
                   average='weighted')
print('The recall of the model is: ', rec)

f1 = f1_score(Y_test1.argmax(axis=-1),
              custom_y_pred.argmax(axis=-1), 
              average='weighted')
print('The f1-score of the model is: ', f1)

mat_coeff = matthews_corrcoef(Y_test1.argmax(axis=-1),
                              custom_y_pred.argmax(axis=-1))
print('The MCC of the model is: ', mat_coeff)

kappa = cohen_kappa_score(Y_test1.argmax(axis=-1),
                          custom_y_pred.argmax(axis=-1))
print('The cohen kappa score of the model is: ', kappa)

#%%
#print classification report
target_names = ['COVID-19','Normal']
print(classification_report(Y_test1.argmax(axis=-1),
                            custom_y_pred.argmax(axis=-1),
                            target_names=target_names, 
                            digits=4))

# Compute confusion matrix
cnf_matrix = confusion_matrix(Y_test1.argmax(axis=-1),
                              custom_y_pred.argmax(axis=-1))
                              
np.set_printoptions(precision=5)

x_axis_labels = ['COVID-19','Normal'] 
y_axis_labels = ['COVID-19','Normal'] 

plt.figure(figsize=(10,10), dpi=400)
sns.set(font_scale=2)
b = sns.heatmap(cnf_matrix, annot=True, square = True, 
            cbar=False, cmap='Greens', 
            annot_kws={'size': 50},
            fmt='g', 
            xticklabels=x_axis_labels, 
            yticklabels=y_axis_labels)

#%%

#plot AUROC curves
class_ = ['COVID-19','Normal']  
num_classes = len(class_)

lw = 2
# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(num_classes):
    fpr[i], tpr[i], thresholds = roc_curve(Y_test1[:, i], custom_y_pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
  
# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], thresholds = roc_curve(Y_test1.ravel(), 
                                          custom_y_pred.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])


#compute area under the ROC curve
auc_score_micro=roc_auc_score(Y_test1.ravel(),custom_y_pred.ravel())
print(auc_score_micro)

# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(num_classes):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= num_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
fig=plt.figure(figsize=(15,10), dpi=400)
ax = fig.add_subplot(1, 1, 1)
ax.set_facecolor('white')
major_ticks = np.arange(0.0, 1.1, 0.20) 
minor_ticks = np.arange(0.0, 1.1, 0.20)
ax.set_xticks(major_ticks)
ax.set_xticks(minor_ticks, minor=True)
ax.set_yticks(major_ticks)
ax.set_yticks(minor_ticks, minor=True)
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
colors = cycle(['red', 'blue', 'indigo'])
for i, color in zip(range(num_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=3, 
             label='{0} class (AUC = {1:0.4f})'
             .format(class_[i], roc_auc[i]))
plt.plot(fpr["micro"], tpr["micro"],
          label='micro-average (AUC = {0:0.4f})'
                ''.format(roc_auc["micro"]),
          color='green', linestyle='solid', linewidth=4) 
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('False Positive Rate', fontsize=20)
plt.ylabel('True Positive Rate', fontsize=20)
plt.legend(loc="lower right", prop={"size":20})
plt.show()


#%%
#compute precision-recall curves
class_ =  ['COVID-19','Normal']
num_classes = len(class_)

# For each class
precision = dict()
recall = dict()
average_precision = dict()
for i in range(num_classes):
    precision[i], recall[i], thresholds = precision_recall_curve(Y_test1[:, i],
                                                        custom_y_pred[:, i])
    average_precision[i] = average_precision_score(Y_test1[:, i], 
                                                   custom_y_pred[:, i])

# A "micro-average": quantifying score on all classes jointly
precision["micro"], recall["micro"], thresholds = precision_recall_curve(Y_test1.ravel(),
                                                                         custom_y_pred.ravel())
average_precision["micro"] = average_precision_score(Y_test1, 
                                                     custom_y_pred,
                                                     average="micro")

# convert to f score
fscore = (2 * precision["micro"] * recall["micro"]) / (precision["micro"] + recall["micro"])

print('Average precision score, micro-averaged over all classes: {0:0.4f}'
      .format(average_precision["micro"]))

# area under the PR curve
print("The area under the PR curve is", metrics.auc(recall["micro"], 
                                                    precision["micro"]))

# plot the PR curve for the model
no_skill = len(Y_test[Y_test==1]) / len(Y_test)
fig=plt.figure(figsize=(15,10), dpi=40)
ax = fig.add_subplot(1, 1, 1)
ax.set_facecolor('white')
major_ticks = np.arange(0.0, 1.1, 0.20) 
minor_ticks = np.arange(0.0, 1.1, 0.20)
ax.set_xticks(major_ticks)
ax.set_xticks(minor_ticks, minor=True)
colors = cycle(['red', 'blue', 'indigo'])
for i, color in zip(range(num_classes), colors):
    pyplot.plot(recall[i], precision[i], 
                color=color, lw=3, 
                label='Precision-recall for class {0} (area = {1:0.4f})'
                .format(class_[i], average_precision[i]))
    
pyplot.plot(recall["micro"], precision["micro"], 
            color='green', linestyle='solid', linewidth=4,
            label='micro-average Precision-recall (area = {0:0.4f})'
              ''.format(average_precision["micro"]))    
# axis labels
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Recall', fontsize=20)
plt.ylabel('Precision', fontsize=20)
plt.legend(loc="lower left", prop={"size":20})
plt.show()

#%%
# Perform CRM-based visualization

# custom code to increase DPI and save image
def writePNGwithdpi(im, filename, dpi=(72,72)):
   """Save the image as PNG with embedded dpi"""

   # Encode as PNG into memory
   retval, buffer = cv2.imencode(".png", im)
   s = buffer.tobytes()

   # Find start of IDAT chunk
   IDAToffset = s.find(b'IDAT') - 4
   pHYs = b'pHYs' + struct.pack('!IIc',int(dpi[0]/0.0254),int(dpi[1]/0.0254),b"\x01" ) 
   pHYs = struct.pack('!I',9) + pHYs + struct.pack('!I',zlib.crc32(pHYs))
   with open(filename, "wb") as out:
      out.write(buffer[0:IDAToffset])
      out.write(pHYs)
      out.write(buffer[IDAToffset:])

#%%
# generate Class Revlevance Map (CRM)

def Generate_CRM_2Class(thisModel, thisImg_path, Threshold):             
    try:
        # preprocess input image      
        width, height = thisModel.input.shape[1:3].as_list()
        original_img = cv2.imread(thisImg_path) 
        resized_original_image = cv2.resize(original_img, (width,height))        
    
        input_image = img_to_array(resized_original_image)
        input_image = np.array(input_image, dtype="float") /255.0       
        input_image = input_image.reshape((1,) + input_image.shape)
    
        class_weights = thisModel.layers[-1].get_weights()[0]
    
        get_output = K.function([thisModel.layers[0].input], [thisModel.layers[-3].output, #deepest convolutional layer
                                 thisModel.layers[-1].output])
        [conv_outputs, predictions] = get_output([input_image])
        conv_outputs = conv_outputs[ 0, :, :, :]     
        final_output = predictions   
        
        wf0 = np.zeros(dtype = np.float32, shape = conv_outputs.shape[0:2])    
        wf1 = np.zeros(dtype = np.float32, shape = conv_outputs.shape[0:2])    
    
        for i, w in enumerate(class_weights[:, 0]):     
            wf0 += w * conv_outputs[:, :, i]           
        S0 = np.sum(wf0)           # score at node 0 in the final output layer
        for i, w in enumerate(class_weights[:, 1]):     
            wf1 += w * conv_outputs[:, :, i]             
        S1 = np.sum(wf1)           # score at node 1 in the final output layer
    
        #Calculate incremental MSE
        iMSE0 = np.zeros(dtype = np.float32, shape = conv_outputs.shape[0:2])
        iMSE1 = np.zeros(dtype = np.float32, shape = conv_outputs.shape[0:2])
    
        row, col = wf0.shape
        for x in range (row):
                for y in range (col):
                        tmp0 = np.array(wf0)
                        tmp0[x,y] = 0.                   # remove activation at the spatial location (x,y)
                        iMSE0[x,y] = (S0 - np.sum(tmp0)) ** 2
    
                        tmp1 = np.array(wf1)
                        tmp1[x,y] = 0.                  
                        iMSE1[x,y] = (S1 - np.sum(tmp1)) ** 2
         
      
        Final_crm = iMSE0 + iMSE1       # consider both positive and negative contribution
    
        Final_crm /= np.max(Final_crm)    # normalize
        resized_Final_crm = cv2.resize(Final_crm, (height, width)) # upscaling to original image size

        The_CRM = np.array(resized_Final_crm)
        The_CRM[np.where(resized_Final_crm < Threshold)] = 0.  # clean-up (remove data below threshold)

        return[resized_original_image, final_output, resized_Final_crm, The_CRM]
    except Exception as e:
        raise Exception('Error from Generate_CRM_2Class(): ' + str(e))

#%%
# load model and predict
model = load_model('weights/classification/eff0_bs_covid.h5',
                                compile=False)
model.summary()

#%%
img_path = 'data/covid/image1.png'
img = load_img(img_path)

#preprocess the image
x = img_to_array(img)
x = np.expand_dims(x, axis=0)
x /= 255

#predict on the image
preds = model.predict(x)[0]
print(preds)

#%%
#COMPUTE CRM AND SAVE IMAGE
InImage1, OutScores1, aCRM_Img1, tCRM_Img1 = Generate_CRM_2Class(model,
                                                                 img_path, 0.2) 
plt.figure() 
plt.imshow(InImage1)

#measure heatmap and threshold to remove noise
aHeatmap = cv2.applyColorMap(np.uint8(255*aCRM_Img1), cv2.COLORMAP_JET)
aHeatmap[np.where(aCRM_Img1 < 0.2)] = 0 #vary this value and see the effect of noise
superimposed_img = aHeatmap * 0.4 + InImage1 #0.4 here is a heatmap intensity factor

#if we have to increse the DPI and write to disk
writePNGwithdpi(superimposed_img, "crm_covid_image1.png", (300,300))   

#%% 

