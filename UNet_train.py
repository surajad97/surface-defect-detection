import matplotlib.pyplot as plt
import os
import cv2
import matplotlib.image as mpimg
import shutil
import xmltodict
import numpy as np 

from utils import get_mask_seg_ellipse, load_coordinates, load_images_masks

IMAGE_CHANNELS = 1

data_dir1 = "./Data/archive/DAGM_dataset/Class1/Train/"
data_dir2 = "./Data/archive/DAGM_dataset/Class2/Train/"
data_dir3 = "./Data/archive/DAGM_dataset/Class3/Train/"
data_dir4 = "./Data/archive/DAGM_dataset/Class4/Train/"
data_dir5 = "./Data/archive/DAGM_dataset/Class5/Train/"
data_dir6 = "./Data/archive/DAGM_dataset/Class6/Train/"

X1, y1 = load_images_masks(data_dir1, img_type='PNG', img_format='gray', resize=(512, 512))
X2, y2 = load_images_masks(data_dir2, img_type='PNG', img_format='gray', resize=(512, 512))
X3, y3 = load_images_masks(data_dir3, img_type='PNG', img_format='gray', resize=(512, 512))
X4, y4 = load_images_masks(data_dir4, img_type='PNG', img_format='gray', resize=(512, 512))
X5, y5 = load_images_masks(data_dir5, img_type='PNG', img_format='gray', resize=(512, 512))
X6, y6 = load_images_masks(data_dir6, img_type='PNG', img_format='gray', resize=(512, 512))

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Lambda, Conv2DTranspose, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras import backend as K
from scipy.ndimage.measurements import label
import time

def small_unet():
    inputs = Input((512,512, 1))
    #inputs_norm = Lambda(lambda x: x/127.5 - 1.)
    conv1 = Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(16, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(32, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv5)

    up6 = concatenate([Conv2DTranspose(64, kernel_size=(
        2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    conv6 = Conv2D(128, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([Conv2DTranspose(32, kernel_size=(
        2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = Conv2D(64, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([Conv2DTranspose(16, kernel_size=(
        2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(32, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([Conv2DTranspose(8, kernel_size=(
        2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(16, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(16, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=inputs, outputs=conv10)

    return model


model1 = small_unet()
model2 = small_unet()
model3 = small_unet()
model4 = small_unet()
model5 = small_unet()
model6 = small_unet()


def smooth_dice_coeff(smooth=1.):

    smooth = float(smooth)

    # IOU or dice coeff calculation
    def IOU_calc(y_true, y_pred):
            y_true_f = K.flatten(y_true)
            y_pred_f = K.flatten(y_pred)
            intersection = K.sum(y_true_f * y_pred_f)

            return 2*(intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

    def IOU_calc_loss(y_true, y_pred):
        return -IOU_calc(y_true, y_pred)
    return IOU_calc, IOU_calc_loss

IOU_calc, IOU_calc_loss = smooth_dice_coeff(0.00001)


model1.compile(optimizer= Adam(learning_rate=0.0001),loss= IOU_calc_loss )
model2.compile(optimizer= Adam(learning_rate=0.0001),loss= IOU_calc_loss )
model3.compile(optimizer= Adam(learning_rate=0.0001),loss= IOU_calc_loss )
model4.compile(optimizer= Adam(learning_rate=0.0001),loss= IOU_calc_loss )
model5.compile(optimizer= Adam(learning_rate=0.0001),loss= IOU_calc_loss )
model6.compile(optimizer= Adam(learning_rate=0.0001),loss= IOU_calc_loss )

import tensorflow as tf
import sklearn
from sklearn.model_selection import train_test_split
x_train1, x_valid1, y_train1, y_valid1 = train_test_split(X1, y1, test_size=0.16666,  shuffle= True)
x_train2, x_valid2, y_train2, y_valid2 = train_test_split(X2, y2, test_size=0.16666,  shuffle= True)
x_train3, x_valid3, y_train3, y_valid3 = train_test_split(X3, y3, test_size=0.16666,  shuffle= True)
x_train4, x_valid4, y_train4, y_valid4 = train_test_split(X4, y4, test_size=0.16666,  shuffle= True)
x_train5, x_valid5, y_train5, y_valid5 = train_test_split(X5, y5, test_size=0.16666,  shuffle= True)
x_train6, x_valid6, y_train6, y_valid6 = train_test_split(X6, y6, test_size=0.16666,  shuffle= True)


callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

history = model1.fit(x_train1, y_train1, batch_size=10, epochs=25,callbacks=[callback],validation_data = (x_valid1,y_valid1), verbose=1)
model1.save('./model/class1.h5')
history = model2.fit(x_train2, y_train2, batch_size=10, epochs=25,callbacks=[callback],validation_data = (x_valid2,y_valid2), verbose=1)
model2.save('./model/class2.h5')
history = model3.fit(x_train3, y_train3, batch_size=10, epochs=25,validation_data = (x_valid3,y_valid3),callbacks=[callback], verbose=1)
model3.save('./model/class3.h5')
history = model4.fit(x_train4, y_train4, batch_size=10, epochs=25,callbacks=[callback],validation_data = (x_valid4,y_valid4), verbose=1)
model4.save('./model/class4.h5')
history = model5.fit(x_train5, y_train5, batch_size=10, epochs=25,callbacks=[callback],validation_data = (x_valid5,y_valid5), verbose=1)
model5.save('./model/class5.h5')
history = model6.fit(x_train6, y_train6, batch_size=10, epochs=25,callbacks=[callback],validation_data = (x_valid6,y_valid6), verbose=1)
model6.save('./model/class6.h5')