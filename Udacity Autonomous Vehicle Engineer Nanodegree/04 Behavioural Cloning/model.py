#Dependencies are imported
import csv
import numpy as np
import os
import cv2
import math
import pandas as pd
import time
import argparse
import json
import sys
import json
import keras
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Dense, Dropout, Flatten, Lambda, Activation, ELU, Conv2D, Cropping2D
from keras.optimizers import Adam
import matplotlib.image as mpimg
from sklearn.model_selection import train_test_split

#Load data from csv file to pandas dataframe
def load_dataset():
    data_df = pd.read_csv(os.path.join('data', 'driving_log.csv'))
    #drop 90% of data with steering = 0 to balance the dataset
    data_df = data_df[data_df['steering'] != 0].append(data_df[data_df['steering'] == 0].sample(frac=0.1))
    #separated the dataframe data into input and output    
    X = data_df[['center', 'left', 'right']].values
    y = data_df['steering'].values
    return X, y

#Mirror image for augmentation
def flip(image, measurement):
    #apply flip image
    image = np.fliplr(image)
    #reverse steering actuation
    measurement = -measurement
    return image, measurement

#increase dataset with augmentation
def augment_data(X,y):
    #initialize array
    X_aug=[]
    y_aug=[]
    #add right, center and left data into a single array, adjust right and left angles by 0.2    
    for i in range(len(y)):
        X_aug.append(X[i][0])
        X_aug.append(X[i][1])
        X_aug.append(X[i][2])
        y_aug.append(y[i] + 0.0)   
        y_aug.append(y[i] + 0.2)  
        y_aug.append(y[i] - 0.2)
    # initialize array
    x_out = []
    y_out = []
    #shuffle image index before loading
    idx = np.arange(len(y_aug))
    idx = np.random.shuffle(idx)
    for i in range(len(y_aug)):
        #get each image file path
        file_path = str('/home/workspace/CarND-Behavioral-Cloning-P3/data/'+X_aug[i].strip())
        #read image as rgb
        x_i= cv2.cvtColor(cv2.imread(file_path),cv2.COLOR_BGR2RGB)
        # add image and respective steering anlge to an array
        x_out.append(x_i)
        y_out.append(y_aug[i])
        #flip each image and add to an array
        X_i_flip, y_i_flip = flip(x_i, y_aug[i])
        x_out.append(X_i_flip)
        y_out.append(y_i_flip)
    return np.array(x_out), np.array(y_out)

#Create network architecture
def model_build():
    #Model Definition based on Project 3 network
    model = keras.Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((60,25),(0,0)))) 
    model.add(Conv2D(16, (5, 5), strides=(2,2), activation="relu"))
    model.add(Dropout(0.2))
    model.add(Conv2D(32, (5, 5), strides=(2, 2), activation="relu"))
    model.add(Dropout(0.2))
    model.add(Conv2D(64, (5, 5), strides=(2, 2), activation="relu"))
    model.add(Dropout(0.2))
    model.add(Conv2D(128, (3, 3), activation="relu"))
    model.add(Dropout(0.2))
    #model.add(keras.layers.MaxPooling2D(pool_size=(3,3),strides= 2))
    model.add(Conv2D(256, (3, 3), activation="relu"))
    model.add(Flatten())
    #model.add(Dense(100))
    #model.add(Dense(50))
    #model.add(Dense(10))
    model.add(Dense(1))
    model.summary()
    exit()
    model.compile(loss='mse', optimizer='adam')
    return model     

def train_model(model, n_epochs, batch_size, X_train, y_train ):        
    
    #Fit model, (fit_generator was the first trial, but due to performance issues it was changed to fit)
    model.fit(X_train, y_train, batch_size=batch_size, epochs=n_epochs, validation_split=0.1)

#project pipeline
def main():
    #Load image path and steering angles
    X, y = load_dataset()
    #Load images
    X_train, y_train = augment_data(X,y)
    #create neural net
    model = model_build()
    n_epochs = 3
    batch_size = 64
    #train model
    train_model(model, n_epochs, batch_size, X_train, y_train) 
    #exported trained model
    model.save('model.h5')   

main()