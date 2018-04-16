# -*- coding: utf-8 -*-

import os
import csv

samples = []
with open('Data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

import cv2
import numpy as np
import sklearn

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            #read in center, left and right camera images
            #Convert these images to RGB format since they are read in as BGR
            for batch_sample in batch_samples:
                name = 'Data/IMG/'+batch_sample[0].split('/')[-1]
                
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])
                center_image = BGR_to_RGB(center_image)
                images.append(center_image)
                angles.append(center_angle)
                
                #apply correction factor to both left and right camera images
                #since the simulator only collects steering angle with respect to the center camera
                left_correction = 0.25 # this is a parameter to tune
                right_correction = 0.25 # this is a parameter to tune
                left_angle = center_angle + left_correction
                right_angle = center_angle - right_correction
                left_image = cv2.imread('Data/IMG/'+batch_sample[1].split('/')[-1])
                left_image = BGR_to_RGB(left_image)
                angles.append(float(left_angle))
                
                right_image = cv2.imread('Data/IMG/'+batch_sample[2].split('/')[-1])
                right_image = BGR_to_RGB(right_image)
                angles.append(float(right_angle))
                images.append(left_image)
                images.append(right_image)
            
            #Augment our dataset by flipping the above images horizontally
            #This simulates driving in the clockwise direction
            augmented_images = []
            augmented_angles =[]
            for image,angle in zip(images,angles):
                augmented_images.append(image)
                augmented_angles.append(angle)
                augmented_images.append(cv2.flip(image,1))
                augmented_angles.append(angle*-1.0) 
                
            X_train = np.array(augmented_images)
            y_train = np.array(augmented_angles)
            yield sklearn.utils.shuffle(X_train, y_train)

def BGR_to_RGB(image):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return rgb_image
    

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=4)
validation_generator = generator(validation_samples, batch_size=4)

ch, row, col = 3, 160, 320 

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda,Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D

model = Sequential()
#normalization layer
# Preprocess incoming data, centered around zero  
model.add(Lambda(lambda x: x/255.0 - 0.5,
        input_shape=(row, col, ch)
        ))
#crop the top 70 rows of the image (contains non-road regions) and bottom 25 rows (hood of the car)
model.add(Cropping2D(cropping=((70,25), (0,0))))

#5 convolutional layers
model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Flatten())

#fully-connected layers
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, samples_per_epoch=
            len(train_samples), validation_data=validation_generator,
            nb_val_samples=len(validation_samples), nb_epoch=3, shuffle=True)

model.save('model.h5')

print('Done!')
