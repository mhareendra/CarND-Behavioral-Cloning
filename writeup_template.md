# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/PreProcess.PNG "Pre processing pipeline"
[image2]: ./examples/track2_fail.jpg "Track2 Failure Spot"

[image3]: ./examples/left_track1.jpg "Track1 Left Image"
[image4]: ./examples/right_track1.jpg "Track1 Right Image"
[image5]: ./examples/center_track1.jpg "Track1 Center Image"

[image6]: ./examples/left_track2.jpg "Track2 Left Image"
[image7]: ./examples/right_track2.jpg "Track2 Right Image"
[image8]: ./examples/center_track2.jpg "Track2 Center Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
Main submission:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results
The main submission uses the sample dataset provided by Udacity.
The supplementary submission includes the above files for a secondary training data set (collected manually) of track1 and track2. 

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model for track 1, and it contains comments to explain how the code works.

There is another model_track2.py file that contains code to train a model to drive autonomously around track2. 

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model follows the architecture of the NVIDIA end-to-end deep learning model.
The network architecture consists of multiple layers, including a lambda normalization layer, a cropping layer, 5 convolutional layers
and 3 fully connected layers.

The lambda layer normalizes the image and the cropping layer removes 70 rows from the top and 25 rows from the bottom of the image.

The filter depths for the convolutional layers are 24, 36, 48, 64 and 64 respectively. A Filter size of 5x5 is used for the first 3 layers and 3x3 is used for the next 2 layers.  These layers use RELU activation to introduce non-linearity. 

#### 2. Attempts to reduce overfitting in the model

The model was trained and validated on very large data sets to ensure that the model was not overfitting. The shuffle parameter was also set  to true.
The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

Two data sets of training data were attempted for track 1. The first data set is sourced from the sample training data provided by Udacity, the second data set is obtained by manually driving around the track. Later sections provide more information about the training data.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

Track 1:
Coming up with the working solution involved multiple steps. The first attempt involved using the LeNet architecture to train the model using images obtained by driving approximately 60% of the track. The images were normalized and cropped to remove unnecesary regions. 
The trained model drove well for a few seconds before driving off the road on the right side. The followup attempt involved collecting training data for the entire track, which drove well for a few more seconds but did not drastically improve the results. This clearly indicated that the training model wasn't powerful enough. Using images from the left and right cameras in the training process helped to an extent but not a lot.

The next attempt involved using NVIDIA's end-to-end deep learning model to solve this problem. The trained model immediately produced better results but did not make it well past the first bend. The vehicle, however, drove close to the center of the road for a major portion of its drive. This indicated that it needed more data to be able to better learn the driving patterns. Udacity's sample training data set was used since it contained enough data for multiple laps around the track. The validation loss on the 3 epochs was around: 0.04, 0.03 and 0.025 respectively. The trained model performed very well on the track and kept the car near the middle of the road throughout.

To really test the architecture, new training data was manually collected and used to train a model. This performed really well too but not as well as the sample training data, in terms of how smooth the drive was. This could be attributed to the fact that the manual data itself was a bit jittery which was reflected in how the model drove.

Track 2:

The same network architecture was chosen and training data was collected for track 2. The trained model immediately failed on the first appearance of a shadow.
The training process was changed to add a pre-processing step that used adaptive histogram equaliztion on the luminance component of the image. The equalized image was used to generate an rgb image that was then fed as input to the network. The same pre-processing step was added to drive.py. 

![alt text][image1]

The model worked very well for about 80% of the track after which the car ran off the road at a very dark curve.

![alt text][image2]
#### 2. Creation of the Training Set & Training Process
Track 1:
First data set is directly obtained from Udacity's sample training data. 
The second data set contains images obtained by driving around the track 5 times, yielding about 19,308 images from all the cameras. The vehicle was deliberately driven near the center of the road to force the model to do the same while driving autonomously.

![alt text][image3]     ![alt text][image4]      ![alt text][image5]

Track 2:
A similar approach was used for capturing data (4 laps) for track 2 producing about 11,800 images.

![alt text][image6]     ![alt text][image7]      ![alt text][image8]

The trainin data involved images from the center, left and right cameras. The data set was augmented to simulate clockwise driving by flipping ever image. Because of the large dataset, a generator was used to avoid holding all the images in memory.

I finally randomly shuffled the data set and put 80% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3. I used an adam optimizer so that manually training the learning rate wasn't necessary.
