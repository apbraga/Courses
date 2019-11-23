# **Behavioral Cloning** 


**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode (Speed changed from 9 to 20)
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a 5 layer convolution neural network with 5x5 and 3x3 filter sizes and depths between 32 and 256 (model.py lines 74) and a single neuron as output .

The model includes RELU layers to introduce nonlinearity (code line 74), and the data is normalized in the model using a Keras lambda layer (code line 77).

Input is also cropped within the model to remove the unwanted top and bottom portion of the image( model.py line 78). 

| Layer         		|     Output shape	        					| Number of trainable parameter |
|:---------------------:|:---------------------------------------------:|:--------------|
| lambda_1 (Lambda)        		| (None, 160, 320, 3) 							| 0|
| cropping2d_1 (Cropping2D)     	| (None, 36, 158, 16) 	| 0|
| conv2d_1 (Conv2D) 				|	(None, 36, 158, 16) 											| 1216|
| dropout_1 (Dropout))     	| (None, 36, 158, 16) 				| 0|
| conv2d_2 (Conv2D) 	    | (None, 16, 77, 32)   						| 12832|
| dropout_1 (Dropout)	| (None, 16, 77, 32)        									|0 |
| conv2d_3 (Conv2D) 			| (None, 6, 37, 64)         									| 51264|
|	dropout_1 (Dropout)				|	(None, 6, 37, 64)											| 0|
|	conv2d_4 (Conv2D) 				|		(None, 4, 35, 128)										| 73856|
|	dropout_4 (Dropout)				|		(None, 4, 35, 128)							| 0|
|	conv2d_5 (Conv2D)				|			(None, 2, 33, 256)									|295168 |
|	Flatten					|				(None, 16896)								| 0|
|	Softmax					|			(None, 1)									| 16897|

Total params: 451,233

Trainable params: 451,233

Non-trainable params: 0

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 80, 82, 84, 86). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 102). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 96).

#### 4. Appropriate training data

The dataset provided by udacity was used for the training. Data augmentation was used to scale up to 20k training samples.

Images for the left and right camera were used combined with a drift on the steering angle, and each image was flipped in the horizontal axis and its steering angle inverted. 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to build a network general enough to handle parts of the track that had few samples in training, so a great effort to reduce overfitting was done, as result dropout layers gave the best performance, as well as removing the fully connected layer within the output and the convolutional layers.

My first step was to use a convolution neural network model similar to the Nvidia, but during training it would overfit rather quickly, leading to the collection of more data.

Then i borrow the network developed in the last project for traffic sign recognition, and on trial and error it was decided to remove the max pooling layers, remove one conv layer and add more dropouts.

This resulted in a good generality after only 3 epochs.

For all the trials it was used Adam optimization and MSE as loss function.
The final step was to run the simulator to see how well the car was driving around track one. 

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 74) consisted of a convolution neural network with the following layers and layer sizes:
| Layer         		|     Output shape	        					| Number of trainable parameter |
|:---------------------:|:---------------------------------------------:|:--------------|
| lambda_1 (Lambda)        		| (None, 160, 320, 3) 							| 0|
| cropping2d_1 (Cropping2D)     	| (None, 36, 158, 16) 	| 0|
| conv2d_1 (Conv2D) 				|	(None, 36, 158, 16) 											| 1216|
| dropout_1 (Dropout))     	| (None, 36, 158, 16) 				| 0|
| conv2d_2 (Conv2D) 	    | (None, 16, 77, 32)   						| 12832|
| dropout_1 (Dropout)	| (None, 16, 77, 32)        									|0 |
| conv2d_3 (Conv2D) 			| (None, 6, 37, 64)         									| 51264|
|	dropout_1 (Dropout)				|	(None, 6, 37, 64)											| 0|
|	conv2d_4 (Conv2D) 				|		(None, 4, 35, 128)										| 73856|
|	dropout_4 (Dropout)				|		(None, 4, 35, 128)							| 0|
|	conv2d_5 (Conv2D)				|			(None, 2, 33, 256)									|295168 |
|	Flatten					|				(None, 16896)								| 0|
|	Softmax					|			(None, 1)									| 16897|

#### 3. Creation of the Training Set & Training Process

The dataset provided by udacity was used for the training. Data augmentation was used to scale up to 20k training samples.

Images for the left and right camera were used combined with a drift on the steering angle, and each image was flipped in the horizontal axis and its steering angle inverted. 


One key point that gave a great leap on performance was to drop around 90% of the data points with steering angle =0, the dataset is very unbalanced so this helps the model to learn more about curve and not overfitting on straight condition.

Check out the result in this youtube link: https://youtu.be/FdVlMMWQjcw
