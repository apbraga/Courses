# **Traffic Sign Recognition**
Alex Braga alexbraga101@gmail.com
Autonomous Vehicle Nanodegree | Udacity

https://github.com/apbraga/Traffic-Sign-Classifier/blob/master/Traffic_Sign_Classifier.ipynb

## Writeup
---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/1.png "Visualization"
[image2]: ./examples/2.png "Grayscaling"
[image3]: ./examples/3.png "Random Noise"
[image4]: ./examples/4.png "Traffic Sign 1"
[image5]: ./examples/5.jpeg "Traffic Sign 2"
[image6]: ./examples/6.jpeg "Traffic Sign 3"
[image7]: ./examples/7.jpeg "Traffic Sign 4"
[image8]: ./examples/8.jpeg "Traffic Sign 5"
[image9]: ./examples/9.jpeg "Traffic Sign 5"
---


### Data Set Summary & Exploration

#### 1. Summary

Using numpy library the following information was gathered from the dataset.

* Number of training examples = 34799
* Number of testing examples = 4410
* Number of testing examples = 34799
* Image data shape = (32, 32, 3)
* Number of classes = 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the labels are distributed among the datasets for training, validation and testing.

* Training Data Distribution

![alt text][image1]

* Validation Data Distribution

![alt text][image2]

* Test Data Distribution

![alt text][image3]

The distribution is pretty much similar between the datasets, but not among the labels.
In this sense, we may find it difficult to get the Model to predict the traffic signs with small quantity of examples.

### Design and Test a Model Architecture

#### 1. Image Process Pipeline

As a first step, the image is converted to Grayscale to reduce the effect of color grading among images and the effect of different cameras representing color independently.

The second step is to Equalize each image according to its histogram to reduce the effect of light condition in each image.

The third step is to apply a smoothing filter to reduce the effect of noise in the performance of the Model. After testing several filters, the one that provided the best performance for the model was the Laplacian.

As sanity check we reshape the image to a 32x32x1 image, this will also become handy when we bring images from others sources to be tested.

Finally the 0-255 Grayscale image is transformed into a 0.- 1. matrix.(This provided better model accuracy than using -1 - 1 scale)

``` python
  def preprocess(img):
    out= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    out= cv2.equalizeHist(out)
    out= cv2.Laplacian(out, cv2.CV_64F)
    #out= cv2.bilateralFilter(out, 9, 75, 75)
    #out= cv2.medianBlur(out,5)
    out= out.reshape(32,32,1)
    out= out/255
    return
```


#### 2. Model Architecture

My final model consisted of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x1 Image 							|
| Convolution 1x1     	| 1x1 stride, same padding, 64 Filters 	|
| Convolution 5x5					|	1x1 stride, same padding, 128 Filters 											|
| Max pooling	3x3      	| 2x2 stride 				|
| Convolution 3x3	    | 1x1 stride, same padding,  256 Filters   						|
| Max pooling 3x3		| 2x2 stride       									|
| Convolution 3x3				| 1x1 stride, same padding, 512 Filters         									|
|	Max pooling					|	2x2 stride											|
|	Dropout					|		Rate = 0.2										|
|	Flatten					|				-								|
|	Softmax					|			43 Classes									|

#### 3. Training Parameters

To train the model, I used an the following parameters:

* Loss function: Categorical Crossentropy
* Optimizer: AdaDelta
* Metrics: Categorical Accuracy and Top k=5 Categorical Accuracy
* Batch size: 64
* epochs: 5


#### 4. Model Evaluation

My final model results were without data augmentation:
* training set accuracy of 1.0
* validation set accuracy of 0.9920
* test set accuracy of 0.9739

My final model results were with data augmentation:
* training set accuracy of 0.9995
* validation set accuracy of 0.9920
* test set accuracy of 0.9754


I started with a Model based on MicronNet (https://arxiv.org/pdf/1804.00497v3.pdf), and from it the first improvements were related to the image processing, deciding the image filter to be applied, finally leading to the Laplacian.

After Tweaking training epochs, the model was still achieving accuracy lower the 85%.
From this moment on I started to modify the model architecture, the first big change was to drop the fully connected layers from MicronNet, In my case they were causing the model to underfit, after removing the FC layers the accuracy rose to 0.92.

The further improvements to reach the final performance level were the optimizer change from Adam to AdaDelta and the filter size on each conv layer, and finally the dropout layer to avoid overfitting.

### Test a Model on New Images

#### 1. External Source images

Here are five German traffic signs that I found on the web:

![alt text][image5] ![alt text][image6] ![alt text][image7]
![alt text][image8] ![alt text][image9]

Reason per image:
* Stop: High resolution and tilted Sign
* Bumpy: Complex symbol and tilted sign
* Priority: Color related sign, to check performance using grayscale
* Caution: Common sign external shape (triangle) and busy background
* Intersection Priority: Complex symbol

#### 2. Prediction on external images

Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| Stop Sign      		| Stop sign   									|
| Bumpy road     			| Bumpy road 										|
| Priority road					| Priority road											|
| General caution	      		| General caution					 				|
| Right-of-way at the next intersection		| Right-of-way at the next intersection      							|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set approximately 99%.

#### 3. Top K prediction

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .55         			| 14 Stop  									|
| .07     				| 2	Speed limit (50km/h)									|
| .06					| 13	Yield										|
| .03	      			| 8	No passing				 				|
| .02				    | 5  Speed limit (80km/h)    							|


For the second image ...

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .54         			| 22 Bumpy road  									|
| .13     				| 25 Road work										|
| .13					| 20 Dangerous curve to the right											|
| .06	      			| 26 Traffic signals					 				|
| .03				    | 13 Yield    							|


For the third image ...

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .98         			| 12 Priority road  									|
| .009     				| 32 End of all speed and passing limits										|
| .0009					| 9	No passing										|
| .0003      			| 15 No vehicles					 				|
| .0002			    | 22  Bumpy road    							|


For the fourth image ...

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .96         			| 18 General caution  									|
| .003     				| 24 Road narrows on the right										|
| .00005					| 40 Roundabout mandatory											|
| .00005	      			| 11 Right-of-way at the next intersection					 				|
| .00004				    | 27 Pedestrians     							|


For the last image ...

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .99         			| 11  Right-of-way at the next intersection 									|
| .00008     				| 27 	Pedestrians									|
| .00006					| 30	Beware of ice/snow										|
| .00001	      			| 31	Wild animals crossing				 				|
| .000005				    | 24  Road narrows on the right   							|
