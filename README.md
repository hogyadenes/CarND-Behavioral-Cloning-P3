**Behavioral Cloning** 

In this project for udacity I developed a convolutional neural network in Keras to drive a car in a simulator. The network is based on [NVIDIA's approach](https://arxiv.org/pdf/1604.07316) of End-to-End Learning for Self Driving Cars. This is basically a deep learning only solution without any conventional feature extraction (such as image processing, filtering etc) - apart from normalizing the input data. My solution runs in the default udacity setup, which uses the center camera images of the simulated car to drive it safely on the road. For the detailed description of my solution please read the following sections.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* README.md summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model is based on the NVIDIA solution for self driving cars. This is a multilayer convolutional neural network built in Keras consisting of one cropping, one normalizing, 5 convolutional and 4 fully connected layers with a single output (which is the driving angle).

The cropping layer cuts off the top 55 and bottom 25 pixels which are usually don't contain valuable information thus effectively reducing the input size to the half (from 160x320x3 to 80x320x3). 

The model uses RELU as activation function to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer.

The detailed description of the model is the following:
|Layer	|Description|
|Input	|160x320x3 RGB image A|
|Cropping	|Crop top 55 pixels and bottom 25 pixels; output shape = 80x320x3|
|Normalization	|value/255 - 0.5|
|Convolution 5x5	|5x5 kernel, 2x2 stride, 24 output channels, output shape = 38x158x24|
|RELU	|	|
|Convolution 5x5	|5x5 kernel, 2x2 stride, 36 output channels, output shape = 17x77x36|
|RELU	|	|
|Convolution 5x5	|5x5 kernel, 2x2 stride, 48 output channels, output shape = 7x37x48|
|RELU	|	|
|Convolution 5x5	|3x3 kernel, 1x1 stride, 64 output channels, output shape = 5x35x64|
|RELU	|	|
|Convolution 5x5	|3x3 kernel, 1x1 stride, 64 output channels, output shape = 3x33x64|
|RELU	|	|
|Flatten|	Input 3x33x64, output 6336|
|Fully connected|	Input 6336, output 100|
|Fully connected|	Input 100, output 50|
|Fully connected|	Input 50, output 10|
|Fully connected|	Input 10, output 1 (labels)|

The network has 770619 total training parameters.

####2. Attempts to reduce overfitting in the model

I experimented with dropout but the results did not improve and the network itself did not show signs of overfitting either (the results on the training and validation set correlated reasonably). The reason for this could be that the input is very noisy and nondeterministic (especially when using keyboard input during training). As I created quite a lot of extra training data the noise itself could prevent overfitting.

The model was trained and validated on five different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track. The results are quite good on the hilly track either.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road, driving in the opposite direction and on both tracks. 

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to create a relatively simple approach but my goal was to achieve at least a partial solution on the more difficult track as well.

My first step was to use a convolution neural network model similar to the LeNet. I thought this model might be appropriate because I used it successfully in the previous (traffic sign identification) project. However, as it did not give perfect results even on the difficult parts of the first track, I moved on to a more powerful network, which NVIDIA proposed as a self driving car solution.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
