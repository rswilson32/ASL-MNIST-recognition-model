# ASL-MNIST-recognition-model

## The Project
This model classifies images of the American Sign Language (ASL) alphabet. According to the National Institute of Health, over 860,000 North Americans use ASL either as a primary or secondary language as of 2006, and we hope that our model can contribute to making communication more accessible for individuals who sign. This was my final project for my machine learning class at UCLA. While it was a group project, we worked on creating three separate convolutional neural networks (CNN) to compare different architectures and their performances on our data. The following code and opencv program were my contributions. Since we were developing an image recognition model, convolutional neural networks were the natural choice as they are flexible, and we need a model that is invariant under translation of the input. We believe that using convolutional neural networks will maximize our overall image recognition accuracy. 

## The Data
The training and testing data is from the Sign Language MNIST data, which contains 27,455 training images and 7,712 testing images. Each training and test case represents a label (0-25) as a one-to-one map for each alphabetic letter A-Z (and no cases for 9=J or 25=Z because they involve motions). Each 28x28 pixel image is represented as a vector with grayscale values between 0-255. 
The original hand gesture image data represented multiple users repeating the gesture against different backgrounds. To create new data, “an image pipeline was used based on ImageMagick and included cropping to hands-only, gray-scaling, resizing, and then creating at least 50+ variations to enlarge the quantity”. 

<img src="american_sign_language.PNG" alt="ASL Alphabet" width="700"/>

## Model
The model is built in PyTorch with four convolution layers and a fully connected layer at the end. I also employed max pooling to down-sample our feature maps. At the end of our pipeline, the network includes two fully connected layers followed by batch normalization and dropout. I used dropout in the fully connected layer instead of in the convolutions. The first convolutional layer processes a single input channel, gradually increasing the number of channels in the next layers. ReLU activation functions are applied throughout the network. A dropout rate of 0.2 is also applied at the end. Through hyperparameter tuning, this CNN achieved 98.2% accuracy with a batch size of 64, Adam optimizer, a learning rate of 0.001, and 8 epochs. 

## Camera Application
We also implemented a real-time ASL alphabet camera recognition system using the opencv library designed for computer vision and my trained model’s weights. The program recognizes images in a given region of interest (ROI) which we designate with a green square. The frame is then captured,  set to gray-scale, and blurred. We are then able to classify the hand gesture using our previously trained CNN. I'm working on somehow adding the opencv application to my website because I think it would be neat.

