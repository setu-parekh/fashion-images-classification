# Fashion Images Classification
Implementing a Convolutional Neural Network model for classification of fashion images

## Summary
* [Introduction & General Information](#introduction--general-information)
* [Objectives](#objectives)
* [Data Used](#data-used)
* [Approach & Methodology](#approach--methodology)
* [Performance Evaluation](#performance-evaluation)
* [Conclusion](#conclusion)
* [Run Locally](#run-locally)


## Introduction & General Information
**Convolutional Neural Networks (CNN)**

- CNN are widely used for image classification. It is a process in which we provide input to the model in the form of images and obtain the image class or the probability that the input image belongs to a particular class.
- Humans can look at an image and recognize what it is, but it is not same for machines. Each image is a series of pixel values arranged in particular order. We have to represent the image in a manner a machine can understand. This can be achieved using a CNN Model.
- If we have a black and white image, the pixels are arranged in the form of 2D array. Each pixel has a value between 0 - 255.
  - 0 means completely white
  - 255 means completely black
  - Grayscale exists if the number lies between 0 and 255
- If we have a colored image, then the pixels are arranged in the form of 3D array. This 3D array has blue, green and red layers.
  - Each color has pixel values ranging from 0 - 255
  - We can find the exact color by combining the pixel values of each of the 3 layers.

**Keras and TensorFlow**
- In deep learning or machine learning, we have datasets which are mostly multi-dimensional, where each dimension represents different features of the data.
- Tensor is the way of representing such multi-dimensional data. In this project, we are dealing with fashion object images. There can be many aspects to an image such as shape, edges, boundaries etc. In order to classify these images correctly as different objects, the convolutional network will have to learn to discriminate these features. These features are incorporated by TensorFlow
- Keras is a high-level library which is built on top of TensorFlow. It provides a scikit-learn type API for building Neural Networks. Hence, it simplifies the process of building neural networks as we don't have to worry about mathematical aspects of TensorFlow algebra.

## Objectives
- Building a convolutional neural networks model using Keras to classify the images of fashion articles into 10 different class items.
- Evaluating the performance of the model using classification report(Precision, Recall, F1-Score and Accuracy) and Confusion matrix.


## Data Used
(Source: https://github.com/zalandoresearch/fashion-mnist/tree/master/data/fashion)

- We are using Fashion MNIST dataset for the purpose of this project. This dataset consists of images related to fashion articles. There are 60,000 images for training and 10,000 images for testing purpose.
  - There are total 784 pixels (28 x 28) in each image.
  - Each pixel has an integer value between the range 0 - 255.
  - Higher integer value represents darker pixel and lower integer value represents lighter pixels.

- Each image is associated with one of the 10 classes. These classes are namely:
  - 0 - T_shirt/Top
  - 1 - Trouser
  - 2 - Pullover
  - 3 - Dress
  - 4 - Coat
  - 5 - Sandal
  - 6 - Shirt
  - 7 - Sneaker
  - 8 - Bag
  - 9 - Ankle boot

- ![Visualizing images from the input dataset](https://github.com/setu-parekh/fashion-images-classification/blob/main/images/input_data_visualization.png)

## Approach & Methodology
- Loading the fashion MNIST dataset and converting the train/test data into pandas dataframe. These dataframes will have class label as the first column followed by 784 columns for pixel values.
- Converting these dataframes into numpy array as it is the acceptable form of input for TensorFlow and Keras.
- Pre-processing train and test numpy arrays in order to make them ready to be fed into the CNN model.
- Visualizing few of the images from the train dataset to get better insight of the data being used.
- Building and training convolutional neural network model based on the training dataset.
- Evaluating the performance of class predictions using test dataset.

## Performance Evaluation
![Confusion Matrix](https://github.com/setu-parekh/fashion-images-classification/blob/main/images/confusion_matrix.png)

- The blue colored sections represents number of labels correctly classified by the model and light colored sections represents the number of labels mis-classified by the model.

- It can be inferred that the model lacks precision while predicting 'Shirt'. Out of 1000 instances, it is correctly classified for only 650 times whereas mis-classified as 'T-Shirt/Top' for 158 times.

- The model performs very well in predicting Trouser and Bag.
  - Trouser is correctly classified for 966 times out of 1000 instances.
  - Bag is correctly classified for 966 times out of 1000 instances.

## Conclusion
- Model was able to classify the images with 89% accuracy.
- Trouser and Bag were the most accurately predicted objects.
- Shirt was the most mis-classified object as T-Shirt/Top.

## Run Locally
- Make sure Python 3 is installed. Reference to install: [Download and Install Python 3](https://www.python.org/downloads/)
- Clone the project: `git clone https://github.com/setu-parekh/fashion-images-classification.git`
- Route to the cloned project: `cd fashion-images-Classification`
- Download the data zip files from the following links and save the files in the route `cd fashion-images-Classification`

  | Name  | Download |
  | --- | --- |
  | Training set images | [train-images-idx3-ubyte.gz](http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz) |
  | Training set labels | [train-labels-idx1-ubyte.gz](http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz) |
  | Test set images | [t10k-images-idx3-ubyte.gz](http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz) |
  | Test set labels | [t10k-labels-idx1-ubyte.gz](http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz) |


- Install necessary packages: `pip install -r requirements.txt`
- Run Jupyter Notebook: `jupyter notebook`
- Select the notebook to open: `fashion_image_classification.ipynb`



