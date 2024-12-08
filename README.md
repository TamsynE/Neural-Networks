# AI: Neural-Networks

## Overview
An accumulation of independent programming projects for AI, including experimentation.

### 1. Feed-forward Neural Neural Network

MNIST, the Modified National Institute of Standards and Technology, is a sub-set database of handwritten digits derived from the larger NIST database. It is used popularly to train machine learning models to recognize handwritten digits. 

The MNIST database has 70,000 examples, which is split into a training set and a test set. The first 60 000 examples are included in the training set, whilst the last 10 000 examples are part of the test set. Each example has an image (28x28 pixel greyscale) and a label of digits ranging from 0 to 9 to be used for training and testing of a neural network, for example, a feed-forward network or multi-layer perceptron, to classify the digits.

##### Experimental Overview
We will be titrating the training set for training (100, 500, and 60000 samples) and testing a basic neural network with 2 hidden layers, 64 neurons per hidden layer, 16 epochs, and a batch size of 32. This titration allows us to test whether our neural network is sound and produces the expected accuracy.

To ensure we are not reusing the same model for each experiment, we will create, train, and evaluate each model with the designated variables adjusted accordingly each time. Due to resource exhaustion, we will need to choose parameters to fit our hardware. During initial testing of the number of neurons per layer on accuracy of the model, 1024 neurons per hidden layer in my model could not compute (memory exhaustion).

For ease of use, code-conciseness, and efficiency, I have transformed the process we follow in creating, training, and evaluating the basic neural network into reusable functions where we can adjust the parameters accordingly based on the experiment.

- Experiment 1: Neurons | How does the number of neurons per hidden layer of the neural network affect the accuracy and training time?

- Experiment 2: Layers | How does the number of hidden layers in the neural network affect the accuracy and training time?

- Experiment 3: Epochs | How does the number of epochs used when training the neural network affect the accuracy and training time?

- Experiment 4: Batch Size | How does the batch size used when training the neural network affect the accuracy and training time?

- Experiment 5: Activation Functions | How does changing the activation functions of the hidden layers of the model affect the accuracy and training time?

### 2. Convolutional Neural Network with CIFAR-10
This project builds and experiments on a classifier for a multiclass image dataset, CIFAR-10 - specifically a convolutional neural network (CNN) that uses convolutional and fully connected layers. 

- Experiment 1: Comparing ANN and CNN
  
- Experiment 2: Line Search
    - Experiment 2A: Kernel Size
    - Experiment 2B: Max Pooling Size
    - Experiment 2C: Convolutional Filters Per Layer
    - Experiment 2D: Number of Convolutional Layers
 
- Experiment 3: Examining The First Layer

### 3. Natural Language Processing: Fine-tuning a Pretrained Model
This project involves fine-tuning a pretrained model (BART) with a CNN/DailyMail dataset from Hugging Face for the purpose of text summarization (information extraction). This dataset includes samples (full news articles) and labels (summaries of the articles) that can be used for fine-tuning. Because fine-tuning was computationally intensive with such a large dataset for my CPU, I fine-tuned the BART generative model with a smaller subset of my dataset for proof of concept. However, this could be done with the full dataset.

