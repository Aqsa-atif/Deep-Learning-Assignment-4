# Deep-Learning-Assignment-4
## Question no.1:
Create a CNN network with the CIFAR-10 dataset using your registration no. * 7.8 as reference. 
## Solution:
Registration No. : 400517
400517 * 7.8 = 3124032.6 = 3124033
Rounding each to the nearest odd: 3135133
Let's break down the code and discuss the choices made for loss function, optimizer, and other hyperparameters.
# Data Splitting:
The dataset is split into training, validation, and test sets. 47,000 samples are used for training, 3,000 for validation, and the rest for testing.
# Model Architecture:
The model is a convolutional neural network (CNN) built using the TensorFlow Functional API. It consists of convolutional layers, batch normalization, max-pooling, and dense layers.
# Compilation:
The model is compiled using the Sparse Categorical Crossentropy loss, the Adam optimizer, and accuracy as the metric.
# Loss Function (`SparseCategoricalCrossentropy):
The choice of `SparseCategoricalCrossentropy suggests that the task is a classification problem with integer labels (as opposed to one-hot encoded labels). This loss function is suitable for multi-class classification problems.
# Optimizer (Adam):
The Adam optimizer is a popular choice for training neural networks. It adapts the learning rates of each parameter individually, providing a good balance between convergence speed and generalization. The learning rate is set to 0.001.
# Training:
The model is trained using the training data, and the validation data is used to monitor the model's performance during training.
# Batch Size (64):
The number of samples used in each iteration during training. Smaller batch sizes consume less memory but might slow down training. Larger batch sizes can speed up training but may require more memory.
# Epochs (30):
The number of times the entire training dataset is passed forward and backward through the neural network. More epochs allow the model to learn from the data for a longer duration.
# Evaluation:
The model is evaluated on the test set to assess its generalization performance.



## Considering the new number as the number of filters in each layer
Here I’m downloading the CIFAR-10 dataset from the internet and splitting it into train and validation sets.
Here is a Convolutional Neural Network (CNN) model for image classification. It has 10 layers, including 7 convolutional layers, 2 pooling layers, and 1 fully connected layer. The model takes a 32x32x3 image as input and outputs a 10-dimensional vector, where each element represents the probability of the image belonging to one of the 10 classes in the CIFAR-10 dataset.
Compiles the model for training with SparseCategoricalCrossentropy loss, Adam optimizer, and accuracy metric. The learning rate is set to 0.001.
The plot of training and validation loss curves shows that the training loss decreases rapidly at first, but then begins to plateau. The validation loss also decreases at first, but then starts to increase again, indicating that the model is overfitting the training data.
The training and validation loss curves show a rapid decrease in training loss, plateauing, and a subsequent increase in validation loss, indicating the model is overfitting the training data. The validation loss curve increases after 10 epochs, indicating the model is learning specific patterns rather than generalizing to unseen data. To prevent overfitting, techniques like reducing epochs, using regularization techniques, and increasing training data size can be used.
The training accuracy chart indicates that the model's training accuracy is higher than its validation accuracy, indicating overfitting. The model learns to fit training data well but struggles to generalize to new data. This results in worse performance when evaluated on the validation set, a new set of data. To avoid overfitting, techniques like regularization, data augmentation, and early stopping can be used.
## Predictions: 
Several factors may impact the model's performance. Its simple architecture, characterized by a limited number of convolutional layers and filters, may hinder the learning of complex features and subtle patterns. The model's training period of 30 epochs might not be sufficient for full convergence, and the small batch size of 64 could lead to training instability and overfitting. Additionally, the provided blurry and low-resolution image poses challenges for the model to focus on specific objects. To enhance performance, it is recommended to employ a more complex architecture, increase the number of filters in convolutional layers, extend the training duration, use a larger batch size, evaluate on the validation set, enhance image resolution, and opt for a model specifically designed for image classification.
## Summary:
The model uses convolutional filters to extract features from input images, with different filter sizes to capture different levels of detail. The maxPooling2D layer downsamples feature maps to reduce data size and control overfitting. Batch normalization stabilizes the training process and improves generalization. The relu activation function adds non-linearity to the network. The dense layer is used for fully connected layers, where each neuron is connected to all neurons in the previous layer. The flatten layer flattens the input tensor from a 3D to 1D vector before feeding it to the fully connected layers. The final layer, Softmax, outputs probabilities for each class.
Convolutional filters are effective for image classification tasks, and batch normalization stabilizes the learning process and improves convergence speed. Non-linear activation functions like relu are necessary for learning complex decision boundaries and improving model performance. Fully connected layers combine features extracted by convolutional layers to learn abstract representations for classification. The flatten layer prepares feature maps for fully connected layers, expecting a 1D vector input. Softmax outputs a probability distribution over the 10 classes in CIFAR-10, allowing the model to make predictions. The model is trained for 30 epochs with a batch size of 64.
Considering the new number as the size of filters in each layer
This type of architecture is much more better and performs very well on validation set than the last as we need a large number of filters to be applied to our image in order to extract meaningful features
Here I’m downloading the CIFAR-10 dataset from the internet and splitting it into train and validation sets.
The CNN has 10 output classes, which suggests that it is used for a multi-class classification task. The CNN uses 2 convolutional layers, 2 pooling layers, and 2 fully connected layers. The first convolutional layer has 32 filters, the second convolutional layer has 64 filters, the first fully connected layer has 64 units, and the second fully connected layer has 10 units. The CNN uses the softmax activation function in the output layer to produce class probabilities.
The image shows the training and validation loss and accuracy of a machine learning model over 30 epochs. The model is trained on a dataset of e-commerce product reviews to predict the rating of a product based on the customer review.
The model achieves a high training accuracy of 98.68% and a validation accuracy of 83.07%. This suggests that the model is able to learn the patterns in the training data and generalize well to new data.
The plot shows the training and validation loss curves of a model, indicating its performance on training data and unseen data. The ideal scenario is for both curves to decrease over time, with the validation loss curve remaining lower than the training loss curve, indicating the model is learning without overfitting to the training data. However, the training loss curve decreases rapidly at first, plateauing around epoch 10, and the validation loss curve also decreases but increases again after epoch 10, suggesting the model is starting to overfit to the training data. To prevent overfitting, one can stop training the model before it overfits or use regularization techniques like L1 or L2 regularization.
The plot shows training and validation accuracy curves for a machine learning model. The training accuracy increases rapidly at first, but then starts to plateau, while the validation accuracy peaks around epoch 15 and then starts to decrease. This suggests that the model is starting to overfit to the training data after epoch 15
We are not using 1x1 instead using 3x3 because it can lead to loss of spatial information, increased parameters, limited representational power, diminished gradient flow, and redundant results. Therefore, it's best to use larger filters for tasks requiring spatial information, complex non-linearities, or model complexity and training time.
## Predictions:
Several factors may impact the model's performance. Its simple architecture, characterized by a limited number of convolutional layers and filters, may hinder the learning of complex features and subtle patterns. The model's training period of 30 epochs might not be sufficient for full convergence, and the small batch size of 64 could lead to training instability and overfitting. Additionally, the provided blurry and low-resolution image poses challenges for the model to focus on specific objects. To enhance performance, it is recommended to employ a more complex architecture, increase the number of filters in convolutional layers, extend the training duration, use a larger batch size, evaluate on the validation set, enhance image resolution, and opt for a model specifically designed for image classification.
 
