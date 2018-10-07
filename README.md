# stem-classifier
Simple binary classifier to reduce Type II error rate of another algorithm by over 95%

- transfer-learning.py 

Trains a one layer neural network, using a pretrained (ImageNet) ResNet50's convolutional layers as a general purpose feature extractor. Pruning or retraining an uninitialized feature extractor may be a future step, since the imagenet dataset is not extremely similar to the domain we are interested in, which is crops


- predict.py

Reloads a network trained with transfer-learning.py and predicts a user specified image 
