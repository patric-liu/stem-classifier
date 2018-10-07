import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.utils import to_categorical
import os
import pickle
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.applications import ResNet50

vgg_conv = ResNet50(weights='imagenet',
                 include_top=False,
                 input_shape=(224, 224, 3))

vgg_conv.summary() # Display Resnet50 model structure

# Location of training data and validation data
path = os.getcwd() + '/clean-dataset'
train_dir = path + '/train'
validation_dir = path + '/validation'

nTrain = 550 # number of images to train on each epoch
nVal = 120 # numer of images to evaluate model on
datagen = ImageDataGenerator() 
batch_size = 10 # minibatch size
num_labels = 2 # number of categories for the data

# Find outputs for training data
train_features = np.zeros(shape=(nTrain, 7, 7, 2048)) 
train_labels = np.zeros(shape=(nTrain, num_labels)) 
train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True)
    
i = 0
for inputs_batch, labels_batch in train_generator:
    if i % 5 == 0:
        print("starting features for batch {}".format(i))
    if i * batch_size >= nTrain:
        break
    
    features_batch = vgg_conv.predict(inputs_batch)
    train_features[i * batch_size: (i + 1) * batch_size] = features_batch
    train_labels[i * batch_size: (i + 1) * batch_size] = labels_batch
    i += 1
train_features = np.reshape(train_features, (nTrain, 7 * 7 * 2048))

# find outputs for validation data
validation_features = np.zeros(shape=(nVal, 7, 7, 2048))
validation_labels = np.zeros(shape=(nVal, num_labels))
validation_generator = datagen.flow_from_directory(
    validation_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False)
i = 0
for inputs_batch, labels_batch in validation_generator:
    if i % 5 == 0:
        print("starting features for batch {}".format(i))
    features_batch = vgg_conv.predict(inputs_batch)
    validation_features[i * batch_size: (i + 1) * batch_size] = features_batch
    validation_labels[i * batch_size: (i + 1) * batch_size] = labels_batch
    i += 1
    if i * batch_size >= nVal:
        break
validation_features = np.reshape(validation_features, (nVal, 7 * 7 * 2048))

''' Now we add in a fully connected layer and output layer to complete the model
If we take the output of the convolution layers of the ResNet as the input to these
new layers, we are just creating a simple 1 hidden layer network
'''
from keras import models, layers, optimizers, regularizers

# Create the model
model = models.Sequential()
model.add(layers.Dense(512, activation='relu', input_dim=7 * 7 * 2048, kernel_regularizer=regularizers.l2(0.01)))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(num_labels, activation='softmax'))

# Choose the optimizer
model.compile(optimizer=optimizers.RMSprop(lr=2e-4, decay=0.07),
              loss='categorical_crossentropy',
              metrics=['acc'])
              
# Train the model
history = model.fit(train_features,
                    train_labels,
                    epochs=25,
                    batch_size=batch_size,
                    validation_data=(validation_features, validation_labels))
                    
ground_truth = validation_generator.classes

# Getting the mapping from class index to class label
label2index = validation_generator.class_indices
idx2label = dict((v, k) for k, v in label2index.items())

# Evaluate model on evaluation dataset
predictions = model.predict_classes(validation_features)
prob = model.predict(validation_features)
errors = (ground_truth[:len(predictions)] != predictions).sum()

# Number of errors
print("No of errors = {}/{}".format(errors, nVal), "Accuracy = {}".format(1-errors/nVal))

''' GET ERROR INDICES, this lets us locate specific errors and better understand its performance'''

# returns indices of where classification was incorrect
matches = (ground_truth[:len(predictions)] != predictions).tolist()
error_indices = [i for i, x in enumerate(matches) if x]
# get class indices
class_indices = [0]+[i+1 for i, (a,b) in enumerate(zip(ground_truth[1:], ground_truth[:-1])) if a != b]
# initialize output array, first dimension spans classes
error_indices_class = [[] for i in range(len(class_indices))]
# Assign errors to appropriate classes and shift values to match folder values
for error_index in error_indices:
    for i, class_index in reversed(list(enumerate(class_indices))):
        if error_index >= class_index:
            error_indices_class[i].append(error_index - class_index)
            break
            
print(error_indices_class, 'error indices')

# save architecture and weights and misc info
from keras.models import model_from_json
save_path = os.getcwd() + '/model_weights/model_v3.h5'
model.save_weights(save_path)
with open(os.getcwd() + '/model_weights/model_architecture3.json', 'w') as f:
    f.write(model.to_json())
with open('info.pkl', 'wb') as output:
    pickle.dump(idx2label, output, -1)
