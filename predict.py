import numpy as np
import matplotlib.pyplot as plt
import keras
import os
from keras.preprocessing import image
from keras.applications import ResNet50
from keras.layers import Input, Flatten
from keras.models import model_from_json, Model
from keras.utils import plot_model
from keras import models, layers

'''
Transfer Learning Tester

Loads model previously trained with transfer learning and runs custom images 
through the model to aid in model evaluation and debugging

'''

# build model
res_conv = ResNet50(weights='imagenet',
                    include_top=False,
                    input_shape=(224, 224, 3))
                    
# FC layer/model reconstruction from JSON
with open(os.getcwd() + '/model_weights/model_architecture.0.json', 'r') as f:
    fc = model_from_json(f.read())
# load weights into the FC model
fc.load_weights(os.getcwd() + '/model_weights/model_0.0.h5')

input_image = Input(shape=(224,224,3))
features = res_conv(input_image)
predictions = fc(Flatten()(features))
full_model = Model(inputs = input_image, outputs = predictions)

img_path = os.getcwd() + '/testing_files/test.png'


img = image.load_img(path = img_path, grayscale=False,
                     target_size=(224, 224, 3))

img = image.img_to_array(img) # coverts from PIL format to numpy array
#img = img/255 
img_4dim = img[np.newaxis, :, :, :]

predictions = full_model.predict(img_4dim)[0].tolist()

import pickle
with open('randata.pkl', 'rb') as input: #info
    idx2label = pickle.load(input)

classname = idx2label[predictions.index(max(predictions))]
    
print(predictions, "class:", classname)

plt.imshow(img/255)
#plt.title("predicted class:", classname)
plt.show()
