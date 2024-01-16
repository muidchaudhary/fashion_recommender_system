

import numpy as np
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from numpy.linalg import norm
import os
from tqdm import tqdm
import pickle

model = ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))
model.trainable = False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

#print(model.summary())

def extract_feature(img_path,model):
                                                                # Defining image size
img = image.load_img(img_path,target_size=(224,224))
    img_arrey = image.img_to_array(img)                          #converting img to array
    expanded_img_array = np.expand_dims(img_arrey,axis=0)        #expanding bcz preprocessing of keras takes img in batch not single img so expan and pass it in batch
    preprocessed_img = preprocess_input(expanded_img_array)      # process single img as a batch with only having 1 img
    result = model.predict(preprocessed_img).flatten()           # img is in 2D(1,2048) convert 1D (2048) remove batch passed RGB 2 BGR with zero centered
    normalized_result = result/norm(result)

    return normalized_result    # image daalo norm result nikalo

# Joining images & file together in filename variable
filenames = []
for file in os.listdir('images'):
    filenames.append(os.path.join('images',file))

#Creating feature extracter with it extract all feature from images given
# Getting image and our model together (resnet preprocessed model)
# Giving file name and getting  feature

#print(len(filenames))
#print(filenames[0:5])

# Here we'll fit out all 3 things together 1- model, 2- Extraction feature function, 3- file (our dataset)

feature_list = []
for file in tqdm(filenames):
    feature_list.append(extract_feature(file,model))

#print(np.array(feature_list).shape)


#Dumping this features in pickle file so that we can use and compare anywhere

pickle.dump(feature_list,open("embedding.pkl","wb"))
pickle.dump(filenames,open("filename.pkl","wb"))

