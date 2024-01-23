import os
import pickle

import numpy as np
import streamlit as st
import tensorflow
from PIL import Image
from numpy.linalg import norm
from sklearn.neighbors import NearestNeighbors
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.preprocessing import image

st.title('Fashion Recommender System')

#OUR DATA SET from where recommendation will be given
feature_list = pickle.load(open('embedding.pkl','rb'))
filenames = pickle.load(open('filename.pkl','rb'))

# Model creation
model = ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))
model.trainable = False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

#  STEP's - we have to avail file upload facility to users and save them
            # file upload -> save
            # load file   ->

# Pending ==  extract feature from those image (create function)
#feature extraction of uploaded image
def extract_feature(img_path,model):
     img = image.load_img(img_path,target_size=(224,224))
     img_arrey = image.img_to_array(img)
     expanded_img_arrey = np.expand_dims(img_arrey,axis=0)
     preprocessed_img = preprocess_input(expanded_img_arrey)
     result = model.predict(preprocessed_img).flatten()
     normalized_result = result / norm(result)

     return normalized_result

def recommend(features,feature_list):
    neighbors = NearestNeighbors(n_neighbors=6, metric='euclidean', algorithm='brute')  # AWS me ANNOY use karenge
    neighbors.fit(feature_list)  # OUR WHOLE DATA EXTRACTED FEATURE IS IN THIS LIST (VARIABLE OF PKL)

    distance, indices = neighbors.kneighbors([features])

    return indices
              # recommend (upload se nearest distance wali dataset se image)

              # show recommendation

def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join('uploads',uploaded_file.name),'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except:
        return 0


uploaded_file = st.file_uploader("Upload an image")
if uploaded_file is not None:
    if save_uploaded_file(uploaded_file):
        #Display file
        display_image = Image.open(uploaded_file)
        st.image(display_image)
        #Feature extraction
        feature = extract_feature(os.path.join("uploads",uploaded_file.name),model)
        #st.text(feature)  see whether feature are extracted or not properly
        #recoomedation
        indices = recommend(feature,feature_list)
        #show
        col1,col2,col3,col4,col5 = st.columns(5)

        with col1:
            image = Image.open(filenames[indices[0][0]])  # Replace 'path_to_your_image.jpg' with your image file
            st.image(image, width=170)
            #st.image(filenames[indices[0][0]])
        with col2:
            image =Image.open(filenames[indices[0][1]])
            st.image(image, width=170)
            #st.image(filenames[indices[0][1]])
        with col3:
            image = Image.open(filenames[indices[0][2]])
            st.image(image,width=170)
            #st.image(filenames[indices[0][2]])
        with col4:
            image = Image.open(filenames[indices[0][3]])
            st.image(image,width=170)
            #st.image(filenames[indices[0][3]])
        with col5:
            image = Image.open(filenames[indices[0][4]])
            st.image(image,width=190)
            #st.image(filenames[indices[0][4]])
    else:
        st.header('Please Upload Again \n Some error happened in upload ')





