import streamlit as st
import os
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import GlobalMaxPooling2D, Dense, Dropout
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from numpy.linalg import norm
import pickle
from sklearn.neighbors import NearestNeighbors
from tensorflow.keras.optimizers import Adam  # Import Adam optimizer

st.title('Fashion Recommender System')

# Construct absolute paths for files
embedding_path = os.path.join(os.getcwd(), 'embedding.pkl')
filename_path = os.path.join(os.getcwd(), 'filename.pkl')

# Load data using absolute paths
feature_list = pickle.load(open(embedding_path, 'rb'))
filenames = pickle.load(open(filename_path, 'rb'))

# Model creation
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

# Create a functional model with additional layers
model = tf.keras.Sequential([
    base_model,
    GlobalMaxPooling2D(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(256, activation='relu'),
    Dropout(0.3),
    Dense(128, activation='relu')
])

# Extract features using the functional model
extractor = tf.keras.Model(inputs=model.inputs, outputs=model.layers[-1].output)

# Extract features for all images
all_features = []
for img_path in filenames:
    img = Image.open(img_path).convert("RGB").resize((224, 224))
    img_array = np.array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    features = extractor.predict(preprocessed_img).flatten()
    normalized_features = features / norm(features)
    all_features.append(normalized_features)

# Convert the list of features to NumPy array
all_features = np.array(all_features)

# Fit NearestNeighbors model
neighbors = NearestNeighbors(n_neighbors=6, metric='euclidean', algorithm='brute')
neighbors.fit(all_features)

def extract_feature(img_path, model):
    img = Image.open(img_path).convert("RGB").resize((224, 224))
    img_array = np.array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)
    return normalized_result

def recommend(features, neighbors_model):
    distance, indices = neighbors_model.kneighbors([features])
    return indices

def save_uploaded_file(uploaded_file):
    try:
        upload_dir = 'uploads'
        os.makedirs(upload_dir, exist_ok=True)
        with open(os.path.join(upload_dir, uploaded_file.name), 'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except:
        return 0

uploaded_file = st.file_uploader("Upload an image")
if uploaded_file is not None:
    if save_uploaded_file(uploaded_file):
        display_image = Image.open(uploaded_file).convert("RGB").resize((224, 224))
        st.image(display_image)
        feature = extract_feature(os.path.join("uploads", uploaded_file.name), base_model)
        indices = recommend(feature, neighbors)

        num_cols = 5
        columns = st.columns(num_cols)
        for i in range(min(num_cols, len(indices[0]))):
            with columns[i]:
                image = Image.open(filenames[indices[0][i]])
                st.image(image, width=170 if i < num_cols - 1 else 190)
    else:
        st.header('Please Upload Again \n Some error happened in upload')
