import streamlit as st
import os
from PIL import Image
import numpy as np
import pickle
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm
from funcs import *
import streamlit.components.v1 as components

feature_list = np.array(pickle.load(open('embeddings.pkl','rb')))
filenames = pickle.load(open('filenames.pkl','rb'))

model = ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))
model.trainable = False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

st.title('Fashion Recommender System')

def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join('uploads',uploaded_file.name),'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except:
        return 0

def feature_extraction(img_path,model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)

    return normalized_result

def recommend(features,feature_list):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)

    distances, indices = neighbors.kneighbors([features])

    return indices

# steps
# file upload -> save
uploaded_file = st.file_uploader("Choose an image")
if uploaded_file is not None:
    if save_uploaded_file(uploaded_file):
        # display the file
        display_image = Image.open(uploaded_file)
        st.image(display_image)
        # feature extract
        features = feature_extraction(os.path.join("uploads",uploaded_file.name),model)
        #st.text(features)
        # recommendation
        indices = recommend(features,feature_list)
        # Show recommended images
        st.header("Recommended Images")

        # Display recommended images along with the query image
        st.subheader("Query Image:")
        st.image(display_image, caption='Query Image', use_column_width=True)

        # Create columns to display images
        col1, col2, col3, col4, col5 = st.columns(5)
        # Display recommended images in separate columns
        with col1:
            st.image(filenames[indices[0][0]], width=150, caption='Recommended Image 1')
        with col2:
            st.image(filenames[indices[0][1]], width=150, caption='Recommended Image 2')
        with col3:
            st.image(filenames[indices[0][2]], width=150, caption='Recommended Image 3')
        with col4:
            st.image(filenames[indices[0][3]], width=150, caption='Recommended Image 4')
        with col5:
            st.image(filenames[indices[0][4]], width=150, caption='Recommended Image 5')

        # Calculate a form of accuracy
        # You need a set of similar images or ground truth labels for this
        similar_images = []  # List of similar images to the query image
        num_relevant = 0
        for i in range(5):  # Check the top 5 recommended images
            if filenames[indices[0][i]] in similar_images:
                num_relevant += 1
        accuracy = num_relevant / 5  # Total number of relevant recommendations out of 5
        st.write("Accuracy:", accuracy)
    else:
        st.header("Some error occurred in file upload")
