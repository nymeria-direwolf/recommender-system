import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from funcs import *
import streamlit.components.v1 as components
import pickle
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm
import random

def main():
    page_selection = []  # Set page_selection to an empty list or None as default

    st.set_page_config(layout="wide", initial_sidebar_state='expanded')
    
    
        
    sidebar_header = '''This is a demo to illustrate a recommender system that finds similar items to a given clothing article:'''
    
    page_options = ["Shirts",
                    "Bottoms",
                    "Tops"
                    ]
    
    st.sidebar.info(sidebar_header)

    page_options_more = [
                    "Popular on instagram",
                    "Trends",
                    "Recommends"
                    
                    ]
    
    
   

    page_selection_more = st.sidebar.radio("Try",page_options_more)

    if page_selection_more == "Trends" :
         page_selection = st.sidebar.multiselect("Choose", page_options, default=None)


##########################################################################################   
#########################################################################################
    if page_selection_more == "Trends"  :
        st.header("Choose your own fit from the drop down!")             
##########################################################################################          


    if "Shirts" in page_selection:

        st.subheader("Trending under Shirts")
        
        Simagefile = pd.read_csv('imageMEN_filenames.csv')
        Simagename = Simagefile.image_names.unique()
        
        Simage1 = np.random.choice(Simagename)
        Simage2 = np.random.choice(Simagename)
        Simage3 = np.random.choice(Simagename)
        while Simage2 == Simage1:
            Simage2 = np.random.choice(Simagename)
        while Simage3 == Simage2  or Simage3 == Simage1:
            Simage3 = np.random.choice(Simagename)  
            
        col1, col2, col3 = st.columns(3)
        with col1:   
            #st.text("Image 1") 
            st.image(get_item_image_without_jpeg_menshirt(str(Simage1), width=200, height=300))
            
        with col2:  
            #st.text("Image 2")  
            st.image(get_item_image_without_jpeg_menshirt(str(Simage2), width=200, height=300))
            
        with col3:
            #st.text("Image 3")
            st.image(get_item_image_without_jpeg_menshirt(str(Simage3), width=200, height=300))
            
        st.write("Would you like to see similar recommendations from the dataset?")
        if st.button("Yes",key="yeSSS_button"):
            feature_list = np.array(pickle.load(open('embeddings.pkl','rb')))  
            filenames = pickle.load(open('filenames.pkl','rb'))

            model = ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))
            model.trainable = False

            model = tensorflow.keras.Sequential([
                model,
                GlobalMaxPooling2D()
            ])

            st.subheader('Similar items from the dataset')

            
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
            
            

            
            # feature extract
            Sfeatures1 = feature_extraction(os.path.join("menshirt",Simage1),model)
            Sfeatures2 = feature_extraction(os.path.join("menshirt",Simage2),model)
            Sfeatures3 = feature_extraction(os.path.join("menshirt",Simage3),model)
            #st.text(features)
            # recommendention
            Sindices1 = recommend(Sfeatures1,feature_list)
            Sindices2 = recommend(Sfeatures2,feature_list)
            Sindices3 = recommend(Sfeatures3,feature_list)
            # Show recommended images
            
            st.write("Similar to image 1:")
            # Create columns to display images
            col1, col2, col3, col4, col5 = st.columns(5)
            # Display images in separate columns
            with col1:
                st.image(filenames[Sindices1[0][0]], width=150)  # Adjust width as needed
            with col2:
                st.image(filenames[Sindices1[0][1]], width=150)  # Adjust width as needed
            with col3:
                st.image(filenames[Sindices1[0][2]], width=150)  # Adjust width as needed
            with col4:
                st.image(filenames[Sindices1[0][3]], width=150)  # Adjust width as needed
            with col5:
                st.image(filenames[Sindices1[0][4]], width=150)  # Adjust width as needed

            st.text("")
            st.write("Similar to image 2:")
            # Create columns to display images
            col1, col2, col3, col4, col5 = st.columns(5)
            # Display images in separate columns
            with col1:
                st.image(filenames[Sindices2[0][0]], width=150)  # Adjust width as needed
            with col2:
                st.image(filenames[Sindices2[0][1]], width=150)  # Adjust width as needed
            with col3:
                st.image(filenames[Sindices2[0][2]], width=150)  # Adjust width as needed
            with col4:
                st.image(filenames[Sindices2[0][3]], width=150)  # Adjust width as needed
            with col5:
                st.image(filenames[Sindices2[0][4]], width=150)  # Adjust width as needed

            st.text("")
            st.write("Similar to image 3:")
            # Create columns to display images
            col1, col2, col3, col4, col5 = st.columns(5)
            # Display images in separate columns
            with col1:
                st.image(filenames[Sindices3[0][0]], width=150)  # Adjust width as needed
            with col2:
                st.image(filenames[Sindices3[0][1]], width=150)  # Adjust width as needed
            with col3:
                st.image(filenames[Sindices3[0][2]], width=150)  # Adjust width as needed
            with col4:
                st.image(filenames[Sindices3[0][3]], width=150)  # Adjust width as needed
            with col5:
                st.image(filenames[Sindices3[0][4]], width=150)  # Adjust width as needed
   
            
        
        if st.button("No", key="nOOO_button"):
            # Handle 'No' feedback
            st.write("You can continue with the other categories!")    
                 
##########################################################################################

    if "Bottoms" in page_selection:

        st.subheader("Trending under Bottoms")

        Pimagefile = pd.read_csv('imagePANT_filenames.csv')
        Pimagename = Pimagefile.image_names.unique()
        
        Pimage1 = np.random.choice(Pimagename)
        Pimage2 = np.random.choice(Pimagename)
        Pimage3 = np.random.choice(Pimagename)
        while Pimage2 == Pimage1:
            Pimage2 = np.random.choice(Pimagename)
        while Pimage3 == Pimage2  or Pimage3 == Pimage1:
            Pimage3 = np.random.choice(Pimagename)  
            
        col1, col2, col3 = st.columns(3)
        with col1:    
            st.image(get_item_image_without_jpeg_pants(str(Pimage1), width=200, height=300))
        with col2:    
            st.image(get_item_image_without_jpeg_pants(str(Pimage2), width=200, height=300))
        with col3:
            st.image(get_item_image_without_jpeg_pants(str(Pimage3), width=200, height=300))

        st.write("Would you like to similar recommendations from the dataset?")
        if st.button("Yes", key="yesss_button" ):

            feature_list = np.array(pickle.load(open('embeddings.pkl','rb')))  
            filenames = pickle.load(open('filenames.pkl','rb'))

            model = ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))
            model.trainable = False

            model = tensorflow.keras.Sequential([
                model,
                GlobalMaxPooling2D()
            ])

            st.subheader('Similar items from the dataset')

            
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
            
            

            
            # feature extract
            Pfeatures1 = feature_extraction(os.path.join("pants",Pimage1),model)
            Pfeatures2 = feature_extraction(os.path.join("pants",Pimage2),model)
            Pfeatures3 = feature_extraction(os.path.join("pants",Pimage3),model)
            #st.text(features)
            # recommendention
            Pindices1 = recommend(Pfeatures1,feature_list)
            Pindices2 = recommend(Pfeatures2,feature_list)
            Pindices3 = recommend(Pfeatures3,feature_list)
            # Show recommended images

            st.write("Similar to image 1:")            
            # Create columns to display images
            col1, col2, col3, col4, col5 = st.columns(5)
            # Display images in separate columns
            with col1:
                st.image(filenames[Pindices1[0][0]], width=150)  # Adjust width as needed
            with col2:
                st.image(filenames[Pindices1[0][1]], width=150)  # Adjust width as needed
            with col3:
                st.image(filenames[Pindices1[0][2]], width=150)  # Adjust width as needed
            with col4:
                st.image(filenames[Pindices1[0][3]], width=150)  # Adjust width as needed
            with col5:
                st.image(filenames[Pindices1[0][4]], width=150)  # Adjust width as needed


            st.text("")
            st.write("Similar to image 2:")
            # Create columns to display images
            col1, col2, col3, col4, col5 = st.columns(5)
            # Display images in separate columns
            with col1:
                st.image(filenames[Pindices2[0][0]], width=150)  # Adjust width as needed
            with col2:
                st.image(filenames[Pindices2[0][1]], width=150)  # Adjust width as needed
            with col3:
                st.image(filenames[Pindices2[0][2]], width=150)  # Adjust width as needed
            with col4:
                st.image(filenames[Pindices2[0][3]], width=150)  # Adjust width as needed
            with col5:
                st.image(filenames[Pindices2[0][4]], width=150)  # Adjust width as needed


            st.text("")
            st.write("Similar to image 3:")
            # Create columns to display images
            col1, col2, col3, col4, col5 = st.columns(5)
            # Display images in separate columns
            with col1:
                st.image(filenames[Pindices3[0][0]], width=150)  # Adjust width as needed
            with col2:
                st.image(filenames[Pindices3[0][1]], width=150)  # Adjust width as needed
            with col3:
                st.image(filenames[Pindices3[0][2]], width=150)  # Adjust width as needed
            with col4:
                st.image(filenames[Pindices3[0][3]], width=150)  # Adjust width as needed
            with col5:
                st.image(filenames[Pindices3[0][4]], width=150)  # Adjust width as needed

            
        

            
        
        if st.button("No", key = "noooo_button"):
            # Handle 'No' feedback
            st.write("You can continue with the other categories!")    
          
            
##########################################################################################
            
    
    if "Tops" in page_selection:
        
        st.subheader("Trending under Tops")
        imagefile = pd.read_csv('imageTOPS_filenames.csv')
        imagename = imagefile.image_names.unique()
        
        image1 = np.random.choice(imagename)
        image2 = np.random.choice(imagename)
        image3 = np.random.choice(imagename)
        while image2 == image1:
            image2 = np.random.choice(imagename)
        while image3 == image2  or image3 == image1:
            image3 = np.random.choice(imagename)  
            
        col1, col2, col3 = st.columns(3)
        with col1:    
            st.image(get_item_image_without_jpeg_tops(str(image1), width=200, height=300))
        with col2:    
            st.image(get_item_image_without_jpeg_tops(str(image2), width=200, height=300))
        with col3:
            st.image(get_item_image_without_jpeg_tops(str(image3), width=200, height=300))

        st.write("Would you like to similar recommendations from the dataset?")
        if st.button("Yes"):

            feature_list = np.array(pickle.load(open('embeddings.pkl','rb')))  
            filenames = pickle.load(open('filenames.pkl','rb'))

            model = ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))
            model.trainable = False

            model = tensorflow.keras.Sequential([
                model,
                GlobalMaxPooling2D()
            ])

            st.subheader('Similar items from the dataset')

            
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
            
            # feature extract
            features1 = feature_extraction(os.path.join("tops",image1),model)
            features2 = feature_extraction(os.path.join("tops",image2),model)
            features3 = feature_extraction(os.path.join("tops",image3),model)
            #st.text(features)
            # recommendention
            indices1 = recommend(features1,feature_list)
            indices2 = recommend(features2,feature_list)
            indices3 = recommend(features3,feature_list)
            # Show recommended images

            st.write("Similar to image 2:")            
            # Create columns to display images
            col1, col2, col3, col4, col5 = st.columns(5)
            # Display images in separate columns
            with col1:
                st.image(filenames[indices1[0][0]], width=150)  # Adjust width as needed
            with col2:
                st.image(filenames[indices1[0][1]], width=150)  # Adjust width as needed
            with col3:
                st.image(filenames[indices1[0][2]], width=150)  # Adjust width as needed
            with col4:
                st.image(filenames[indices1[0][3]], width=150)  # Adjust width as needed
            with col5:
                st.image(filenames[indices1[0][4]], width=150)  # Adjust width as needed

            st.text("")
            st.write("Similar to image 2:")
            # Create columns to display images
            col1, col2, col3, col4, col5 = st.columns(5)
            # Display images in separate columns
            with col1:
                st.image(filenames[indices2[0][0]], width=150)  # Adjust width as needed
            with col2:
                st.image(filenames[indices2[0][1]], width=150)  # Adjust width as needed
            with col3:
                st.image(filenames[indices2[0][2]], width=150)  # Adjust width as needed
            with col4:
                st.image(filenames[indices2[0][3]], width=150)  # Adjust width as needed
            with col5:
                st.image(filenames[indices2[0][4]], width=150)  # Adjust width as needed


            st.text("")
            st.write("Similar to image 3:")
            # Create columns to display images
            col1, col2, col3, col4, col5 = st.columns(5)
            # Display images in separate columns
            with col1:
                st.image(filenames[indices3[0][0]], width=150)  # Adjust width as needed
            with col2:
                st.image(filenames[indices3[0][1]], width=150)  # Adjust width as needed
            with col3:
                st.image(filenames[indices3[0][2]], width=150)  # Adjust width as needed
            with col4:
                st.image(filenames[indices3[0][3]], width=150)  # Adjust width as needed
            with col5:
                st.image(filenames[indices3[0][4]], width=150)  # Adjust width as needed

            
        

            
        
        if st.button("No"):
            # Handle 'No' feedback
            st.write("You can continue with the other categories!")    
                       
########################################################################################## 
            
    if page_selection_more == "Recommends"  : 
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
                # recommendention
                indices = recommend(features,feature_list)
                # Show recommended images
                st.header("Recommended Images")

                # Create columns to display images
                col1, col2, col3, col4, col5 = st.columns(5)
                # Display images in separate columns
                with col1:
                    st.image(filenames[indices[0][0]], width=150)  # Adjust width as needed
                with col2:
                    st.image(filenames[indices[0][1]], width=150)  # Adjust width as needed
                with col3:
                    st.image(filenames[indices[0][2]], width=150)  # Adjust width as needed
                with col4:
                    st.image(filenames[indices[0][3]], width=150)  # Adjust width as needed
                with col5:
                    st.image(filenames[indices[0][4]], width=150)  # Adjust width as needed
            else:
                st.header("Some error occured in file upload")
        
#########################################################################################  
    if page_selection_more == "Popular on instagram" :
        
        
        image_folder = "/Users/khushisingh/outfits"
        def get_random_images(folder_path, num_images):
            images = os.listdir(folder_path)
            random_images = random.sample(images, num_images)
            return random_images

        random_images = get_random_images(image_folder, 10)


        st.header("Popular on instagram")

        # Display the images in two rows of five columns each
        col1, col2, col3 = st.columns(3)

        with col1:
            for i in range(3):
                st.image(os.path.join(image_folder, random_images[i]),  width=200 )

        with col2:
            for i in range(3, 6):
                st.image(os.path.join(image_folder, random_images[i]), width=200)

        with col3:
            for i in range(6, 9):
                st.image(os.path.join(image_folder, random_images[i]), width=200)
               
        st.write("Like it?")
        if st.button("Yes"):
            # Handle 'Yes' feedback
            st.write("Thank you for your feedback!")
        if st.button("No"):
            # Handle 'No' feedback
            st.write("We'll work on improving it. Thank you for your feedback!")
#########################################################################################
 
##########################################################################################                                                                                        

if __name__ == '__main__':
    main()