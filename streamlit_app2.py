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

    st.set_page_config(layout="wide", initial_sidebar_state='expanded')
    
    
        
    sidebar_header = '''This is a demo to illustrate a recommender system that finds similar items to a given clothing article:'''
    
    page_options = ["Choose",
                    "Trending",
                    "Popular on instagram",
                    "Streetstyle",
                    "Formal",
                    "Ethnic",
                    "Recommends",
                    ]
    
    st.sidebar.info(sidebar_header)


    
    page_selection = st.sidebar.radio("Try", page_options)


########################################################################################
    if page_selection == "Recommends"  : 
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
        

##########################################################################################    
########################################################################################
    if page_selection == "Trending"  : 


                feature_list = np.array(pickle.load(open('embeddings.pkl','rb')))  
                filenames = pickle.load(open('filenames.pkl','rb'))

                model = ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))
                model.trainable = False

                model = tensorflow.keras.Sequential([
                    model,
                    GlobalMaxPooling2D()
                ])

                st.title('Fashion Recommender System')

                
                def feature_extraction(img_path,model):
                    img = image.load_img(img_path, target_size=(224, 224))
                    img_array = image.img_to_array(img)
                    expanded_img_array = np.expand_dims(img_array, axis=0)
                    preprocessed_img = preprocess_input(expanded_img_array)
                    result = model.predict(preprocessed_img).flatten()
                    normalized_result = result / norm(result)

                    return normalized_result

                def recommend(features,feature_list):
                    neighbors = NearestNeighbors(n_neighbors=10, algorithm='brute', metric='euclidean')
                    neighbors.fit(feature_list)

                    distances, indices = neighbors.kneighbors([features])

                    return indices


                trend_rcmnds = pd.read_csv('image_filenames.csv')
                trend = trend_rcmnds.image_names.unique()
                get_item = st.sidebar.button('Get Random Item')
                if get_item:
                        rand_trend = np.random.choice(trend)
                        st.sidebar.image(get_item_image_without_jpeg(str(rand_trend), width=200, height=300))
                

               
                        # feature extract
                        features = feature_extraction(os.path.join("outfits",rand_trend),model)
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

                        col1, col2, col3, col4, col5 = st.columns(5)
                        # Display images in separate columns
                        with col1:
                            st.image(filenames[indices[0][5]], width=150)  # Adjust width as needed
                        with col2:
                            st.image(filenames[indices[0][6]], width=150)  # Adjust width as needed
                        with col3:
                            st.image(filenames[indices[0][7]], width=150)  # Adjust width as needed
                        with col4:
                            st.image(filenames[indices[0][8]], width=150)  # Adjust width as needed
                        with col5:
                            st.image(filenames[indices[0][9]], width=150)  # Adjust width as needed                            
                    
                
                    

#########################################################################################  
    if page_selection == "Popular on instagram" :
        
        
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
                st.image(os.path.join(image_folder, random_images[i]),  width=200 , caption= "for you")

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
    if page_selection == "Streetstyle" :   

                feature_list = np.array(pickle.load(open('embeddings.pkl','rb')))  
                filenames = pickle.load(open('filenames.pkl','rb'))

                model = ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))
                model.trainable = False

                model = tensorflow.keras.Sequential([
                    model,
                    GlobalMaxPooling2D()
                ])

                st.title('Fashion Recommender System')

                
                def feature_extraction(img_path,model):
                    img = image.load_img(img_path, target_size=(224, 224))
                    img_array = image.img_to_array(img)
                    expanded_img_array = np.expand_dims(img_array, axis=0)
                    preprocessed_img = preprocess_input(expanded_img_array)
                    result = model.predict(preprocessed_img).flatten()
                    normalized_result = result / norm(result)

                    return normalized_result

                def recommend(features,feature_list):
                    neighbors = NearestNeighbors(n_neighbors=10, algorithm='brute', metric='euclidean')
                    neighbors.fit(feature_list)

                    distances, indices = neighbors.kneighbors([features])

                    return indices


                trend_rcmnds = pd.read_csv('imageSS_filenames.csv')
                trend = trend_rcmnds.image_names.unique()
                get_item = st.sidebar.button('Get Random Item')
                if get_item:
                        rand_trend = np.random.choice(trend)
                        st.sidebar.image(get_item_image_without_jpeg_street(str(rand_trend), width=200, height=300))
                

               
                        # feature extract
                        features = feature_extraction(os.path.join("streetstyle",rand_trend),model)
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

                        col1, col2, col3, col4, col5 = st.columns(5)
                        # Display images in separate columns
                        with col1:
                            st.image(filenames[indices[0][5]], width=150)  # Adjust width as needed
                        with col2:
                            st.image(filenames[indices[0][6]], width=150)  # Adjust width as needed
                        with col3:
                            st.image(filenames[indices[0][7]], width=150)  # Adjust width as needed
                        with col4:
                            st.image(filenames[indices[0][8]], width=150)  # Adjust width as needed
                        with col5:
                            st.image(filenames[indices[0][9]], width=150)  # Adjust width as needed                            
                    
                
#########################################################################################
    if page_selection == "Formal" :   

                feature_list = np.array(pickle.load(open('embeddings.pkl','rb')))  
                filenames = pickle.load(open('filenames.pkl','rb'))

                model = ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))
                model.trainable = False

                model = tensorflow.keras.Sequential([
                    model,
                    GlobalMaxPooling2D()
                ])

                st.title('Fashion Recommender System')

                
                def feature_extraction(img_path,model):
                    img = image.load_img(img_path, target_size=(224, 224))
                    img_array = image.img_to_array(img)
                    expanded_img_array = np.expand_dims(img_array, axis=0)
                    preprocessed_img = preprocess_input(expanded_img_array)
                    result = model.predict(preprocessed_img).flatten()
                    normalized_result = result / norm(result)

                    return normalized_result

                def recommend(features,feature_list):
                    neighbors = NearestNeighbors(n_neighbors=10, algorithm='brute', metric='euclidean')
                    neighbors.fit(feature_list)

                    distances, indices = neighbors.kneighbors([features])

                    return indices


                trend_rcmnds = pd.read_csv('imageFO_filenames.csv')
                trend = trend_rcmnds.image_names.unique()
                get_item = st.sidebar.button('Get Random Item')
                if get_item:
                        rand_trend = np.random.choice(trend)
                        st.sidebar.image(get_item_image_without_jpeg_formal(str(rand_trend), width=200, height=300))
                

               
                        # feature extract
                        features = feature_extraction(os.path.join("formal",rand_trend),model)
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

                        col1, col2, col3, col4, col5 = st.columns(5)
                        # Display images in separate columns
                        with col1:
                            st.image(filenames[indices[0][5]], width=150)  # Adjust width as needed
                        with col2:
                            st.image(filenames[indices[0][6]], width=150)  # Adjust width as needed
                        with col3:
                            st.image(filenames[indices[0][7]], width=150)  # Adjust width as needed
                        with col4:
                            st.image(filenames[indices[0][8]], width=150)  # Adjust width as needed
                        with col5:
                            st.image(filenames[indices[0][9]], width=150)  # Adjust width as needed                            
                    
                                          
#########################################################################################     
    if page_selection == "Ethnic" :   

                feature_list = np.array(pickle.load(open('embeddings.pkl','rb')))  
                filenames = pickle.load(open('filenames.pkl','rb'))

                model = ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))
                model.trainable = False

                model = tensorflow.keras.Sequential([
                    model,
                    GlobalMaxPooling2D()
                ])

                st.title('Fashion Recommender System')

                
                def feature_extraction(img_path,model):
                    img = image.load_img(img_path, target_size=(224, 224))
                    img_array = image.img_to_array(img)
                    expanded_img_array = np.expand_dims(img_array, axis=0)
                    preprocessed_img = preprocess_input(expanded_img_array)
                    result = model.predict(preprocessed_img).flatten()
                    normalized_result = result / norm(result)

                    return normalized_result

                def recommend(features,feature_list):
                    neighbors = NearestNeighbors(n_neighbors=10, algorithm='brute', metric='euclidean')
                    neighbors.fit(feature_list)

                    distances, indices = neighbors.kneighbors([features])

                    return indices


                trend_rcmnds = pd.read_csv('imageET_filenames.csv')
                trend = trend_rcmnds.image_names.unique()
                get_item = st.sidebar.button('Get Random Item')
                if get_item:
                        rand_trend = np.random.choice(trend)
                        st.sidebar.image(get_item_image_without_jpeg_ethnic(str(rand_trend), width=200, height=300))
                

               
                        # feature extract
                        features = feature_extraction(os.path.join("ethnic",rand_trend),model)
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

                        col1, col2, col3, col4, col5 = st.columns(5)
                        # Display images in separate columns
                        with col1:
                            st.image(filenames[indices[0][5]], width=150)  # Adjust width as needed
                        with col2:
                            st.image(filenames[indices[0][6]], width=150)  # Adjust width as needed
                        with col3:
                            st.image(filenames[indices[0][7]], width=150)  # Adjust width as needed
                        with col4:
                            st.image(filenames[indices[0][8]], width=150)  # Adjust width as needed
                        with col5:
                            st.image(filenames[indices[0][9]], width=150)  # Adjust width as needed                            
                                                
#########################################################################################  
                            
    if page_selection == "Choose"  : 
        
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

        st.write("Would you like to see more of tops?")
        if st.button("Yes"):
            image_folder = "/Users/khushisingh/Desktop/fashion/tops"
            def get_random_images(folder_path, num_images):
                images = os.listdir(folder_path)
                random_images = random.sample(images, num_images)
                return random_images
                
            random_images = get_random_images(image_folder, 10)
            col1, col2, col3 = st.columns(3)

            with col1:
                for i in range(3):
                    st.image(os.path.join(image_folder, random_images[i]), width=200 )
                    

            with col2:
                for i in range(3, 6):
                    st.image(os.path.join(image_folder, random_images[i]), width=200 )

            with col3:
                for i in range(6, 9):
                    st.image(os.path.join(image_folder, random_images[i]), width=200)
                
            
        
        if st.button("No"):
            # Handle 'No' feedback
            st.write("You can continue with the other categories!")    
                 




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

        st.write("Would you like to see more of Bottoms?")
        if st.button("Yes",key="yesss_button"):
            Pimage_folder = "/Users/khushisingh/Desktop/fashion/pants"
            def Pget_random_images(folder_path, num_images):
                Pimages = os.listdir(folder_path)
                Prandom_images = random.sample(Pimages, num_images)
                return Prandom_images
                
            Prandom_images = Pget_random_images(Pimage_folder, 10)
            col1, col2, col3 = st.columns(3)

            with col1:
                for i in range(3):
                    st.image(os.path.join(Pimage_folder, Prandom_images[i]), width=200 )
                    

            with col2:
                for i in range(3, 6):
                    st.image(os.path.join(Pimage_folder, Prandom_images[i]), width=200 )

            with col3:
                for i in range(6, 9):
                    st.image(os.path.join(Pimage_folder, Prandom_images[i]), width=200)
                
            
        
        if st.button("No", key="nooo_button"):
            # Handle 'No' feedback
            st.write("You can continue with the other categories!")    
                 





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
            st.image(get_item_image_without_jpeg_menshirt(str(Simage1), width=200, height=300))
        with col2:    
            st.image(get_item_image_without_jpeg_menshirt(str(Simage2), width=200, height=300))
        with col3:
            st.image(get_item_image_without_jpeg_menshirt(str(Simage3), width=200, height=300))

        st.write("Would you like to see more of Shirts?")
        if st.button("Yes",key="yeSSS_button"):
            Simage_folder = "/Users/khushisingh/Desktop/fashion/menshirt"
            def Sget_random_images(folder_path, num_images):
                Simages = os.listdir(folder_path)
                Srandom_images = random.sample(Simages, num_images)
                return Srandom_images
                
            Srandom_images = Sget_random_images(Simage_folder, 10)
            col1, col2, col3 = st.columns(3)

            with col1:
                for i in range(3):
                    st.image(os.path.join(Simage_folder, Srandom_images[i]), width=200 )
                    

            with col2:
                for i in range(3, 6):
                    st.image(os.path.join(Simage_folder, Srandom_images[i]), width=200 )

            with col3:
                for i in range(6, 9):
                    st.image(os.path.join(Simage_folder, Srandom_images[i]), width=200)
                
            
        
        if st.button("No", key="nOOO_button"):
            # Handle 'No' feedback
            st.write("You can continue with the other categories!")    
               

             
##########################################################################################                                                                             

if __name__ == '__main__':
    main()