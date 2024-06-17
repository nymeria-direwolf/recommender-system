import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
import numpy as np
from numpy.linalg import norm
import os
from tqdm import tqdm
import pickle
from PIL import UnidentifiedImageError

model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

print(model.summary())

def extract_features(img_path, model):
    try:
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        expanded_img_array = np.expand_dims(img_array, axis=0)
        preprocessed_img = preprocess_input(expanded_img_array)
        result = model.predict(preprocessed_img).flatten()
        normalized_result = result / norm(result)
        return normalized_result
    except UnidentifiedImageError as e:
        print(f"Error loading image {img_path}: {e}")
        return None

filenames = []

for file in os.listdir('formal'):
    filenames.append(os.path.join('formal', file))

feature_list = []

for file in tqdm(filenames):
    features = extract_features(file, model)
    if features is not None:
        feature_list.append(features)

pickle.dump(feature_list, open('embeddingsFO.pkl', 'wb'))
pickle.dump(filenames, open('filenamesFO.pkl', 'wb'))
