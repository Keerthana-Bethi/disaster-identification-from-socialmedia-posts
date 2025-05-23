"""
Image preprocessing utilities for disaster identification model.
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import requests
from io import BytesIO
import base64

def load_image_from_url(url, target_size=(224, 224)):
    """
    Load an image from a URL and preprocess it.
    
    Args:
        url: URL of the image
        target_size: Tuple of (height, width) to resize the image to
        
    Returns:
        Preprocessed image array
    """
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content))
        img = img.convert('RGB')  # Convert to RGB (in case of grayscale or RGBA)
        img = img.resize(target_size)
        return img
    except Exception as e:
        print(f"Error loading image from URL: {e}")
        # Return a blank image of the target size
        return Image.new('RGB', target_size, color='gray')

def load_image_from_upload(uploaded_file, target_size=(224, 224)):
    """
    Load an image from an uploaded file.
    
    Args:
        uploaded_file: File uploaded through Streamlit
        target_size: Tuple of (height, width) to resize the image to
        
    Returns:
        Preprocessed image array
    """
    try:
        img = Image.open(uploaded_file)
        img = img.convert('RGB')  # Convert to RGB (in case of grayscale or RGBA)
        img = img.resize(target_size)
        return img
    except Exception as e:
        print(f"Error loading uploaded image: {e}")
        # Return a blank image of the target size
        return Image.new('RGB', target_size, color='gray')

def preprocess_image(img, model_name):
    """
    Preprocess image based on the model requirements.
    
    Args:
        img: PIL Image object
        model_name: String identifying the model ('efficientnet', 'densenet', 'resnet')
        
    Returns:
        Preprocessed image tensor
    """
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    
    if model_name == 'efficientnet':
        return efficientnet_preprocess(img_array)
    elif model_name == 'densenet':
        return densenet_preprocess(img_array)
    elif model_name == 'resnet':
        return resnet_preprocess(img_array)
    else:
        # Default preprocessing (scale to [0,1])
        return img_array / 255.0

def data_augmentation():
    """
    Create a data augmentation pipeline for training.
    
    Returns:
        Data augmentation sequence
    """
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomContrast(0.1),
    ])

def image_to_base64(img):
    """
    Convert a PIL image to base64 string for display in web interface.
    
    Args:
        img: PIL Image object
        
    Returns:
        Base64 encoded string
    """
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")
