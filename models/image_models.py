"""
Image-based models for disaster identification.
"""
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB3, DenseNet201, ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
import numpy as np
import os

# Global model instances
EFFICIENT_NET_MODEL = None
DENSENET_MODEL = None
RESNET_MODEL = None

def load_efficientnet(num_classes=2):
    """
    Load and prepare EfficientNetB3 model for disaster identification.
    
    Args:
        num_classes: Number of output classes
        
    Returns:
        Compiled model
    """
    global EFFICIENT_NET_MODEL
    
    if EFFICIENT_NET_MODEL is None:
        # Load base model with pre-trained weights
        base_model = EfficientNetB3(
            weights='imagenet',
            include_top=False,
            input_shape=(224, 224, 3)
        )
        
        # Add custom layers
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.3)(x)
        x = Dense(128, activation='relu')(x)
        predictions = Dense(num_classes, activation='softmax')(x)
        
        # Create model
        EFFICIENT_NET_MODEL = Model(inputs=base_model.input, outputs=predictions)
        
        # Freeze base model layers
        for layer in base_model.layers:
            layer.trainable = False
        
        # Compile model
        EFFICIENT_NET_MODEL.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
    
    return EFFICIENT_NET_MODEL

def load_densenet(num_classes=2):
    """
    Load and prepare DenseNet201 model for disaster identification.
    
    Args:
        num_classes: Number of output classes
        
    Returns:
        Compiled model
    """
    global DENSENET_MODEL
    
    if DENSENET_MODEL is None:
        # Load base model with pre-trained weights
        base_model = DenseNet201(
            weights='imagenet',
            include_top=False,
            input_shape=(224, 224, 3)
        )
        
        # Add custom layers
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.3)(x)
        x = Dense(128, activation='relu')(x)
        predictions = Dense(num_classes, activation='softmax')(x)
        
        # Create model
        DENSENET_MODEL = Model(inputs=base_model.input, outputs=predictions)
        
        # Freeze base model layers
        for layer in base_model.layers:
            layer.trainable = False
        
        # Compile model
        DENSENET_MODEL.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
    
    return DENSENET_MODEL

def load_resnet(num_classes=2):
    """
    Load and prepare ResNet50 model for disaster identification.
    
    Args:
        num_classes: Number of output classes
        
    Returns:
        Compiled model
    """
    global RESNET_MODEL
    
    if RESNET_MODEL is None:
        # Load base model with pre-trained weights
        base_model = ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=(224, 224, 3)
        )
        
        # Add custom layers
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.3)(x)
        x = Dense(128, activation='relu')(x)
        predictions = Dense(num_classes, activation='softmax')(x)
        
        # Create model
        RESNET_MODEL = Model(inputs=base_model.input, outputs=predictions)
        
        # Freeze base model layers
        for layer in base_model.layers:
            layer.trainable = False
        
        # Compile model
        RESNET_MODEL.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
    
    return RESNET_MODEL

def predict_with_image_models(img, binary_classification=True):
    """
    Make predictions using all three image models.
    
    Args:
        img: Preprocessed image (PIL Image object)
        binary_classification: If True, use binary classification (disaster/not disaster)
                              If False, use multi-class classification
    
    Returns:
        Dictionary with predictions from each model and ensemble
    """
    from preprocessing.image_preprocessing import preprocess_image
    
    num_classes = 2 if binary_classification else 6  # 6 classes: flood, fire, earthquake, hurricane, tornado, not_disaster
    
    # Load models
    efficient_net = load_efficientnet(num_classes)
    densenet = load_densenet(num_classes)
    resnet = load_resnet(num_classes)
    
    # Preprocess image for each model
    efficient_net_input = preprocess_image(img, 'efficientnet')
    densenet_input = preprocess_image(img, 'densenet')
    resnet_input = preprocess_image(img, 'resnet')
    
    # Make predictions
    efficient_net_pred = efficient_net.predict(efficient_net_input)
    densenet_pred = densenet.predict(densenet_input)
    resnet_pred = resnet.predict(resnet_input)
    
    # Ensemble prediction (average)
    ensemble_pred = np.mean([efficient_net_pred, densenet_pred, resnet_pred], axis=0)
    
    # Get model with highest confidence
    model_confidences = [
        np.max(efficient_net_pred),
        np.max(densenet_pred),
        np.max(resnet_pred)
    ]
    best_model_idx = np.argmax(model_confidences)
    best_model_name = ['EfficientNetB3', 'DenseNet201', 'ResNet50'][best_model_idx]
    
    # Return results
    return {
        'efficient_net': efficient_net_pred[0],
        'densenet': densenet_pred[0],
        'resnet': resnet_pred[0],
        'ensemble': ensemble_pred[0],
        'best_model': best_model_name,
        'best_model_confidence': model_confidences[best_model_idx]
    }

def get_best_image_model(metrics):
    """
    Determine the best performing image model based on metrics.
    
    Args:
        metrics: Dictionary of model performance metrics
        
    Returns:
        Name of the best performing model
    """
    models = ['efficientnet', 'densenet', 'resnet']
    
    # Use F1 score as the primary metric
    best_f1 = 0
    best_model = None
    
    for model in models:
        if metrics.get(f'{model}_f1', 0) > best_f1:
            best_f1 = metrics.get(f'{model}_f1', 0)
            best_model = model
    
    return best_model or 'ensemble'  # Default to ensemble if no metrics available
