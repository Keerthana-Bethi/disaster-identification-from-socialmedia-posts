"""
Model fusion techniques for multimodal disaster identification.
"""
import numpy as np

def simple_average_fusion(image_preds, text_preds):
    """
    Perform simple averaging of image and text predictions.
    
    Args:
        image_preds: Dictionary containing image model predictions
        text_preds: Dictionary containing text model predictions
        
    Returns:
        Dictionary with fused predictions
    """
    # Get ensemble predictions from each modality
    image_ensemble = image_preds['ensemble']
    text_ensemble = text_preds['ensemble']
    
    # Average the ensemble predictions
    fused_pred = np.mean([image_ensemble, text_ensemble], axis=0)
    
    # Calculate confidence
    confidence = np.max(fused_pred)
    predicted_class = np.argmax(fused_pred)
    
    return {
        'prediction': fused_pred,
        'predicted_class': predicted_class,
        'confidence': confidence
    }

def weighted_fusion(image_preds, text_preds, image_weight=0.5, text_weight=0.5):
    """
    Perform weighted fusion of image and text predictions.
    
    Args:
        image_preds: Dictionary containing image model predictions
        text_preds: Dictionary containing text model predictions
        image_weight: Weight for image predictions (default: 0.5)
        text_weight: Weight for text predictions (default: 0.5)
        
    Returns:
        Dictionary with fused predictions
    """
    # Ensure weights sum to 1
    total_weight = image_weight + text_weight
    image_weight = image_weight / total_weight
    text_weight = text_weight / total_weight
    
    # Get ensemble predictions from each modality
    image_ensemble = image_preds['ensemble']
    text_ensemble = text_preds['ensemble']
    
    # Apply weights
    fused_pred = (image_weight * image_ensemble) + (text_weight * text_ensemble)
    
    # Calculate confidence
    confidence = np.max(fused_pred)
    predicted_class = np.argmax(fused_pred)
    
    return {
        'prediction': fused_pred,
        'predicted_class': predicted_class,
        'confidence': confidence
    }

def best_model_fusion(image_preds, text_preds):
    """
    Select the best model from image and text modalities based on confidence.
    
    Args:
        image_preds: Dictionary containing image model predictions
        text_preds: Dictionary containing text model predictions
        
    Returns:
        Dictionary with fused predictions
    """
    # Get best model predictions from each modality
    image_best_model = image_preds['best_model']
    text_best_model = text_preds['best_model']
    
    image_confidence = image_preds['best_model_confidence']
    text_confidence = text_preds['best_model_confidence']
    
    # Choose the modality with higher confidence
    if image_confidence > text_confidence:
        best_model = image_best_model
        best_pred = image_preds[image_best_model.lower()] if image_best_model.lower() in image_preds else image_preds['ensemble']
    else:
        best_model = text_best_model
        best_pred = text_preds[text_best_model.lower()] if text_best_model.lower() in text_preds else text_preds['ensemble']
    
    # Calculate confidence
    confidence = np.max(best_pred)
    predicted_class = np.argmax(best_pred)
    
    return {
        'prediction': best_pred,
        'predicted_class': predicted_class,
        'confidence': confidence,
        'selected_model': best_model
    }

def adaptive_fusion(image_preds, text_preds, metrics=None):
    """
    Adaptively fuse predictions based on model performance metrics.
    
    Args:
        image_preds: Dictionary containing image model predictions
        text_preds: Dictionary containing text model predictions
        metrics: Dictionary of performance metrics for each model
        
    Returns:
        Dictionary with fused predictions
    """
    if metrics is None:
        # If no metrics provided, fall back to weighted fusion with equal weights
        return weighted_fusion(image_preds, text_preds)
    
    # Calculate weights based on F1 scores
    image_f1 = metrics.get('image_f1', 0.5)
    text_f1 = metrics.get('text_f1', 0.5)
    
    # Perform weighted fusion with dynamically calculated weights
    return weighted_fusion(image_preds, text_preds, image_weight=image_f1, text_weight=text_f1)

def get_disaster_category(prediction, binary_classification=True):
    """
    Convert numerical prediction to disaster category label.
    
    Args:
        prediction: Model prediction (indices)
        binary_classification: If True, use binary classification (disaster/not disaster)
                              If False, use multi-class classification
        
    Returns:
        String label or category name
    """
    if binary_classification:
        categories = ['Not Disaster', 'Disaster']
    else:
        categories = ['Flood', 'Fire', 'Earthquake', 'Hurricane', 'Tornado', 'Not Disaster']
    
    predicted_class = np.argmax(prediction)
    return categories[predicted_class]

def make_fused_prediction(image, text, fusion_method='weighted', binary_classification=True):
    """
    Make prediction using fused image and text models.
    
    Args:
        image: PIL Image object
        text: Raw text string
        fusion_method: Fusion method to use ('simple', 'weighted', 'best_model', 'adaptive')
        binary_classification: If True, use binary classification (disaster/not disaster)
                              If False, use multi-class classification
        
    Returns:
        Dictionary with fused prediction results
    """
    from models.image_models import predict_with_image_models
    from models.text_models import predict_with_text_models
    
    # Get predictions from image and text models
    image_preds = predict_with_image_models(image, binary_classification)
    text_preds = predict_with_text_models(text, binary_classification)
    
    # Apply fusion method
    if fusion_method == 'simple':
        fused_result = simple_average_fusion(image_preds, text_preds)
    elif fusion_method == 'best_model':
        fused_result = best_model_fusion(image_preds, text_preds)
    elif fusion_method == 'adaptive':
        # Would need metrics from a validation set for proper adaptive fusion
        fused_result = adaptive_fusion(image_preds, text_preds)
    else:  # default to weighted
        fused_result = weighted_fusion(image_preds, text_preds, image_weight=0.5, text_weight=0.5)
    
    # Add category label
    fused_result['category'] = get_disaster_category(fused_result['prediction'], binary_classification)
    
    # Add individual model predictions
    fused_result['image_predictions'] = image_preds
    fused_result['text_predictions'] = text_preds
    
    return fused_result
