"""
Evaluation utilities for disaster identification models.
"""
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd

def calculate_metrics(y_true, y_pred, average='weighted'):
    """
    Calculate performance metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        average: Averaging method for multi-class metrics
        
    Returns:
        Dictionary of metrics
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average=average, zero_division=0)
    recall = recall_score(y_true, y_pred, average=average, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=average, zero_division=0)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

def evaluate_image_models(test_data, binary_classification=True):
    """
    Evaluate performance of image models.
    
    Args:
        test_data: Test dataset
        binary_classification: If True, use binary classification
                              If False, use multi-class classification
    
    Returns:
        Dictionary of metrics for each model
    """
    from models.image_models import predict_with_image_models
    from preprocessing.image_preprocessing import load_image_from_url
    
    y_true = test_data['is_disaster'] if binary_classification else test_data['category_id']
    
    # Predictions from each model
    efficient_net_preds = []
    densenet_preds = []
    resnet_preds = []
    ensemble_preds = []
    
    for _, row in test_data.iterrows():
        # Load and preprocess image
        img = load_image_from_url(row['image_url'])
        
        # Get predictions
        preds = predict_with_image_models(img, binary_classification)
        
        # Store predictions
        efficient_net_preds.append(np.argmax(preds['efficient_net']))
        densenet_preds.append(np.argmax(preds['densenet']))
        resnet_preds.append(np.argmax(preds['resnet']))
        ensemble_preds.append(np.argmax(preds['ensemble']))
    
    # Calculate metrics
    average = 'binary' if binary_classification else 'weighted'
    
    metrics = {
        'efficientnet_metrics': calculate_metrics(y_true, efficient_net_preds, average),
        'densenet_metrics': calculate_metrics(y_true, densenet_preds, average),
        'resnet_metrics': calculate_metrics(y_true, resnet_preds, average),
        'ensemble_metrics': calculate_metrics(y_true, ensemble_preds, average)
    }
    
    # Flatten metrics for easier access
    flat_metrics = {}
    for model, model_metrics in metrics.items():
        model_name = model.split('_')[0]
        for metric_name, value in model_metrics.items():
            flat_metrics[f'{model_name}_{metric_name}'] = value
    
    return flat_metrics

def evaluate_text_models(test_data, binary_classification=True):
    """
    Evaluate performance of text models.
    
    Args:
        test_data: Test dataset
        binary_classification: If True, use binary classification
                              If False, use multi-class classification
    
    Returns:
        Dictionary of metrics for each model
    """
    from models.text_models import predict_with_text_models
    
    y_true = test_data['is_disaster'] if binary_classification else test_data['category_id']
    
    # Predictions from each model
    bert_preds = []
    xlnet_preds = []
    ensemble_preds = []
    
    for _, row in test_data.iterrows():
        # Get predictions
        preds = predict_with_text_models(row['tweet_text'], binary_classification)
        
        # Store predictions
        bert_preds.append(np.argmax(preds['bert']))
        xlnet_preds.append(np.argmax(preds['xlnet']))
        ensemble_preds.append(np.argmax(preds['ensemble']))
    
    # Calculate metrics
    average = 'binary' if binary_classification else 'weighted'
    
    metrics = {
        'bert_metrics': calculate_metrics(y_true, bert_preds, average),
        'xlnet_metrics': calculate_metrics(y_true, xlnet_preds, average),
        'ensemble_metrics': calculate_metrics(y_true, ensemble_preds, average)
    }
    
    # Flatten metrics for easier access
    flat_metrics = {}
    for model, model_metrics in metrics.items():
        model_name = model.split('_')[0]
        for metric_name, value in model_metrics.items():
            flat_metrics[f'{model_name}_{metric_name}'] = value
    
    return flat_metrics

def evaluate_fusion(test_data, fusion_method='weighted', binary_classification=True):
    """
    Evaluate performance of fused models.
    
    Args:
        test_data: Test dataset
        fusion_method: Fusion method to use
        binary_classification: If True, use binary classification
                              If False, use multi-class classification
    
    Returns:
        Dictionary of metrics
    """
    from models.fusion import make_fused_prediction
    from preprocessing.image_preprocessing import load_image_from_url
    
    y_true = test_data['is_disaster'] if binary_classification else test_data['category_id']
    y_pred = []
    
    for _, row in test_data.iterrows():
        # Load and preprocess image
        img = load_image_from_url(row['image_url'])
        
        # Make fused prediction
        result = make_fused_prediction(img, row['tweet_text'], fusion_method, binary_classification)
        
        # Store prediction
        y_pred.append(result['predicted_class'])
    
    # Calculate metrics
    average = 'binary' if binary_classification else 'weighted'
    metrics = calculate_metrics(y_true, y_pred, average)
    
    # Add fusion method to metrics
    metrics['fusion_method'] = fusion_method
    
    return metrics

def create_performance_matrix(metrics, model_names):
    """
    Create a performance matrix for visualization.
    
    Args:
        metrics: Dictionary of metrics
        model_names: List of model names to include
    
    Returns:
        DataFrame containing performance matrix
    """
    performance_matrix = pd.DataFrame(columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])
    
    for model in model_names:
        performance_matrix = performance_matrix.append({
            'Model': model,
            'Accuracy': metrics.get(f'{model}_accuracy', 0),
            'Precision': metrics.get(f'{model}_precision', 0),
            'Recall': metrics.get(f'{model}_recall', 0),
            'F1 Score': metrics.get(f'{model}_f1_score', 0)
        }, ignore_index=True)
    
    return performance_matrix

def plot_confusion_matrix(y_true, y_pred, classes, normalize=False, title=None, cmap=plt.cm.Blues):
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        classes: List of class names
        normalize: Whether to normalize values
        title: Plot title
        cmap: Color map
    
    Returns:
        Figure with confusion matrix
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    
    # Set labels
    ax.set(xticks=np.arange(cm.shape[1]),
          yticks=np.arange(cm.shape[0]),
          xticklabels=classes, yticklabels=classes,
          title=title,
          ylabel='True label',
          xlabel='Predicted label')
    
    # Rotate x tick labels and set alignment
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Loop over data to create text annotations
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black")
    
    fig.tight_layout()
    return fig
