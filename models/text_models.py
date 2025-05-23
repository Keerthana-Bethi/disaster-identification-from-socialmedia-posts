"""
Text-based models for disaster identification.
"""
import torch
import torch.nn as nn
from transformers import BertModel, XLNetModel, BertConfig, XLNetConfig, BertForSequenceClassification, XLNetForSequenceClassification
import numpy as np

# Global model instances
BERT_MODEL = None
XLNET_MODEL = None

def load_bert(num_classes=2):
    """
    Load and prepare BERT model for disaster identification.
    
    Args:
        num_classes: Number of output classes
        
    Returns:
        Model instance
    """
    global BERT_MODEL
    
    if BERT_MODEL is None:
        # Load model with pretrained weights
        BERT_MODEL = BertForSequenceClassification.from_pretrained(
            'bert-base-uncased',
            num_labels=num_classes,
            output_attentions=False,
            output_hidden_states=False
        )
    
    return BERT_MODEL

def load_xlnet(num_classes=2):
    """
    Load and prepare XLNet model for disaster identification.
    
    Args:
        num_classes: Number of output classes
        
    Returns:
        Model instance
    """
    global XLNET_MODEL
    
    if XLNET_MODEL is None:
        # Load model with pretrained weights
        XLNET_MODEL = XLNetForSequenceClassification.from_pretrained(
            'xlnet-base-cased',
            num_labels=num_classes,
            output_attentions=False,
            output_hidden_states=False
        )
    
    return XLNET_MODEL

def predict_with_bert(tokenized_text):
    """
    Make prediction using BERT model.
    
    Args:
        tokenized_text: Dictionary with input_ids and attention_mask tensors
        
    Returns:
        Model prediction
    """
    # Load model
    model = load_bert()
    model.eval()
    
    # Prepare inputs
    input_ids = tokenized_text['input_ids']
    attention_mask = tokenized_text['attention_mask']
    
    # Make prediction
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=1)
    
    return probabilities.detach().numpy()[0]

def predict_with_xlnet(tokenized_text):
    """
    Make prediction using XLNet model.
    
    Args:
        tokenized_text: Dictionary with input_ids and attention_mask tensors
        
    Returns:
        Model prediction
    """
    # Load model
    model = load_xlnet()
    model.eval()
    
    # Prepare inputs
    input_ids = tokenized_text['input_ids']
    attention_mask = tokenized_text['attention_mask']
    
    # Make prediction
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=1)
    
    return probabilities.detach().numpy()[0]

def predict_with_text_models(text, binary_classification=True):
    """
    Make predictions using both text models.
    
    Args:
        text: Raw text string
        binary_classification: If True, use binary classification (disaster/not disaster)
                              If False, use multi-class classification
    
    Returns:
        Dictionary with predictions from each model and ensemble
    """
    from preprocessing.text_preprocessing import clean_text, tokenize_for_bert, tokenize_for_xlnet
    
    # Clean and preprocess text
    cleaned_text = clean_text(text)
    
    # Tokenize for each model
    bert_tokens = tokenize_for_bert(cleaned_text)
    xlnet_tokens = tokenize_for_xlnet(cleaned_text)
    
    # Make predictions
    bert_pred = predict_with_bert(bert_tokens)
    xlnet_pred = predict_with_xlnet(xlnet_tokens)
    
    # Ensemble prediction (average)
    ensemble_pred = np.mean([bert_pred, xlnet_pred], axis=0)
    
    # Get model with highest confidence
    model_confidences = [np.max(bert_pred), np.max(xlnet_pred)]
    best_model_idx = np.argmax(model_confidences)
    best_model_name = ['BERT', 'XLNet'][best_model_idx]
    
    # Return results
    return {
        'bert': bert_pred,
        'xlnet': xlnet_pred,
        'ensemble': ensemble_pred,
        'best_model': best_model_name,
        'best_model_confidence': model_confidences[best_model_idx]
    }

def get_best_text_model(metrics):
    """
    Determine the best performing text model based on metrics.
    
    Args:
        metrics: Dictionary of model performance metrics
        
    Returns:
        Name of the best performing model
    """
    models = ['bert', 'xlnet']
    
    # Use F1 score as the primary metric
    best_f1 = 0
    best_model = None
    
    for model in models:
        if metrics.get(f'{model}_f1', 0) > best_f1:
            best_f1 = metrics.get(f'{model}_f1', 0)
            best_model = model
    
    return best_model or 'ensemble'  # Default to ensemble if no metrics available
