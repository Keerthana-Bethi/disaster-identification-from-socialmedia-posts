"""
Visualization utilities for disaster identification models.
"""
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

def plot_model_comparison(metrics, model_type='image'):
    """
    Plot comparison of model performance metrics.
    
    Args:
        metrics: Dictionary of metrics
        model_type: Type of models ('image', 'text', or 'fusion')
        
    Returns:
        Plotly figure
    """
    if model_type == 'image':
        models = ['efficientnet', 'densenet', 'resnet', 'ensemble']
    elif model_type == 'text':
        models = ['bert', 'xlnet', 'ensemble']
    else:  # fusion
        models = ['simple', 'weighted', 'best_model', 'adaptive']
    
    metric_types = ['accuracy', 'precision', 'recall', 'f1_score']
    
    # Prepare data for plotting
    data = []
    for model in models:
        for metric in metric_types:
            key = f'{model}_{metric}'
            if key in metrics:
                data.append({
                    'Model': model,
                    'Metric': metric.replace('_', ' ').title(),
                    'Value': metrics[key]
                })
    
    if not data:
        return None
    
    df = pd.DataFrame(data)
    
    # Create grouped bar chart
    fig = px.bar(
        df, 
        x='Model', 
        y='Value', 
        color='Metric',
        barmode='group',
        title=f'{model_type.title()} Model Performance Comparison',
        labels={'Value': 'Score', 'Model': 'Model'},
        height=400
    )
    
    # Improve layout
    fig.update_layout(
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        xaxis={'categoryorder': 'total descending'}
    )
    
    return fig

def plot_confidence_distribution(predictions, model_name):
    """
    Plot distribution of confidence scores.
    
    Args:
        predictions: List of prediction confidence values
        model_name: Name of the model
        
    Returns:
        Plotly figure
    """
    fig = px.histogram(
        predictions,
        nbins=20,
        title=f'Confidence Distribution for {model_name}',
        labels={'value': 'Confidence', 'count': 'Frequency'},
        height=300
    )
    
    # Add vertical line for mean confidence
    mean_conf = np.mean(predictions)
    fig.add_vline(x=mean_conf, line_dash='dash', line_color='red', 
                 annotation_text=f'Mean: {mean_conf:.2f}', 
                 annotation_position='top right')
    
    return fig

def plot_disaster_distribution(data):
    """
    Plot distribution of disaster types in the dataset.
    
    Args:
        data: DataFrame with disaster data
        
    Returns:
        Plotly figure
    """
    if 'category' not in data.columns:
        return None
    
    # Count disasters by category
    category_counts = data['category'].value_counts().reset_index()
    category_counts.columns = ['Category', 'Count']
    
    # Create pie chart
    fig = px.pie(
        category_counts,
        values='Count',
        names='Category',
        title='Distribution of Disaster Types',
        hole=0.4
    )
    
    # Improve layout
    fig.update_traces(textposition='inside', textinfo='percent+label')
    
    return fig

def plot_feature_importance(importance_scores, feature_names):
    """
    Plot feature importance.
    
    Args:
        importance_scores: Importance scores for features
        feature_names: Names of features
        
    Returns:
        Plotly figure
    """
    # Sort features by importance
    indices = np.argsort(importance_scores)[::-1]
    sorted_features = [feature_names[i] for i in indices]
    sorted_scores = importance_scores[indices]
    
    # Create horizontal bar chart
    fig = px.bar(
        x=sorted_scores,
        y=sorted_features,
        orientation='h',
        title='Feature Importance',
        labels={'x': 'Importance Score', 'y': 'Feature'},
        height=500
    )
    
    # Improve layout
    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
    
    return fig

def plot_confusion_matrix_interactive(cm, class_names):
    """
    Create an interactive confusion matrix plot.
    
    Args:
        cm: Confusion matrix
        class_names: List of class names
        
    Returns:
        Plotly figure
    """
    # Create heatmap
    fig = px.imshow(
        cm,
        x=class_names,
        y=class_names,
        labels=dict(x="Predicted", y="True", color="Count"),
        title="Confusion Matrix",
        color_continuous_scale="Blues",
        aspect="auto"
    )
    
    # Add text annotations
    annotations = []
    for i in range(len(cm)):
        for j in range(len(cm[i])):
            annotations.append(
                dict(
                    x=j,
                    y=i,
                    text=str(cm[i, j]),
                    showarrow=False,
                    font=dict(color="white" if cm[i, j] > cm.max() / 2 else "black")
                )
            )
    
    fig.update_layout(annotations=annotations)
    
    return fig

def visualize_model_architecture(model_name):
    """
    Create a visual representation of model architecture.
    
    Args:
        model_name: Name of the model to visualize
        
    Returns:
        SVG string representation of the model architecture
    """
    if model_name == 'efficientnet':
        # Simple representation of EfficientNetB3 architecture
        architecture = """
        <svg width="400" height="500" xmlns="http://www.w3.org/2000/svg">
            <rect x="100" y="20" width="200" height="40" fill="#4CAF50" stroke="black" />
            <text x="200" y="45" text-anchor="middle" fill="white">Input (224x224x3)</text>
            
            <rect x="100" y="80" width="200" height="40" fill="#2196F3" stroke="black" />
            <text x="200" y="105" text-anchor="middle" fill="white">EfficientNetB3 Base</text>
            
            <rect x="100" y="140" width="200" height="30" fill="#9C27B0" stroke="black" />
            <text x="200" y="160" text-anchor="middle" fill="white">Global Average Pooling</text>
            
            <rect x="100" y="190" width="200" height="30" fill="#FF9800" stroke="black" />
            <text x="200" y="210" text-anchor="middle" fill="white">Dense (512, ReLU)</text>
            
            <rect x="100" y="240" width="200" height="30" fill="#607D8B" stroke="black" />
            <text x="200" y="260" text-anchor="middle" fill="white">Dropout (0.3)</text>
            
            <rect x="100" y="290" width="200" height="30" fill="#FF9800" stroke="black" />
            <text x="200" y="310" text-anchor="middle" fill="white">Dense (128, ReLU)</text>
            
            <rect x="100" y="340" width="200" height="30" fill="#F44336" stroke="black" />
            <text x="200" y="360" text-anchor="middle" fill="white">Output (Softmax)</text>
            
            <line x1="200" y1="60" x2="200" y2="80" stroke="black" stroke-width="2" />
            <line x1="200" y1="120" x2="200" y2="140" stroke="black" stroke-width="2" />
            <line x1="200" y1="170" x2="200" y2="190" stroke="black" stroke-width="2" />
            <line x1="200" y1="220" x2="200" y2="240" stroke="black" stroke-width="2" />
            <line x1="200" y1="270" x2="200" y2="290" stroke="black" stroke-width="2" />
            <line x1="200" y1="320" x2="200" y2="340" stroke="black" stroke-width="2" />
        </svg>
        """
    elif model_name == 'bert':
        # Simple representation of BERT architecture
        architecture = """
        <svg width="400" height="500" xmlns="http://www.w3.org/2000/svg">
            <rect x="100" y="20" width="200" height="40" fill="#4CAF50" stroke="black" />
            <text x="200" y="45" text-anchor="middle" fill="white">Input Text</text>
            
            <rect x="100" y="80" width="200" height="40" fill="#9C27B0" stroke="black" />
            <text x="200" y="105" text-anchor="middle" fill="white">Tokenization</text>
            
            <rect x="100" y="140" width="200" height="30" fill="#2196F3" stroke="black" />
            <text x="200" y="160" text-anchor="middle" fill="white">Embedding Layer</text>
            
            <rect x="100" y="190" width="200" height="40" fill="#FF9800" stroke="black" />
            <text x="200" y="210" text-anchor="middle" fill="white">BERT Encoder (12 layers)</text>
            
            <rect x="100" y="250" width="200" height="30" fill="#607D8B" stroke="black" />
            <text x="200" y="270" text-anchor="middle" fill="white">[CLS] Token Representation</text>
            
            <rect x="100" y="300" width="200" height="30" fill="#FF9800" stroke="black" />
            <text x="200" y="320" text-anchor="middle" fill="white">Dense Layer</text>
            
            <rect x="100" y="350" width="200" height="30" fill="#F44336" stroke="black" />
            <text x="200" y="370" text-anchor="middle" fill="white">Output (Softmax)</text>
            
            <line x1="200" y1="60" x2="200" y2="80" stroke="black" stroke-width="2" />
            <line x1="200" y1="120" x2="200" y2="140" stroke="black" stroke-width="2" />
            <line x1="200" y1="170" x2="200" y2="190" stroke="black" stroke-width="2" />
            <line x1="200" y1="230" x2="200" y2="250" stroke="black" stroke-width="2" />
            <line x1="200" y1="280" x2="200" y2="300" stroke="black" stroke-width="2" />
            <line x1="200" y1="330" x2="200" y2="350" stroke="black" stroke-width="2" />
        </svg>
        """
    elif model_name == 'fusion':
        # Simple representation of fusion architecture
        architecture = """
        <svg width="600" height="500" xmlns="http://www.w3.org/2000/svg">
            <!-- Image Path -->
            <rect x="50" y="20" width="200" height="40" fill="#4CAF50" stroke="black" />
            <text x="150" y="45" text-anchor="middle" fill="white">Image Input</text>
            
            <rect x="50" y="80" width="200" height="40" fill="#2196F3" stroke="black" />
            <text x="150" y="105" text-anchor="middle" fill="white">Image Model</text>
            
            <rect x="50" y="140" width="200" height="30" fill="#9C27B0" stroke="black" />
            <text x="150" y="160" text-anchor="middle" fill="white">Image Features</text>
            
            <!-- Text Path -->
            <rect x="350" y="20" width="200" height="40" fill="#4CAF50" stroke="black" />
            <text x="450" y="45" text-anchor="middle" fill="white">Text Input</text>
            
            <rect x="350" y="80" width="200" height="40" fill="#2196F3" stroke="black" />
            <text x="450" y="105" text-anchor="middle" fill="white">Text Model</text>
            
            <rect x="350" y="140" width="200" height="30" fill="#9C27B0" stroke="black" />
            <text x="450" y="160" text-anchor="middle" fill="white">Text Features</text>
            
            <!-- Fusion -->
            <rect x="200" y="220" width="200" height="40" fill="#FF9800" stroke="black" />
            <text x="300" y="245" text-anchor="middle" fill="white">Fusion Layer</text>
            
            <rect x="200" y="280" width="200" height="30" fill="#607D8B" stroke="black" />
            <text x="300" y="300" text-anchor="middle" fill="white">Combined Features</text>
            
            <rect x="200" y="330" width="200" height="30" fill="#FF9800" stroke="black" />
            <text x="300" y="350" text-anchor="middle" fill="white">Classification Layer</text>
            
            <rect x="200" y="380" width="200" height="30" fill="#F44336" stroke="black" />
            <text x="300" y="400" text-anchor="middle" fill="white">Output (Softmax)</text>
            
            <!-- Connections -->
            <line x1="150" y1="60" x2="150" y2="80" stroke="black" stroke-width="2" />
            <line x1="150" y1="120" x2="150" y2="140" stroke="black" stroke-width="2" />
            <line x1="450" y1="60" x2="450" y2="80" stroke="black" stroke-width="2" />
            <line x1="450" y1="120" x2="450" y2="140" stroke="black" stroke-width="2" />
            
            <line x1="150" y1="170" x2="150" y2="220" stroke="black" stroke-width="2" />
            <line x1="150" y1="220" x2="200" y2="240" stroke="black" stroke-width="2" />
            <line x1="450" y1="170" x2="450" y2="220" stroke="black" stroke-width="2" />
            <line x1="450" y1="220" x2="400" y2="240" stroke="black" stroke-width="2" />
            
            <line x1="300" y1="260" x2="300" y2="280" stroke="black" stroke-width="2" />
            <line x1="300" y1="310" x2="300" y2="330" stroke="black" stroke-width="2" />
            <line x1="300" y1="360" x2="300" y2="380" stroke="black" stroke-width="2" />
        </svg>
        """
    else:
        # Default architecture visualization
        architecture = """
        <svg width="400" height="300" xmlns="http://www.w3.org/2000/svg">
            <rect x="100" y="20" width="200" height="40" fill="#4CAF50" stroke="black" />
            <text x="200" y="45" text-anchor="middle" fill="white">Input Layer</text>
            
            <rect x="100" y="100" width="200" height="40" fill="#2196F3" stroke="black" />
            <text x="200" y="125" text-anchor="middle" fill="white">Hidden Layers</text>
            
            <rect x="100" y="180" width="200" height="40" fill="#F44336" stroke="black" />
            <text x="200" y="205" text-anchor="middle" fill="white">Output Layer</text>
            
            <line x1="200" y1="60" x2="200" y2="100" stroke="black" stroke-width="2" />
            <line x1="200" y1="140" x2="200" y2="180" stroke="black" stroke-width="2" />
        </svg>
        """
    
    return architecture
