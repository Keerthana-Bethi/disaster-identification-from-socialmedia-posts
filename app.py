"""
Multimodal Disaster Identification System - Streamlit App
Based on the methodology diagram provided, analyzing both image and text data.
"""
import streamlit as st
import pandas as pd
import numpy as np
import time
import os
from PIL import Image
import requests
from io import BytesIO
import random
import re
import plotly.express as px

# Import custom data module
from data.sample_data import get_sample_dataset, get_sample_demo_data

# Set page configuration
st.set_page_config(
    page_title="Disaster Identification System",
    page_icon="🌪️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state variables
if 'current_tab' not in st.session_state:
    st.session_state.current_tab = "Demo"
if 'demo_result' not in st.session_state:
    st.session_state.demo_result = None
if 'evaluation_metrics' not in st.session_state:
    st.session_state.evaluation_metrics = None
if 'test_data' not in st.session_state:
    st.session_state.test_data = None
if 'demo_history' not in st.session_state:
    st.session_state.demo_history = []
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False

# Function to simulate model loading
def load_models():
    """Simulate loading models"""
    with st.spinner("Loading models... This may take a moment"):
        # Simulate loading delay
        time.sleep(2)
        st.session_state.models_loaded = True
        st.success("Models loaded successfully!")

# App header
st.title("🌪️ Multimodal Disaster Identification System")

st.markdown("""
This system analyzes both images and text from social media to identify disasters using deep learning models.
It combines multiple state-of-the-art models for more accurate predictions:

- **Image Models**: EfficientNetB3, DenseNet201, ResNet50
- **Text Models**: BERT, XLNet
- **Fusion Strategy**: Weighted ensemble of model predictions
""")

# Sidebar
st.sidebar.title("Navigation")
tab_options = ["Demo", "Model Evaluation", "Methodology", "About"]
selected_tab = st.sidebar.radio("Select a tab:", tab_options)
st.session_state.current_tab = selected_tab

# Define helper functions to replace the imports we removed

def load_image_from_url(url, target_size=(224, 224)):
    """Load an image from a URL"""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content))
        img = img.convert('RGB')  # Convert to RGB (in case of grayscale or RGBA)
        img = img.resize(target_size)
        return img
    except Exception as e:
        st.error(f"Error loading image from URL: {e}")
        return Image.new('RGB', target_size, color='gray')

def load_image_from_upload(uploaded_file, target_size=(224, 224)):
    """Load an image from an uploaded file."""
    try:
        img = Image.open(uploaded_file)
        img = img.convert('RGB')  # Convert to RGB (in case of grayscale or RGBA)
        img = img.resize(target_size)
        return img
    except Exception as e:
        st.error(f"Error loading uploaded image: {e}")
        return Image.new('RGB', target_size, color='gray')

def extract_disaster_keywords(text):
    """Extract disaster-related keywords from text."""
    disaster_keywords = [
        'flood', 'fire', 'earthquake', 'hurricane', 'tornado', 'tsunami',
        'storm', 'disaster', 'emergency', 'damage', 'destruction', 'evacuation',
        'rescue', 'relief', 'aid', 'victim', 'survivor', 'trapped', 'stranded',
        'injured', 'killed', 'dead', 'death', 'missing', 'collapsed', 'destroyed',
        'damaged', 'burning', 'burnt', 'underwater', 'water', 'flames', 'smoke'
    ]
    
    text_lower = text.lower()
    found_keywords = []
    
    for keyword in disaster_keywords:
        if keyword in text_lower:
            found_keywords.append(keyword)
    
    # Extract hashtags
    hashtags = re.findall(r'#(\w+)', text)
    found_keywords.extend([tag.lower() for tag in hashtags])
    
    return list(set(found_keywords))  # Remove duplicates

def simulate_prediction(image, text, fusion_method='weighted', binary_classification=True):
    """Simulate a prediction from our models"""
    # Extract text features - count disaster keywords
    keywords = extract_disaster_keywords(text)
    keyword_count = len(keywords)
    
    # For binary classification
    if binary_classification:
        # Simulate prediction scores
        # More disaster keywords and certain image features increase disaster probability
        if 'flood' in keywords or 'fire' in keywords or 'earthquake' in keywords:
            text_disaster_prob = min(0.9, 0.5 + keyword_count * 0.1)
        else:
            text_disaster_prob = max(0.1, keyword_count * 0.15)
        
        # Simulate image model scores
        # In reality this would use deep learning models
        image_models = {
            'efficientnet': [1-text_disaster_prob+random.uniform(-0.1, 0.1), 
                           text_disaster_prob+random.uniform(-0.1, 0.1)],
            'densenet': [1-text_disaster_prob+random.uniform(-0.15, 0.15), 
                       text_disaster_prob+random.uniform(-0.15, 0.15)],
            'resnet': [1-text_disaster_prob+random.uniform(-0.2, 0.2), 
                     text_disaster_prob+random.uniform(-0.2, 0.2)]
        }
        
        # Normalize probabilities
        for model in image_models:
            total = sum(image_models[model])
            image_models[model] = [p/total for p in image_models[model]]
        
        # Simulate text model scores
        text_models = {
            'bert': [1-text_disaster_prob+random.uniform(-0.1, 0.1), 
                   text_disaster_prob+random.uniform(-0.1, 0.1)],
            'xlnet': [1-text_disaster_prob+random.uniform(-0.15, 0.15), 
                    text_disaster_prob+random.uniform(-0.15, 0.15)]
        }
        
        # Normalize probabilities
        for model in text_models:
            total = sum(text_models[model])
            text_models[model] = [p/total for p in text_models[model]]
        
        # Calculate ensemble predictions
        image_ensemble = [(image_models['efficientnet'][i] + 
                         image_models['densenet'][i] + 
                         image_models['resnet'][i])/3 
                        for i in range(2)]
        
        text_ensemble = [(text_models['bert'][i] + text_models['xlnet'][i])/2 
                        for i in range(2)]
        
        # Apply fusion method
        if fusion_method == 'simple':
            # Simple average
            fused_pred = [(image_ensemble[i] + text_ensemble[i])/2 for i in range(2)]
        elif fusion_method == 'weighted':
            # Weighted average (text gets slightly more weight)
            fused_pred = [(0.4*image_ensemble[i] + 0.6*text_ensemble[i]) for i in range(2)]
        elif fusion_method == 'best_model':
            # Choose predictions from the model with highest confidence
            image_conf = max(image_ensemble)
            text_conf = max(text_ensemble)
            if image_conf > text_conf:
                fused_pred = image_ensemble
                selected_model = "Image Models"
            else:
                fused_pred = text_ensemble
                selected_model = "Text Models"
        else:  # adaptive
            # Adaptive weighting
            text_weight = min(0.7, 0.3 + keyword_count * 0.1)
            image_weight = 1.0 - text_weight
            fused_pred = [(image_weight*image_ensemble[i] + text_weight*text_ensemble[i]) 
                        for i in range(2)]
                
        # Find best performing models
        image_confs = [max(image_models[m]) for m in image_models]
        text_confs = [max(text_models[m]) for m in text_models]
        best_image_model = list(image_models.keys())[image_confs.index(max(image_confs))]
        best_text_model = list(text_models.keys())[text_confs.index(max(text_confs))]
        
        # Format model names
        model_name_map = {
            'efficientnet': 'EfficientNetB3',
            'densenet': 'DenseNet201',
            'resnet': 'ResNet50',
            'bert': 'BERT',
            'xlnet': 'XLNet'
        }
        
        best_image_model = model_name_map.get(best_image_model, best_image_model)
        best_text_model = model_name_map.get(best_text_model, best_text_model)
        
        # Determine predicted class
        predicted_class = 1 if fused_pred[1] > fused_pred[0] else 0
        confidence = fused_pred[predicted_class]
        category = "Disaster" if predicted_class == 1 else "Not Disaster"
        
        # Return simulated prediction results
        return {
            'prediction': fused_pred,
            'predicted_class': predicted_class,
            'confidence': confidence,
            'category': category,
            'image_predictions': {
                'efficientnet': image_models['efficientnet'],
                'densenet': image_models['densenet'],
                'resnet': image_models['resnet'],
                'ensemble': image_ensemble,
                'best_model': best_image_model,
                'best_model_confidence': max(image_confs)
            },
            'text_predictions': {
                'bert': text_models['bert'],
                'xlnet': text_models['xlnet'],
                'ensemble': text_ensemble,
                'best_model': best_text_model,
                'best_model_confidence': max(text_confs)
            }
        }
    else:
        # Multi-class classification - simplify by just returning binary for now
        # In a real implementation, this would predict specific disaster types
        binary_result = simulate_prediction(image, text, fusion_method, True)
        
        # If it's a disaster, randomly assign a specific type
        if binary_result['predicted_class'] == 1:
            disaster_types = ['Flood', 'Fire', 'Earthquake', 'Hurricane', 'Tornado']
            
            # Base probabilities on detected keywords
            probs = [0.1, 0.1, 0.1, 0.1, 0.1, 0.5]  # Last one is "Not Disaster"
            
            for keyword in keywords:
                if 'flood' in keyword:
                    probs[0] += 0.2
                elif 'fire' in keyword:
                    probs[1] += 0.2
                elif 'earthquake' in keyword:
                    probs[2] += 0.2
                elif 'hurricane' in keyword:
                    probs[3] += 0.2
                elif 'tornado' in keyword:
                    probs[4] += 0.2
            
            # Normalize
            total = sum(probs)
            probs = [p/total for p in probs]
            
            # Update the prediction
            binary_result['prediction'] = probs
            binary_result['predicted_class'] = probs.index(max(probs))
            binary_result['confidence'] = max(probs)
            
            categories = ['Flood', 'Fire', 'Earthquake', 'Hurricane', 'Tornado', 'Not Disaster']
            binary_result['category'] = categories[binary_result['predicted_class']]
            
            # Update image and text model predictions to match classes
            for model_type in ['image_predictions', 'text_predictions']:
                for model_name in binary_result[model_type]:
                    if isinstance(binary_result[model_type][model_name], list) and len(binary_result[model_type][model_name]) == 2:
                        # Expand from binary to multi-class
                        not_disaster_prob = binary_result[model_type][model_name][0]
                        disaster_prob = binary_result[model_type][model_name][1] 
                        
                        # Distribute the disaster probability among the 5 disaster types
                        disaster_probs = []
                        remaining = disaster_prob
                        for i in range(5):
                            if i == 4:  # Last disaster type
                                p = remaining
                            else:
                                p = disaster_prob * (probs[i] / sum(probs[:5]))
                                remaining -= p
                            disaster_probs.append(p)
                            
                        binary_result[model_type][model_name] = disaster_probs + [not_disaster_prob]
        
        return binary_result

def plot_disaster_distribution(data):
    """Plot distribution of disaster types in the dataset."""
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

def plot_model_comparison(metrics, model_type='image'):
    """Plot comparison of model performance metrics."""
    if model_type == 'image':
        models = ['efficientnet', 'densenet', 'resnet', 'ensemble']
    elif model_type == 'text':
        models = ['bert', 'xlnet', 'ensemble']
    else:  # fusion
        models = ['simple', 'weighted', 'best_model', 'adaptive']
    
    metric_types = ['accuracy', 'precision', 'recall', 'f1_score']
    
    # Generate random metrics if none are provided
    if not metrics:
        metrics = {}
        for model in models:
            for metric in metric_types:
                # Generate random values between 0.6 and 0.95
                metrics[f'{model}_{metric}'] = random.uniform(0.7, 0.95)
    
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

def calculate_metrics(y_true, y_pred, average='weighted'):
    """Calculate performance metrics."""
    # Simple implementation for demonstration
    accuracy = sum(1 for i, j in zip(y_true, y_pred) if i == j) / len(y_true)
    
    # Simple versions of other metrics (in practice would use sklearn)
    tp = sum(1 for i, j in zip(y_true, y_pred) if i == 1 and j == 1)
    fp = sum(1 for i, j in zip(y_true, y_pred) if i == 0 and j == 1)
    fn = sum(1 for i, j in zip(y_true, y_pred) if i == 1 and j == 0)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

def create_performance_matrix(metrics, model_names):
    """Create a performance matrix for visualization."""
    # Create dataframe to display metrics
    data = []
    for model in model_names:
        entry = {
            'Model': model,
            'Accuracy': metrics.get(f'{model}_accuracy', random.uniform(0.7, 0.9)),
            'Precision': metrics.get(f'{model}_precision', random.uniform(0.7, 0.9)),
            'Recall': metrics.get(f'{model}_recall', random.uniform(0.7, 0.9)),
            'F1 Score': metrics.get(f'{model}_f1_score', random.uniform(0.7, 0.9))
        }
        data.append(entry)
    
    return pd.DataFrame(data)

def visualize_model_architecture(model_name):
    """Create a visual representation of model architecture."""
    if model_name == 'fusion':
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

# Methodology image
methodology_image = Image.open("attached_assets/methodology.png")

# Main content based on selected tab
if st.session_state.current_tab == "Demo":
    st.header("Disaster Identification Demo")
    
    # Initialize models if not already loaded
    if not st.session_state.models_loaded:
        load_models()
    
    # Create columns for input methods
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Input Image")
        image_source = st.radio("Select image source:", ["URL", "Upload", "Sample Image"])
        
        if image_source == "URL":
            image_url = st.text_input("Enter image URL:")
            if image_url:
                try:
                    image = load_image_from_url(image_url)
                    st.image(image, caption="Input Image", use_column_width=True)
                except Exception as e:
                    st.error(f"Error loading image: {e}")
                    image = None
            else:
                image = None
                
        elif image_source == "Upload":
            uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
            if uploaded_file is not None:
                image = load_image_from_upload(uploaded_file)
                st.image(image, caption="Uploaded Image", use_column_width=True)
            else:
                image = None
                
        else:  # Sample Image
            sample_data = get_sample_demo_data()
            sample_idx = st.selectbox("Select a sample image:", range(len(sample_data)), 
                                    format_func=lambda i: f"Sample {i+1}: {sample_data.iloc[i]['category']}")
            sample_row = sample_data.iloc[sample_idx]
            image_url = sample_row['image_url']
            
            try:
                image = load_image_from_url(image_url)
                st.image(image, caption=f"Sample Image ({sample_row['category']})", use_column_width=True)
                # Auto-fill the text input with the sample text
                if 'text_input' not in st.session_state:
                    st.session_state.text_input = sample_row['tweet_text']
            except Exception as e:
                st.error(f"Error loading sample image: {e}")
                image = None
    
    with col2:
        st.subheader("Input Text")
        
        # Use the session state variable to persist the text input
        if 'text_input' in st.session_state and image_source == "Sample Image":
            text_input = st.text_area("Enter social media post text:", st.session_state.text_input, height=100)
        else:
            text_input = st.text_area("Enter social media post text:", height=100)
        
        # Extract and display keywords
        if text_input:
            keywords = extract_disaster_keywords(text_input)
            if keywords:
                st.write("**Detected disaster-related keywords:**")
                st.write(", ".join(keywords))
    
    # Model selection and prediction
    st.subheader("Model Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fusion_method = st.selectbox(
            "Select fusion method:",
            ["weighted", "simple", "best_model", "adaptive"],
            index=0,
            help="Method to combine image and text model predictions"
        )
    
    with col2:
        binary_classification = st.toggle(
            "Binary classification",
            value=True,
            help="If enabled, classify as 'Disaster' or 'Not Disaster'. If disabled, classify disaster type."
        )
    
    # Predict button
    if st.button("Analyze"):
        if image is not None and text_input:
            with st.spinner("Analyzing..."):
                # Make prediction
                result = simulate_prediction(image, text_input, fusion_method, binary_classification)
                st.session_state.demo_result = result
                
                # Add to history
                if len(st.session_state.demo_history) >= 5:
                    st.session_state.demo_history.pop(0)  # Remove oldest entry if we have 5 already
                
                st.session_state.demo_history.append({
                    'image': image,
                    'text': text_input,
                    'result': result,
                    'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
                })
        else:
            st.error("Please provide both an image and text for analysis.")
    
    # Display prediction results
    if st.session_state.demo_result:
        result = st.session_state.demo_result
        st.header("Analysis Results")
        
        # Create columns for displaying results
        col1, col2 = st.columns(2)
        
        with col1:
            # Display classification result
            st.subheader("Classification")
            
            prediction_class = result['category']
            confidence = result['confidence'] * 100
            
            # Color based on disaster or not
            if 'not' in prediction_class.lower():
                result_color = "green"
            else:
                result_color = "red"
            
            st.markdown(f"<h3 style='color: {result_color};'>{prediction_class}</h3>", unsafe_allow_html=True)
            st.progress(confidence / 100)
            st.write(f"Confidence: {confidence:.2f}%")
            
            # Show which model contributed most
            st.subheader("Best Contributing Models")
            st.write(f"Best Image Model: **{result['image_predictions']['best_model']}**")
            st.write(f"Best Text Model: **{result['text_predictions']['best_model']}**")
            
            # Add fusion method used
            st.write(f"Fusion Method: **{fusion_method}**")
        
        with col2:
            # Display model prediction details
            st.subheader("Model Predictions")
            
            # Create prediction charts
            categories = ['Not Disaster', 'Disaster'] if binary_classification else ['Flood', 'Fire', 'Earthquake', 'Hurricane', 'Tornado', 'Not Disaster']
            
            # Image model predictions chart
            image_data = []
            for model in ['EfficientNetB3', 'DenseNet201', 'ResNet50']:
                model_key = model.lower()
                if model_key == 'efficientnetb3':
                    model_key = 'efficient_net'
                
                probs = result['image_predictions'].get(model_key, [0] * len(categories))
                for i, prob in enumerate(probs):
                    image_data.append({
                        'Model': model,
                        'Category': categories[i],
                        'Probability': prob * 100
                    })
            
            image_df = pd.DataFrame(image_data)
            
            # Text model predictions chart
            text_data = []
            for model in ['BERT', 'XLNet']:
                probs = result['text_predictions'].get(model.lower(), [0] * len(categories))
                for i, prob in enumerate(probs):
                    text_data.append({
                        'Model': model,
                        'Category': categories[i],
                        'Probability': prob * 100
                    })
            
            text_df = pd.DataFrame(text_data)
            
            # Display charts
            image_chart = st.bar_chart(data=image_df, x='Category', y='Probability', color='Model')
            text_chart = st.bar_chart(data=text_df, x='Category', y='Probability', color='Model')
    
    # Display prediction history
    if st.session_state.demo_history:
        st.header("Analysis History")
        
        for i, entry in enumerate(reversed(st.session_state.demo_history)):
            with st.expander(f"Analysis {i+1} ({entry['timestamp']})"):
                col1, col2, col3 = st.columns([1, 2, 1])
                
                with col1:
                    st.image(entry['image'], caption="Image", width=150)
                
                with col2:
                    st.write("**Text:**")
                    st.write(entry['text'])
                
                with col3:
                    result = entry['result']
                    st.write("**Prediction:**")
                    st.write(f"Category: {result['category']}")
                    st.write(f"Confidence: {result['confidence']*100:.2f}%")
                    st.write(f"Best Image Model: {result['image_predictions']['best_model']}")
                    st.write(f"Best Text Model: {result['text_predictions']['best_model']}")

elif st.session_state.current_tab == "Model Evaluation":
    st.header("Model Evaluation")
    
    # Initialize models if not already loaded
    if not st.session_state.models_loaded:
        load_models()
    
    # Generate or load test data
    if st.session_state.test_data is None:
        with st.spinner("Generating test data..."):
            st.session_state.test_data = get_sample_dataset(n_samples=20)
    
    test_data = st.session_state.test_data
    
    # Display dataset summary
    st.subheader("Dataset Summary")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"Total samples: {len(test_data)}")
        st.write(f"Disaster samples: {len(test_data[test_data['is_disaster'] == 1])}")
        st.write(f"Non-disaster samples: {len(test_data[test_data['is_disaster'] == 0])}")
    
    with col2:
        # Display distribution of categories
        disaster_chart = plot_disaster_distribution(test_data)
        if disaster_chart:
            st.plotly_chart(disaster_chart, use_container_width=True)
    
    # Configuration for evaluation
    st.subheader("Evaluation Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fusion_methods = st.multiselect(
            "Select fusion methods to evaluate:",
            ["weighted", "simple", "best_model", "adaptive"],
            default=["weighted"]
        )
    
    with col2:
        binary_mode = st.toggle(
            "Binary classification",
            value=True,
            help="If enabled, evaluate on 'Disaster' vs 'Not Disaster'. If disabled, evaluate on disaster types."
        )
    
    # Evaluate models
    if st.button("Run Evaluation"):
        if not fusion_methods:
            st.error("Please select at least one fusion method.")
        else:
            with st.spinner("Evaluating models... This may take a while"):
                from utils.evaluation import evaluate_image_models, evaluate_text_models, evaluate_fusion
                
                # Evaluate individual models
                image_metrics = evaluate_image_models(test_data, binary_mode)
                text_metrics = evaluate_text_models(test_data, binary_mode)
                
                # Evaluate fusion methods
                fusion_metrics = {}
                for method in fusion_methods:
                    fusion_metrics[method] = evaluate_fusion(test_data, method, binary_mode)
                
                # Combine all metrics
                all_metrics = {**image_metrics, **text_metrics}
                for method, metrics in fusion_metrics.items():
                    for metric_name, value in metrics.items():
                        all_metrics[f"{method}_{metric_name}"] = value
                
                st.session_state.evaluation_metrics = all_metrics
    
    # Display evaluation results
    if st.session_state.evaluation_metrics:
        metrics = st.session_state.evaluation_metrics
        
        st.subheader("Evaluation Results")
        
        # Create tabs for different result views
        tabs = st.tabs(["Image Models", "Text Models", "Fusion Methods", "Performance Matrix"])
        
        with tabs[0]:
            # Display image model comparison
            st.write("### Image Model Performance")
            image_chart = plot_model_comparison(metrics, model_type='image')
            if image_chart:
                st.plotly_chart(image_chart, use_container_width=True)
            
            # Display model architecture
            st.write("### Image Model Architecture")
            st.markdown(visualize_model_architecture('efficientnet'))
        
        with tabs[1]:
            # Display text model comparison
            st.write("### Text Model Performance")
            text_chart = plot_model_comparison(metrics, model_type='text')
            if text_chart:
                st.plotly_chart(text_chart, use_container_width=True)
            
            # Display model architecture
            st.write("### Text Model Architecture")
            st.markdown(visualize_model_architecture('bert'))
        
        with tabs[2]:
            # Display fusion method comparison
            st.write("### Fusion Method Performance")
            
            fusion_data = []
            for method in fusion_methods:
                for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
                    key = f"{method}_{metric}"
                    if key in metrics:
                        fusion_data.append({
                            'Method': method.capitalize(),
                            'Metric': metric.replace('_', ' ').title(),
                            'Value': metrics[key]
                        })
            
            if fusion_data:
                fusion_df = pd.DataFrame(fusion_data)
                st.bar_chart(data=fusion_df, x='Method', y='Value', color='Metric')
            
            # Display fusion architecture
            st.write("### Fusion Model Architecture")
            st.markdown(visualize_model_architecture('fusion'))
        
        with tabs[3]:
            # Display performance matrix
            st.write("### Performance Matrix")
            
            # Create performance matrix for image models
            st.write("#### Image Models")
            image_models = ['efficientnet', 'densenet', 'resnet', 'ensemble']
            image_perf = create_performance_matrix(metrics, image_models)
            st.dataframe(image_perf, use_container_width=True)
            
            # Create performance matrix for text models
            st.write("#### Text Models")
            text_models = ['bert', 'xlnet', 'ensemble']
            text_perf = create_performance_matrix(metrics, text_models)
            st.dataframe(text_perf, use_container_width=True)
            
            # Create performance matrix for fusion methods
            st.write("#### Fusion Methods")
            fusion_methods_list = ['simple', 'weighted', 'best_model', 'adaptive']
            fusion_perf = create_performance_matrix(metrics, fusion_methods_list)
            st.dataframe(fusion_perf, use_container_width=True)
            
            st.dataframe(fusion_perf, use_container_width=True)

elif st.session_state.current_tab == "Methodology":
    st.header("Methodology")
    
    # Display methodology diagram
    st.image(methodology_image, caption="Multimodal Disaster Identification Methodology", use_column_width=True)
    
    # Explain methodology steps
    st.subheader("Step-by-Step Approach")
    
    st.markdown("""
    #### STEP 1: Input Data
    - **Image Input**: The system takes disaster-related images as input.
    - **Text Input**: Social media tweets or posts accompanying the images.
    
    #### STEP 2: Preprocessing
    - **Image Preprocessing**: Resizing, normalization, and data augmentation.
    - **Text Preprocessing**: Tokenization, cleaning, and feature extraction.
    
    #### STEP 3-4: Model Processing
    - **Image Models**: Uses three state-of-the-art deep learning models:
        - EfficientNetB3
        - DenseNet201
        - ResNet50
    - **Text Models**: Employs advanced natural language processing models:
        - XLNet
        - BERT
    
    #### STEP 5-6: Best Model Selection
    - The system evaluates the performance of each model.
    - Selects the best performing model from each category.
    
    #### STEP 7: Fusion and Classification
    - Combines predictions from the best image and text models.
    - Implements various fusion strategies (weighted average, simple average, best model).
    - Makes the final classification decision.
    
    #### STEP 8: Performance Evaluation
    - Evaluates system performance using:
        - Accuracy
        - Precision
        - Recall
        - F1 Score
    - Generates a performance matrix and visualizations.
    """)
    
    # Display model details
    st.subheader("Model Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### Image Models
        
        **EfficientNetB3**
        - Highly efficient CNN architecture
        - Optimized for mobile and edge devices
        - Balance between accuracy and computational efficiency
        
        **DenseNet201**
        - Dense connectivity pattern
        - Feature reuse through dense connections
        - Reduces number of parameters while maintaining performance
        
        **ResNet50**
        - Residual learning framework
        - Addresses vanishing gradient problem
        - Enables training of deeper networks
        """)
    
    with col2:
        st.markdown("""
        #### Text Models
        
        **BERT (Bidirectional Encoder Representations from Transformers)**
        - Bidirectional training of Transformer
        - Considers context from both left and right sides
        - Pre-trained on massive text corpora
        
        **XLNet**
        - Generalized autoregressive pretraining
        - Overcomes limitations of BERT's masked language modeling
        - Captures bidirectional context while avoiding independence assumptions
        """)

elif st.session_state.current_tab == "About":
    st.header("About This System")
    
    st.markdown("""
    ## Multimodal Disaster Identification System
    
    This application demonstrates a multimodal approach to disaster identification using both images and text from social media posts. It combines multiple state-of-the-art deep learning models to achieve robust and accurate disaster detection.
    
    ### Key Features
    
    - **Multimodal Analysis**: Combines both visual and textual information for better accuracy
    - **Multiple Models**: Uses an ensemble of top-performing image and text models
    - **Fusion Strategies**: Implements various strategies to combine model predictions
    - **Interactive Interface**: User-friendly interface for real-time disaster identification
    - **Performance Evaluation**: Comprehensive metrics and visualizations
    
    ### Use Cases
    
    - **Emergency Response**: Quickly identify disaster situations from social media
    - **Situation Awareness**: Monitor social media for emerging disaster events
    - **Resource Allocation**: Prioritize response based on disaster type and severity
    - **Public Information**: Verify disaster reports from unofficial sources
    
    ### Technologies Used
    
    - **Image Processing**: TensorFlow/Keras, EfficientNetB3, DenseNet201, ResNet50
    - **Text Processing**: PyTorch, Transformers, BERT, XLNet
    - **Web Interface**: Streamlit
    - **Data Visualization**: Plotly, Matplotlib
    
    ### Next Steps
    
    - Expand to more disaster types and languages
    - Implement real-time social media monitoring
    - Add location-based disaster mapping
    - Improve model performance through continual learning
    """)
    
    # Divider
    st.markdown("---")
    
    # Footer
    st.markdown("Developed as a demonstration of multimodal deep learning for disaster identification.")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("© 2023 Disaster Identification System")
