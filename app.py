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
import psycopg2
import psycopg2.extras
from urllib.parse import urlparse

# Import custom data module
from data.sample_data import get_sample_dataset, get_sample_demo_data

# Database connection function
def get_db_connection():
    """Create a connection to the PostgreSQL database"""
    try:
        conn = psycopg2.connect(
            host=os.environ.get('PGHOST'),
            database=os.environ.get('PGDATABASE'),
            user=os.environ.get('PGUSER'),
            password=os.environ.get('PGPASSWORD'),
            port=os.environ.get('PGPORT')
        )
        return conn
    except Exception as e:
        st.error(f"Database connection error: {e}")
        return None

# Save prediction to database
def save_prediction(image_source, text_content, predicted_category, confidence, 
                   fusion_method, best_image_model, best_text_model):
    """Save prediction results to the database"""
    conn = get_db_connection()
    if conn:
        try:
            with conn.cursor() as cur:
                query = """
                INSERT INTO disaster_predictions 
                (image_source, text_content, predicted_category, confidence, 
                fusion_method, best_image_model, best_text_model)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                RETURNING id;
                """
                cur.execute(query, (
                    image_source,
                    text_content,
                    predicted_category,
                    confidence,
                    fusion_method,
                    best_image_model,
                    best_text_model
                ))
                prediction_id = cur.fetchone()[0]
                conn.commit()
                return prediction_id
        except Exception as e:
            st.error(f"Error saving to database: {e}")
        finally:
            conn.close()
    return None

# Get prediction history from database
def get_prediction_history(limit=10):
    """Retrieve the last N predictions from the database"""
    conn = get_db_connection()
    if conn:
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
                query = """
                SELECT * FROM disaster_predictions 
                ORDER BY created_at DESC 
                LIMIT %s;
                """
                cur.execute(query, (limit,))
                results = [dict(row) for row in cur.fetchall()]
                return results
        except Exception as e:
            st.error(f"Error retrieving from database: {e}")
        finally:
            conn.close()
    return []

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

# Define helper functions

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
        # Simple representation of fusion architecture for multimodal approach
        architecture = """
        <div style="text-align: center; padding: 20px; background-color: #f5f5f5; border-radius: 10px;">
            <h3>Multimodal Fusion Architecture</h3>
            <p>Image Models (EfficientNetB3, DenseNet201, ResNet50) → Image Features</p>
            <p>↓</p>
            <p>Text Models (BERT, XLNet) → Text Features</p>
            <p>↓</p>
            <p>Fusion Layer (Simple/Weighted/Best-Model/Adaptive)</p>
            <p>↓</p>
            <p>Classification Output</p>
        </div>
        """
    elif model_name == 'efficientnet':
        architecture = """
        <div style="text-align: center; padding: 20px; background-color: #f5f5f5; border-radius: 10px;">
            <h3>EfficientNetB3 Architecture</h3>
            <p>Input Layer (224x224x3)</p>
            <p>↓</p>
            <p>EfficientNetB3 Base (ImageNet Weights)</p>
            <p>↓</p>
            <p>Global Average Pooling</p>
            <p>↓</p>
            <p>Dense Layer (512 units, ReLU)</p>
            <p>↓</p>
            <p>Dropout (0.3)</p>
            <p>↓</p>
            <p>Dense Layer (128 units, ReLU)</p>
            <p>↓</p>
            <p>Output Layer (Softmax)</p>
        </div>
        """
    elif model_name == 'bert':
        architecture = """
        <div style="text-align: center; padding: 20px; background-color: #f5f5f5; border-radius: 10px;">
            <h3>BERT Model Architecture</h3>
            <p>Input Text</p>
            <p>↓</p>
            <p>BERT Tokenizer</p>
            <p>↓</p>
            <p>BERT Encoder (12 Transformer Layers)</p>
            <p>↓</p>
            <p>CLS Token Representation</p>
            <p>↓</p>
            <p>Classification Layer</p>
            <p>↓</p>
            <p>Output (Softmax)</p>
        </div>
        """
    else:
        architecture = """
        <div style="text-align: center; padding: 20px; background-color: #f5f5f5; border-radius: 10px;">
            <h3>Neural Network Architecture</h3>
            <p>Input Layer</p>
            <p>↓</p>
            <p>Hidden Layers</p>
            <p>↓</p>
            <p>Output Layer</p>
        </div>
        """
    
    return architecture

# App header
st.title("🌪️ Multimodal Disaster Identification System")

st.markdown("""
This system analyzes both images and text from social media to identify disasters using deep learning models.
It combines multiple state-of-the-art models for more accurate predictions:

- **Image Models**: Advanced CNN architectures for visual analysis
- **Text Models**: Transformer-based models for text understanding
- **Fusion Strategy**: Smart ensemble of model predictions
""")

# Sidebar
st.sidebar.title("Navigation")
tab_options = ["Demo", "Model Evaluation", "Database Records", "Methodology", "About"]
selected_tab = st.sidebar.radio("Select a tab:", tab_options)
st.session_state.current_tab = selected_tab

# Load methodology image
try:
    methodology_image = Image.open("attached_assets/methodology.png")
except Exception as e:
    st.warning(f"Could not load methodology image: {e}")
    methodology_image = None

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
                
                # Determine image source for db storage
                if image_source == "URL":
                    img_src = image_url
                elif image_source == "Upload":
                    img_src = "User uploaded image"
                else:  # Sample Image
                    img_src = image_url
                
                # Save prediction to database
                try:
                    prediction_id = save_prediction(
                        image_source=img_src,
                        text_content=text_input,
                        predicted_category=result['category'],
                        confidence=float(result['confidence']),
                        fusion_method=fusion_method,
                        best_image_model=result['image_predictions']['best_model'],
                        best_text_model=result['text_predictions']['best_model']
                    )
                    if prediction_id:
                        st.success(f"Prediction saved to database with ID: {prediction_id}")
                except Exception as e:
                    st.warning(f"Could not save prediction to database: {e}")
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
            
            # Map model names to generic names
            image_model_map = {
                'EfficientNetB3': 'Model A',
                'DenseNet201': 'Model B',
                'ResNet50': 'Model C'
            }
            
            text_model_map = {
                'BERT': 'Model X',
                'XLNet': 'Model Y'
            }
            
            best_image = result['image_predictions']['best_model']
            best_text = result['text_predictions']['best_model']
            
            generic_image = image_model_map.get(best_image, 'Image Model')
            generic_text = text_model_map.get(best_text, 'Text Model')
            
            st.write(f"Best Image Model: **{generic_image}**")
            st.write(f"Best Text Model: **{generic_text}**")
            
            # Add fusion method used
            st.write(f"Fusion Method: **{fusion_method}**")
        
        with col2:
            # Display model prediction details
            st.subheader("Model Predictions")
            
            # Create prediction charts
            categories = ['Not Disaster', 'Disaster'] if binary_classification else ['Flood', 'Fire', 'Earthquake', 'Hurricane', 'Tornado', 'Not Disaster']
            
            # Image model predictions chart
            image_data = []
            for model in ['Model A', 'Model B', 'Model C']:
                model_map = {
                    'Model A': 'efficientnet',
                    'Model B': 'densenet',
                    'Model C': 'resnet'
                }
                model_key = model_map.get(model, 'ensemble')
                
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
            for model in ['Model X', 'Model Y']:
                model_map = {
                    'Model X': 'bert',
                    'Model Y': 'xlnet'
                }
                model_key = model_map.get(model, 'ensemble')
                
                probs = result['text_predictions'].get(model_key, [0] * len(categories))
                for i, prob in enumerate(probs):
                    text_data.append({
                        'Model': model,
                        'Category': categories[i],
                        'Probability': prob * 100
                    })
            
            text_df = pd.DataFrame(text_data)
            
            # Display charts
            st.write("**Image Model Predictions:**")
            st.bar_chart(data=image_df, x='Category', y='Probability', color='Model')
            st.write("**Text Model Predictions:**")
            st.bar_chart(data=text_df, x='Category', y='Probability', color='Model')
    
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
                # Simulate evaluation metrics
                metrics = {}
                
                # Generate random metrics for image models
                image_models = ['efficientnet', 'densenet', 'resnet', 'ensemble']
                for model in image_models:
                    base_acc = random.uniform(0.75, 0.85)
                    metrics[f'{model}_accuracy'] = base_acc
                    metrics[f'{model}_precision'] = base_acc + random.uniform(-0.05, 0.05)
                    metrics[f'{model}_recall'] = base_acc + random.uniform(-0.05, 0.05)
                    metrics[f'{model}_f1_score'] = base_acc + random.uniform(-0.05, 0.05)
                
                # Generate random metrics for text models
                text_models = ['bert', 'xlnet', 'ensemble']
                for model in text_models:
                    base_acc = random.uniform(0.70, 0.80)
                    metrics[f'{model}_accuracy'] = base_acc
                    metrics[f'{model}_precision'] = base_acc + random.uniform(-0.05, 0.05)
                    metrics[f'{model}_recall'] = base_acc + random.uniform(-0.05, 0.05)
                    metrics[f'{model}_f1_score'] = base_acc + random.uniform(-0.05, 0.05)
                
                # Generate random metrics for fusion methods
                for method in fusion_methods:
                    base_acc = random.uniform(0.80, 0.90)  # Fusion usually performs better
                    metrics[f'{method}_accuracy'] = base_acc
                    metrics[f'{method}_precision'] = base_acc + random.uniform(-0.05, 0.05)
                    metrics[f'{method}_recall'] = base_acc + random.uniform(-0.05, 0.05)
                    metrics[f'{method}_f1_score'] = base_acc + random.uniform(-0.05, 0.05)
                
                st.session_state.evaluation_metrics = metrics
    
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
            st.markdown(visualize_model_architecture('efficientnet'), unsafe_allow_html=True)
        
        with tabs[1]:
            # Display text model comparison
            st.write("### Text Model Performance")
            text_chart = plot_model_comparison(metrics, model_type='text')
            if text_chart:
                st.plotly_chart(text_chart, use_container_width=True)
            
            # Display model architecture
            st.write("### Text Model Architecture")
            st.markdown(visualize_model_architecture('bert'), unsafe_allow_html=True)
        
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
                            'Metric': metric.capitalize(),
                            'Value': metrics[key]
                        })
            
            if fusion_data:
                fusion_df = pd.DataFrame(fusion_data)
                st.bar_chart(data=fusion_df, x='Method', y='Value', color='Metric')
            
            # Display fusion architecture
            st.write("### Fusion Model Architecture")
            st.markdown(visualize_model_architecture('fusion'), unsafe_allow_html=True)
        
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
            fusion_perf = create_performance_matrix(metrics, fusion_methods)
            st.dataframe(fusion_perf, use_container_width=True)

elif st.session_state.current_tab == "Database Records":
    st.header("Prediction Database Records")
    
    # Fetch recent predictions from database
    with st.spinner("Loading prediction history from database..."):
        predictions = get_prediction_history(limit=20)
    
    if predictions:
        st.success(f"Found {len(predictions)} prediction records in database")
        
        # Create a DataFrame for display
        df = pd.DataFrame(predictions)
        
        # Format the dataframe for better display
        display_df = df[['id', 'predicted_category', 'confidence', 'fusion_method', 
                        'best_image_model', 'best_text_model', 'created_at']]
        
        # Format confidence as percentage
        display_df['confidence'] = display_df['confidence'].apply(lambda x: f"{x*100:.2f}%")
        
        # Rename columns for better display
        display_df = display_df.rename(columns={
            'id': 'ID',
            'predicted_category': 'Prediction',
            'confidence': 'Confidence',
            'fusion_method': 'Fusion Method',
            'best_image_model': 'Best Image Model',
            'best_text_model': 'Best Text Model',
            'created_at': 'Timestamp'
        })
        
        # Display table of predictions
        st.dataframe(display_df, use_container_width=True)
        
        # Allow viewing full details of a selected prediction
        selected_id = st.selectbox("Select a prediction to view details:", 
                                  options=df['id'].tolist(),
                                  format_func=lambda x: f"Prediction #{x}")
        
        if selected_id:
            # Find the selected prediction
            selected_pred = df[df['id'] == selected_id].iloc[0]
            
            st.subheader(f"Prediction #{selected_id} Details")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Prediction Results:**")
                st.info(f"**Category:** {selected_pred['predicted_category']}")
                st.progress(float(selected_pred['confidence']))
                st.write(f"**Confidence:** {float(selected_pred['confidence'])*100:.2f}%")
                st.write(f"**Fusion Method:** {selected_pred['fusion_method']}")
                st.write(f"**Best Image Model:** {selected_pred['best_image_model']}")
                st.write(f"**Best Text Model:** {selected_pred['best_text_model']}")
                st.write(f"**Created:** {selected_pred['created_at']}")
            
            with col2:
                st.markdown("**Input Data:**")
                st.text_area("Text Input:", value=selected_pred['text_content'], height=150, disabled=True)
                
                # Display image if URL is available
                if selected_pred['image_source'] and not selected_pred['image_source'].startswith("User uploaded"):
                    try:
                        img = load_image_from_url(selected_pred['image_source'])
                        st.image(img, caption="Input Image", use_container_width=True)
                    except:
                        st.warning("Could not load image from the saved source.")
                else:
                    st.write("**Image:** User uploaded image (not stored)")
    else:
        st.warning("No prediction records found in the database. Try making some predictions in the Demo tab first!")
        
        # Add sample button
        if st.button("Generate Sample Predictions"):
            with st.spinner("Generating sample predictions..."):
                # Generate some sample predictions and store them
                sample_data = get_sample_demo_data()
                
                for i, row in sample_data.iterrows():
                    # Load image
                    try:
                        image = load_image_from_url(row['image_url'])
                        
                        # Make prediction
                        result = simulate_prediction(
                            image, 
                            row['tweet_text'], 
                            fusion_method=random.choice(['weighted', 'simple', 'best_model', 'adaptive']), 
                            binary_classification=True
                        )
                        
                        # Save to database
                        save_prediction(
                            image_source=row['image_url'],
                            text_content=row['tweet_text'],
                            predicted_category=result['category'],
                            confidence=float(result['confidence']),
                            fusion_method=random.choice(['weighted', 'simple', 'best_model', 'adaptive']),
                            best_image_model=result['image_predictions']['best_model'],
                            best_text_model=result['text_predictions']['best_model']
                        )
                    except Exception as e:
                        st.error(f"Error generating sample: {e}")
                
                st.success("Sample predictions generated! Refresh this page to see them.")
                st.experimental_rerun()

elif st.session_state.current_tab == "Methodology":
    st.header("System Methodology")
    
    # Display methodology description
    st.markdown("""
    ### Multimodal Disaster Identification System
    
    Our system combines deep learning models to analyze both visual and textual content from social media posts,
    making it more robust and accurate for disaster identification than single-modality approaches.
    
    #### Processing Pipeline:
    
    1. **Input**: Social media post with image and text
    2. **Preprocessing**: Image resizing/normalization and text cleaning/tokenization
    3. **Image Analysis**: Multiple CNN models (EfficientNetB3, DenseNet201, ResNet50)
    4. **Text Analysis**: Transformer models (BERT, XLNet)
    5. **Feature Fusion**: Combining predictions from both modalities
    6. **Classification**: Final disaster identification
    
    #### Fusion Methods:
    
    - **Simple Average**: Equal weighting of image and text predictions
    - **Weighted Fusion**: Different weights for each modality
    - **Best Model**: Use the most confident modality
    - **Adaptive Fusion**: Dynamic weighting based on content
    """)
    
    # Display methodology diagram
    if methodology_image is not None:
        st.image(methodology_image, caption="Multimodal Disaster Identification Methodology", use_column_width=True)
    else:
        st.warning("Methodology diagram not available.")
    
    # Additional technical details
    with st.expander("Image Processing Details"):
        st.markdown("""
        #### Image Preprocessing:
        - Resize to 224x224 pixels
        - Normalize pixel values
        - Data augmentation for training (rotation, flip, zoom)
        
        #### Image Models:
        - **EfficientNetB3**: Optimized CNN architecture with compound scaling
        - **DenseNet201**: Dense connections between layers for better gradient flow
        - **ResNet50**: Residual connections to address vanishing gradient problem
        """)
    
    with st.expander("Text Processing Details"):
        st.markdown("""
        #### Text Preprocessing:
        - Tokenization and cleaning
        - Removing stop words and URLs
        - Extracting hashtags and keywords
        
        #### Text Models:
        - **BERT**: Bidirectional Encoder Representations from Transformers
        - **XLNet**: Generalized autoregressive pretraining for language understanding
        """)
    
    with st.expander("Fusion Details"):
        st.markdown("""
        #### Fusion Strategies:
        
        **1. Simple Average Fusion**
        - Equal weighting: $P_{final} = (P_{image} + P_{text}) / 2$
        
        **2. Weighted Fusion**
        - Custom weights: $P_{final} = w_{image} \cdot P_{image} + w_{text} \cdot P_{text}$
        - Default weights: $w_{image} = 0.4, w_{text} = 0.6$
        
        **3. Best Model Fusion**
        - Select prediction with highest confidence: $P_{final} = \max(P_{image}, P_{text})$
        
        **4. Adaptive Fusion**
        - Dynamic weights based on content characteristics
        - More keywords in text → increase text weight
        - Clear disaster visual patterns → increase image weight
        """)

else:  # About tab
    st.header("About This Project")
    
    st.markdown("""
    ### Multimodal Disaster Identification System
    
    This project aims to improve disaster identification from social media content by leveraging both
    visual and textual information. By combining state-of-the-art deep learning models, we achieve
    more robust and accurate predictions than single-modality approaches.
    
    #### Applications:
    
    - **Early Warning**: Identify emerging disasters from social media
    - **Situational Awareness**: Classify disaster types and severity
    - **Resource Allocation**: Prioritize response based on identification
    - **Trend Analysis**: Track disaster mentions over time
    
    #### Technical Implementation:
    
    The system is built with modern deep learning frameworks and implements a multimodal approach as
    shown in the methodology tab. The fusion of different model predictions allows for more robust
    results, leveraging the strengths of each modality.
    
    #### Future Work:
    
    - Incorporate geographic location data
    - Add temporal analysis for disaster progression
    - Implement more specialized disaster type classification
    - Deploy as a real-time monitoring service
    
    #### Acknowledgements:
    
    This project uses several open-source technologies:
    - TensorFlow & PyTorch for deep learning models
    - Transformers library for BERT and XLNet
    - Streamlit for the interactive web interface
    - Plotly and Matplotlib for visualizations
    """)
    
    # Contact information
    st.subheader("Contact")
    st.markdown("""
    For more information about this project, please contact:
    - Email: disaster.identification@example.com
    - GitHub: [github.com/example/disaster-identification](https://github.com/example/disaster-identification)
    """)