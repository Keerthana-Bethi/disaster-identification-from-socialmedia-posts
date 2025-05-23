"""
Multimodal Disaster Identification System - Streamlit App
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
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

# Import custom data module - the only one we'll directly import
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
                result = make_fused_prediction(image, text_input, fusion_method, binary_classification)
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
            st.components.v1.html(visualize_model_architecture('efficientnet'), height=500)
        
        with tabs[1]:
            # Display text model comparison
            st.write("### Text Model Performance")
            text_chart = plot_model_comparison(metrics, model_type='text')
            if text_chart:
                st.plotly_chart(text_chart, use_container_width=True)
            
            # Display model architecture
            st.write("### Text Model Architecture")
            st.components.v1.html(visualize_model_architecture('bert'), height=500)
        
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
            st.components.v1.html(visualize_model_architecture('fusion'), height=500)
        
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
            fusion_perf = pd.DataFrame(columns=['Method', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])
            
            for method in fusion_methods:
                fusion_perf = fusion_perf._append({
                    'Method': method.capitalize(),
                    'Accuracy': metrics.get(f'{method}_accuracy', 0),
                    'Precision': metrics.get(f'{method}_precision', 0),
                    'Recall': metrics.get(f'{method}_recall', 0),
                    'F1 Score': metrics.get(f'{method}_f1_score', 0)
                }, ignore_index=True)
            
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
