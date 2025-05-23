"""
Text preprocessing utilities for disaster identification model.
"""
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from transformers import BertTokenizer, XLNetTokenizer
import torch

# Download required NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# Initialize tokenizers
BERT_TOKENIZER = None
XLNET_TOKENIZER = None

def load_tokenizers():
    """Load the tokenizers for BERT and XLNet"""
    global BERT_TOKENIZER, XLNET_TOKENIZER
    
    if BERT_TOKENIZER is None:
        BERT_TOKENIZER = BertTokenizer.from_pretrained('bert-base-uncased')
    
    if XLNET_TOKENIZER is None:
        XLNET_TOKENIZER = XLNetTokenizer.from_pretrained('xlnet-base-cased')

def clean_text(text):
    """
    Clean and normalize text data.
    
    Args:
        text: String text to clean
        
    Returns:
        Cleaned text string
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    
    # Remove user mentions and hashtags (or keep hashtags as they might be informative)
    text = re.sub(r'@\w+', '', text)
    
    # Extract hashtags for separate processing
    hashtags = re.findall(r'#(\w+)', text)
    
    # Remove special characters and numbers
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    
    # Tokenize
    tokens = nltk.word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    # Add back hashtags without the # symbol
    tokens.extend(hashtags)
    
    # Join tokens back into a string
    cleaned_text = ' '.join(tokens)
    
    return cleaned_text

def tokenize_for_bert(text, max_length=128):
    """
    Tokenize text for BERT model.
    
    Args:
        text: String text to tokenize
        max_length: Maximum sequence length
        
    Returns:
        Dictionary of tokenized inputs
    """
    load_tokenizers()
    
    encoded = BERT_TOKENIZER.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    
    return {
        'input_ids': encoded['input_ids'],
        'attention_mask': encoded['attention_mask']
    }

def tokenize_for_xlnet(text, max_length=128):
    """
    Tokenize text for XLNet model.
    
    Args:
        text: String text to tokenize
        max_length: Maximum sequence length
        
    Returns:
        Dictionary of tokenized inputs
    """
    load_tokenizers()
    
    encoded = XLNET_TOKENIZER.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    
    return {
        'input_ids': encoded['input_ids'],
        'attention_mask': encoded['attention_mask']
    }

def extract_disaster_keywords(text):
    """
    Extract disaster-related keywords from text.
    
    Args:
        text: String text to analyze
        
    Returns:
        List of extracted keywords
    """
    disaster_keywords = [
        'flood', 'fire', 'earthquake', 'hurricane', 'tornado', 'tsunami',
        'storm', 'disaster', 'emergency', 'damage', 'destruction', 'evacuation',
        'rescue', 'relief', 'aid', 'victim', 'survivor', 'trapped', 'stranded',
        'injured', 'killed', 'dead', 'death', 'missing', 'collapsed', 'destroyed',
        'damaged', 'burning', 'burnt', 'underwater', 'water', 'flames', 'smoke',
        'tremor', 'tremors', 'shock', 'aftershock', 'epicenter', 'magnitude',
        'wind', 'raining', 'rain', 'flooding', 'flooded', 'rising', 'overflow',
        'wildfire', 'bushfire', 'eruption', 'volcanic', 'landslide', 'mudslide',
        'avalanche', 'blizzard', 'hailstorm', 'drought', 'famine', 'epidemic',
        'pandemic', 'outbreak', 'explosion', 'bomb', 'terror', 'terrorist',
        'crash', 'accident', 'derailment', 'collision', 'sinking', 'capsized'
    ]
    
    text_lower = text.lower()
    found_keywords = []
    
    for keyword in disaster_keywords:
        if keyword in text_lower:
            found_keywords.append(keyword)
    
    # Also extract hashtags
    hashtags = re.findall(r'#(\w+)', text)
    found_keywords.extend([tag.lower() for tag in hashtags])
    
    return list(set(found_keywords))  # Remove duplicates
