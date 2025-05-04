import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
import os
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import hashlib
import re
import json

# Initialize NLTK components
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

ps = PorterStemmer()

# Custom CSS for better UI
st.markdown("""
    <style>
    /* Main container styles */
    .main {
        background-color: #ffffff;
    }
    .stApp {
        background-color: #ffffff;
    }
    
    /* Text area styles */
    .stTextArea > div > div > textarea {
        font-size: 18px !important;
        padding: 15px !important;
        border-radius: 10px !important;
        border: 2px solid #e9ecef !important;
        background-color: #ffffff !important;
        color: #000000 !important;
    }
    .stTextArea > div > div > textarea:focus {
        border-color: #4CAF50 !important;
        box-shadow: 0 0 0 0.2rem rgba(76, 175, 80, 0.25) !important;
    }
    
    /* Button styles */
    .stButton > button {
        background-color: #4CAF50 !important;
        color: white !important;
        border-radius: 10px !important;
        padding: 12px 24px !important;
        font-size: 18px !important;
        font-weight: bold !important;
        border: none !important;
        width: 100% !important;
        transition: all 0.3s ease !important;
        margin: 10px 0 !important;
    }
    .stButton > button:hover {
        background-color: #45a049 !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1) !important;
    }
    
    /* DataFrame styles */
    .stDataFrame {
        border-radius: 10px !important;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1) !important;
    }
    .dataframe {
        font-size: 16px !important;
        color: #000000 !important;
    }
    
    /* Alert styles */
    .stAlert {
        border-radius: 10px !important;
        padding: 20px !important;
        margin: 15px 0 !important;
        font-size: 18px !important;
        font-weight: 500 !important;
    }
    div[data-baseweb="notification"] {
        margin: 15px 0 !important;
        padding: 20px !important;
        border-radius: 10px !important;
        font-size: 18px !important;
    }
    
    /* Success message */
    .success {
        background-color: #d4edda !important;
        color: #155724 !important;
        border: 2px solid #c3e6cb !important;
    }
    
    /* Error message */
    .error {
        background-color: #f8d7da !important;
        color: #721c24 !important;
        border: 2px solid #f5c6cb !important;
    }
    
    /* Warning message */
    .warning {
        background-color: #fff3cd !important;
        color: #856404 !important;
        border: 2px solid #ffeeba !important;
    }
    
    /* Spinner */
    .stSpinner {
        color: #4CAF50 !important;
    }
    
    /* Typography */
    h1 {
        color: #1a1a1a !important;
        font-size: 2.5em !important;
        font-weight: 700 !important;
        margin-bottom: 0.5em !important;
    }
    h2 {
        color: #1a1a1a !important;
        font-size: 1.8em !important;
        font-weight: 600 !important;
        margin-top: 1em !important;
    }
    p {
        color: #1a1a1a !important;
        font-size: 1.1em !important;
        line-height: 1.6 !important;
    }
    
    /* Footer */
    .footer {
        text-align: center !important;
        padding: 20px !important;
        color: #4a4a4a !important;
        font-size: 1em !important;
        margin-top: 30px !important;
    }
    
    /* Labels */
    .css-1offfwp {
        font-size: 16px !important;
        color: #1a1a1a !important;
        font-weight: 500 !important;
    }
    </style>
    """, unsafe_allow_html=True)

class TextPreprocessor:
    def __init__(self):
        self.ps = PorterStemmer()
        self.initialize_nltk()
        
    def initialize_nltk(self):
        """Initialize NLTK components"""
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('punkt')
            nltk.download('stopwords')
    
    def clean_text(self, text: str) -> str:
        """Advanced text cleaning with additional features"""
        try:
            # Convert to lowercase
            text = text.lower()
            
            # Remove URLs
            text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
            
            # Remove email addresses
            text = re.sub(r'\S+@\S+', '', text)
            
            # Remove phone numbers
            text = re.sub(r'\d{3}[-\.\s]??\d{3}[-\.\s]??\d{4}|\(\d{3}\)\s*\d{3}[-\.\s]??\d{4}|\d{10}', '', text)
            
            # Remove special characters and numbers
            text = re.sub(r'[^a-zA-Z\s]', '', text)
            
            # Tokenization
            tokens = nltk.word_tokenize(text)
            
            # Remove stopwords and stem
            stop_words = set(stopwords.words('english'))
            tokens = [self.ps.stem(token) for token in tokens if token not in stop_words]
            
            return " ".join(tokens)
        except Exception as e:
            st.error(f"Error in text cleaning: {str(e)}")
            return ""

class SpamClassifier:
    def __init__(self):
        self.preprocessor = TextPreprocessor()
        self.tfidf, self.model = self.load_models()
        
    def load_models(self) -> Tuple[object, object]:
        """Load the vectorizer and model with error handling"""
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            vectorizer_path = os.path.join(current_dir, 'vectorizer.pkl')
            model_path = os.path.join(current_dir, 'model.pkl')
            
            with open(vectorizer_path, 'rb') as file:
                tfidf = pickle.load(file)
            with open(model_path, 'rb') as file:
                model = pickle.load(file)
            return tfidf, model
        except Exception as e:
            st.error(f"Error loading models: {str(e)}")
            return None, None
    
    def analyze_text(self, text: str) -> Dict:
        """Analyze text and return detailed metrics"""
        try:
            # Clean text
            cleaned_text = self.preprocessor.clean_text(text)
            
            # Get vector representation
            vector = self.tfidf.transform([cleaned_text])
            
            # Get prediction and probabilities
            prediction = self.model.predict(vector)[0]
            proba = self.model.predict_proba(vector)[0]
            
            # Calculate additional metrics
            word_count = len(text.split())
            char_count = len(text)
            unique_words = len(set(text.lower().split()))
            
            # Generate text fingerprint
            text_hash = hashlib.md5(text.encode()).hexdigest()
            
            return {
                'prediction': prediction,
                'confidence': proba[1],
                'word_count': word_count,
                'char_count': char_count,
                'unique_words': unique_words,
                'text_hash': text_hash,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        except Exception as e:
            st.error(f"Error in prediction: {str(e)}")
            return None

class DashboardUI:
    def __init__(self):
        self.classifier = SpamClassifier()
        self.load_custom_theme()
        
    def load_custom_theme(self):
        """Load custom theme and styling"""
        with open('style.css', 'w') as f:
            f.write("""
            /* Your existing CSS here */
            """)
        
        st.markdown("""
            <style>
            /* Your existing CSS here */
            </style>
        """, unsafe_allow_html=True)
    
    def display_metrics(self, metrics: Dict):
        """Display detailed analysis metrics"""
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Word Count", metrics['word_count'])
        with col2:
            st.metric("Character Count", metrics['char_count'])
        with col3:
            st.metric("Unique Words", metrics['unique_words'])
        
        # Display confidence gauge chart
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = metrics['confidence'] * 100,
            title = {'text': "Spam Confidence"},
            gauge = {
                'axis': {'range': [0, 100]},
                'bar': {'color': "#4CAF50" if metrics['confidence'] < 0.5 else "#dc3545"},
                'steps': [
                    {'range': [0, 50], 'color': "#d4edda"},
                    {'range': [50, 100], 'color': "#f8d7da"}
                ]
            }
        ))
        st.plotly_chart(fig, use_container_width=True)
    
    def run(self):
        """Run the main application"""
        st.title("📧 Advanced Email/SMS Analyzer")
        
        st.markdown("""
        <div style='background-color: #f8f9fa; padding: 20px; border-radius: 10px; margin-bottom: 30px;'>
            <p style='font-size: 18px; color: #1a1a1a; margin: 0;'>
                This advanced system uses machine learning and natural language processing to analyze messages.
                It provides detailed metrics and insights about the text content.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Input section
        input_text = st.text_area("Enter your message for analysis:", height=150)
        
        # Initialize session state
        if 'history' not in st.session_state:
            st.session_state.history = []
        
        if st.button('Analyze Text'):
            if not input_text.strip():
                st.warning("Please enter some text to analyze.")
                return
            
            with st.spinner('Performing advanced analysis...'):
                metrics = self.classifier.analyze_text(input_text)
                
                if metrics:
                    # Display results
                    result_style = "error" if metrics['prediction'] == 1 else "success"
                    result_icon = "🚫" if metrics['prediction'] == 1 else "✅"
                    result_text = "Spam Detected" if metrics['prediction'] == 1 else "Clean Message"
                    
                    st.markdown(f"""
                    <div class='stAlert {result_style}' style='text-align: center;'>
                        <span style='font-size: 24px;'>{result_icon} {result_text}</span>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Display detailed metrics
                    self.display_metrics(metrics)
                    
                    # Add to history
                    st.session_state.history.append({
                        'text': input_text[:100] + '...' if len(input_text) > 100 else input_text,
                        'prediction': result_text,
                        'confidence': metrics['confidence'],
                        'timestamp': metrics['timestamp'],
                        'hash': metrics['text_hash']
                    })
        
        # Display history with enhanced visualization
        if st.session_state.history:
            st.markdown("""
            <div style='margin-top: 40px;'>
                <h2 style='color: #1a1a1a; margin-bottom: 20px;'>Analysis History</h2>
            </div>
            """, unsafe_allow_html=True)
            
            history_df = pd.DataFrame(st.session_state.history)
            st.dataframe(history_df, use_container_width=True)
            
            # Advanced visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                # Confidence distribution
                fig1 = px.histogram(history_df, x='confidence',
                                  title='Confidence Score Distribution',
                                  color='prediction',
                                  color_discrete_map={'Spam Detected': '#dc3545', 'Clean Message': '#28a745'})
                st.plotly_chart(fig1, use_container_width=True)
            
            with col2:
                # Prediction timeline
                fig2 = px.line(history_df, x='timestamp', y='confidence',
                              title='Confidence Scores Over Time',
                              color='prediction',
                              color_discrete_map={'Spam Detected': '#dc3545', 'Clean Message': '#28a745'})
                st.plotly_chart(fig2, use_container_width=True)

if __name__ == "__main__":
    app = DashboardUI()
    app.run()