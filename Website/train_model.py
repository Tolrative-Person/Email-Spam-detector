import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import os
import re
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
from typing import List, Tuple
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class AdvancedTextPreprocessor:
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
    
    def preprocess_text(self, text: str) -> str:
        """Advanced text preprocessing pipeline"""
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

class SpamClassifierTrainer:
    def __init__(self):
        self.preprocessor = AdvancedTextPreprocessor()
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 3),
            min_df=2,
            max_df=0.95
        )
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
    
    def prepare_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Prepare training and testing data"""
        # Enhanced sample dataset with more examples
        data = {
            'text': [
                'Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)',
                'I am going to the store. Do you need anything?',
                'URGENT! You have won a 1 week FREE membership in our £100,000 Prize Jackpot! Txt the word: CLAIM to No: 81010',
                'Hey, how are you doing?',
                'Congratulations! You have been selected to win a $1000 gift card! Click here to claim now!',
                'Meeting at 2pm tomorrow in the conference room.',
                'You have won a lottery! Send your bank details to claim the prize!',
                'Please review the attached document and provide feedback.',
                'Get rich quick! Invest now and double your money!',
                'The project deadline has been extended to next Friday.',
                'WINNER!! As a valued network customer you have been selected to receivea £900 prize reward! To claim call 09061701461. Claim code KL341. Valid 12 hours only.',
                'Can you send me the meeting notes from yesterday?',
                'Dear valued customer, your account has been suspended. Click here to reactivate.',
                'Don\'t forget to pick up milk on your way home.',
                'IMPORTANT - You have been pre-approved for a store card with a limit of up to £2000. Call 0800 169 2733 to claim.',
                'Looking forward to seeing you at dinner tonight!',
                'Your PayPal account has been limited! Click here to verify your information.',
                'Great presentation today! The client was very impressed.',
                'URGENT: Your bank account has been compromised. Call this number immediately.',
                'Remember we have a team lunch at 1pm today.'
            ],
            'label': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  # 1 for spam, 0 for not spam
        }
        
        df = pd.DataFrame(data)
        
        # Preprocess all texts
        logging.info("Preprocessing texts...")
        df['processed_text'] = df['text'].apply(self.preprocessor.preprocess_text)
        
        # Split the data
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
        return train_df, test_df
    
    def train_model(self):
        """Train the spam classifier model"""
        try:
            # Prepare data
            train_df, test_df = self.prepare_data()
            
            # Fit vectorizer and transform training data
            logging.info("Vectorizing text data...")
            X_train = self.vectorizer.fit_transform(train_df['processed_text'])
            y_train = train_df['label']
            
            # Transform test data
            X_test = self.vectorizer.transform(test_df['processed_text'])
            y_test = test_df['label']
            
            # Train model
            logging.info("Training model...")
            self.model.fit(X_train, y_train)
            
            # Evaluate model
            logging.info("Evaluating model...")
            train_score = self.model.score(X_train, y_train)
            test_score = self.model.score(X_test, y_test)
            cv_scores = cross_val_score(self.model, X_train, y_train, cv=5)
            
            logging.info(f"Train accuracy: {train_score:.4f}")
            logging.info(f"Test accuracy: {test_score:.4f}")
            logging.info(f"Cross-validation scores: {cv_scores}")
            logging.info(f"Average CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            
            # Detailed classification report
            y_pred = self.model.predict(X_test)
            logging.info("\nClassification Report:")
            logging.info("\n" + classification_report(y_test, y_pred))
            
            # Save models
            self.save_models()
            
        except Exception as e:
            logging.error(f"Error in training: {str(e)}")
            raise
    
    def save_models(self):
        """Save the trained model and vectorizer"""
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            
            # Save vectorizer
            vectorizer_path = os.path.join(current_dir, 'vectorizer.pkl')
            with open(vectorizer_path, 'wb') as f:
                pickle.dump(self.vectorizer, f)
            
            # Save model
            model_path = os.path.join(current_dir, 'model.pkl')
            with open(model_path, 'wb') as f:
                pickle.dump(self.model, f)
            
            logging.info("Model and vectorizer saved successfully!")
            
        except Exception as e:
            logging.error(f"Error saving models: {str(e)}")
            raise

if __name__ == "__main__":
    logging.info("Starting model training process...")
    trainer = SpamClassifierTrainer()
    trainer.train_model() 