
![Screenshot 2025-05-04 170102](https://github.com/user-attachments/assets/7e0f867e-ab37-4772-9cd5-d2a165a6904a)



# Advanced Email/SMS Spam Classifier

A sophisticated machine learning application that analyzes and classifies text messages as spam or not spam, with advanced features and detailed analytics.

## Features

- **Advanced Text Analysis**: Uses natural language processing to clean and analyze text
- **Real-time Classification**: Instantly classifies messages as spam or not spam
- **Detailed Metrics**: Provides word count, character count, and unique word analysis
- **Visual Analytics**: Interactive charts showing confidence scores and prediction history
- **Advanced Preprocessing**: Removes URLs, email addresses, phone numbers, and special characters
- **Modern UI**: Clean, responsive interface with real-time feedback

## Technical Details

- **Model**: Random Forest Classifier with optimized hyperparameters
- **Text Processing**: Advanced NLP pipeline with stemming and stopword removal
- **Vectorization**: TF-IDF with n-gram features (1-3)
- **Visualization**: Interactive charts using Plotly
- **Frontend**: Streamlit with custom CSS styling

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/email-spam-classifier.git
cd email-spam-classifier
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Train the model:
```bash
python Website/train_model.py
```

4. Run the application:
```bash
python -m streamlit run Website/app.py
```

## Project Structure

```
email-spam-classifier/
├── Website/
│   ├── app.py              # Main application
│   ├── train_model.py      # Model training script
│   ├── model.pkl           # Trained model
│   └── vectorizer.pkl      # Text vectorizer
├── requirements.txt        # Project dependencies
└── README.md              # Project documentation
```

## Usage

1. Enter your text in the input area
2. Click "Analyze Text" to process the message
3. View the classification results and detailed metrics
4. Check the history section for past analyses

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- NLTK for natural language processing
- Scikit-learn for machine learning
- Streamlit for the web interface
- Plotly for data visualization

## 📧 Contact

For questions or suggestions, please open an issue in the repository.
