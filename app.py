from flask import Flask, request, render_template
import pickle
import numpy as np
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk
import re

# Download required NLTK data
nltk.download('stopwords')

app = Flask(__name__)

# Load the trained model and vectorizer
try:
    with open('trained_model.pkl', 'rb') as file:
        model_data = pickle.load(file)
        model = model_data['model']
        vectorizer = model_data['vectorizer']
except FileNotFoundError:
    raise Exception("Model file not found. Please ensure 'trained_model.pkl' exists.")
except Exception as e:
    raise Exception(f"Error loading model: {str(e)}")

def preprocess_text(text):
    """
    Preprocess the input text using stemming and stopword removal
    """
    port_stem = PorterStemmer()
    # Remove special characters and numbers
    text = re.sub('[^a-zA-Z]', ' ', text)
    # Convert to lowercase
    text = text.lower()
    # Split into words
    words = text.split()
    # Remove stopwords and apply stemming
    words = [port_stem.stem(word) for word in words if word not in stopwords.words('english')]
    # Join words back together
    return ' '.join(words)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input values
        title = request.form['title']
        author = request.form['author']
        
        # Combine author and title
        news_text = author + ' ' + title
        
        # Preprocess the text
        processed_text = preprocess_text(news_text)
        
        # Vectorize using the fitted vectorizer
        vectorized_text = vectorizer.transform([processed_text])
        
        # Make prediction
        prediction = model.predict(vectorized_text)
        prediction_proba = model.predict_proba(vectorized_text)[0]
        
        # Get confidence score
        confidence = prediction_proba[1] if prediction[0] == 1 else prediction_proba[0]
        
        # Prepare output
        result = "FAKE" if prediction[0] == 1 else "REAL"
        output = f"The news is {result} (Confidence: {confidence:.2%})"
        
        return render_template('index.html', prediction_text=output)
    
    except Exception as e:
        return render_template('index.html', 
                             prediction_text=f"An error occurred: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)