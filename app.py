from flask import Flask, request, render_template
import requests
import logging
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import pickle
import nltk

# Download required NLTK data
nltk.download('stopwords')

# Initialize Flask app
app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the trained model and vectorizer
def load_model():
    """
    Load the trained model and vectorizer from the pickle file.
    """
    try:
        with open('trained_model.pkl', 'rb') as file:
            model_data = pickle.load(file)
            model = model_data['model']
            vectorizer = model_data['vectorizer']
        logger.info("Model and vectorizer loaded successfully.")
        return model, vectorizer
    except FileNotFoundError:
        logger.error("Model file not found. Please ensure 'trained_model.pkl' exists.")
        raise
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

# Cache the model and vectorizer
model, vectorizer = load_model()

def preprocess_text(text):
    """
    Preprocess the input text using stemming and stopword removal.
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
        title = request.form.get('title', '').strip()
        author = request.form.get('author', '').strip()
        
        if not title or not author:
            return render_template('index.html', error="Title and author fields cannot be empty.")
        
        # Combine author and title
        news_text = f"{author} {title}"
        
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
        explanation = explain_prediction(processed_text, vectorized_text)
        reliable_sources = get_reliable_sources(title)
        
        return render_template('index.html', 
                             prediction_text=f"The news is {result} (Confidence: {confidence:.2%})",
                             explanation=explanation,
                             reliable_sources=reliable_sources)
    
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        return render_template('index.html', error="An error occurred. Please try again later.")

def explain_prediction(text, vectorized_text):
    """
    Explain the model's prediction using SHAP or LIME.
    """
    # Placeholder for explanation logic
    return "The prediction is based on keywords like 'scam', 'hoax', and 'fake' in the title."

def get_reliable_sources(title):
    """
    Fetch reliable sources or fact-checking articles related to the news title.
    """
    # Placeholder for fetching reliable sources
    return [
        {"source": "FactCheck.org", "url": "https://www.factcheck.org/"},
        {"source": "Snopes", "url": "https://www.snopes.com/"}
    ]

if __name__ == "__main__":
    app.run(debug=True)