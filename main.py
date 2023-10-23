from fastapi import FastAPI, HTTPException
import joblib
from Tools.DatasetPreprocess import CleanString
from pydantic import BaseModel

# Create a FastAPI app instance with metadata
app = FastAPI(
    title='Launching NLP model',
    description='Launching a Natural Language Processing Sentiment model to predict sentiment on film reviews',
    version='0.1.0'
)

# Define input and output data models using Pydantic to verify data types
class InputData(BaseModel):
    text: str

class PredictionResult(BaseModel):
    prediction: str

# Load the pre-trained NLP model and TF-IDF vectorizer
model = joblib.load('MBD_NLP_model.joblib')
tfidf = joblib.load('tfidf_vectorizer.joblib')

# Define a root endpoint for a simple welcome message
@app.get('/')
def read_root():
    return {'message': 'Welcome to the NLP sentiment model API'}

# Define a prediction endpoint for making sentiment predictions
@app.post('/predict/', response_model=PredictionResult)
def predict(input_data: InputData):
    try:
        # Preprocess the input text
        cleaned_text = CleanString(input_data.text)

        # Transform the text using the TF-IDF vectorizer
        vectorized_text = tfidf.transform([cleaned_text])

        # Make a sentiment prediction
        prediction = model.predict(vectorized_text)[0]

        # Convert the prediction to human-readable text
        if prediction == 0:
            prediction_text = 'The sentiment of this text is Negative'
        else:
            prediction_text = 'The sentiment of this text is Positive'

        # Return the prediction result
        return {'prediction': prediction_text}
    
    except:
        # Handle exceptions by raising a 500 Internal Server Error
        raise HTTPException(status_code=500, detail='Internal server error')
