from fastapi import FastAPI, HTTPException
import joblib
from Tools.DatasetPreprocess import CleanString
from pydantic import BaseModel


app = FastAPI(
    title = 'Launching NLP model',
    description = 'Launching a Natural Language Processing Sentiment model to predict sentiment on film reviews',
    version = '0.1.0'
)

class InputData(BaseModel):
    text: str

class PredictionResult(BaseModel):
    prediction: str

model = joblib.load('MBD_NLP_model.joblib')
tfidf = joblib.load('tfidf_vectorizer.joblib')


@app.get('/')
def read_root():
    return {'message': 'Welcome to the NLP sentiment model API'}

@app.post('/predict/', response_model=PredictionResult)
def predict(input_data: InputData):
    try:
        cleaned_text = CleanString(input_data.text)
        vectorized_text = tfidf.transform([cleaned_text])
        # if not vectorized_text:
        #     raise HTTPException(status_code=400, detail='Invalid input data')
        prediction = model.predict(vectorized_text)[0]
        if prediction == 0:
            prediction = 'The sentiment of this text is Negative'
        else:
            prediction = 'The sentiment of this text is Positive'
        return {'prediction': prediction}

    
    except:
        raise HTTPException(status_code=500, detail='Internal server error')


