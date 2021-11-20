# First, we will need to import the library and initialize the main application object:
import joblib
import uvicorn
from fastapi import FastAPI,Request
from pydantic import BaseModel
import pandas as pd
#import nest_asyncio
from typing import List
#from typing import Any, Dict,List,Enum
# import numpy as np     
## API INSTANTIATION
## ----------------------------------------------------------------
       
app = FastAPI(
    title="Fraud Detection API",
    description="A simple API that use Ml model to predict fraud ",
    version="0.1",
)
# Creating the data model for data validation
class ClientData(BaseModel):
    step: List[int]
    type: List[str]
    amount: List[float]
    nameOrig:  List[str]
    oldbalanceOrig: List[float]
    newbalanceOrig: List[float]
    nameDest:  List[str]
    oldbalanceDest: List[float]
    newbalanceDest: List[float]

# Load  the model  a serialized .joblib file
joblib_filename="models/lgbm_localV1.joblib"
model = joblib.load(joblib_filename)

## API ENDPOINTS
## ----------------------------------------------------------------

##################
@app.get('/')
def index():
  '''
  This is a first docstring.
  '''
  return {'message': 'This is a Fraud  Classification API!'}

# Tester
@app.get('/ping')
def ping():
  '''
  This is a first docstring.
  '''
  return ('pong', 200)
# Defining the prediction endpoint without data validation
@app.post('/basic_predict')
async def basic_predict(request: Request):
  '''
  This is a first docstring.
  '''
  # Getting the JSON from the body of the request
  input_data = await request.json()

    # Converting JSON to Pandas DataFrame
  input_df = pd.DataFrame([input_data])

  # Getting the prediction 
  pred = model.predict(input_df)[0]

  return pred

# We now define the function that will be executed for each URL request and return the value:
@app.post("/predict-fraud")
async  def predict_fraud(item :ClientData):
  """
  A simple function that receive a client data and predict Fraud.
  :param client_data:
  :return: prediction, probabilities
  
  """
  # perform prediction
  #df =pd.DataFrame([item])
  h=item.dict()
  df=pd.DataFrame.from_dict(h, orient="columns").reset_index()
  prediction = model.predict(df)
  prediction_final=["Fraud" if (x > 0.5) else "Not Fraud" for x in prediction ]
  return prediction_final

if __name__ == "__main__":
  uvicorn.run(app, host="0.0.0.0", port=5000)
# uvicorn app:app --reload