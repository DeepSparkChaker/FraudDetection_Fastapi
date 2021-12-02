import requests as r
# add transactions
review ={
"step":50,
"type":"PAYMENT",
"amount":9839.64,
"nameOrig":"C1231006815",
"oldbalanceOrig":170136.0,
"newbalanceOrig":160296.36,
"nameDest":"M1979787155",
"oldbalanceDest":0.0,
"newbalanceDest":0.0
}

test2={"step":234,
       "type":"CASH_OUT",
       "amount":305822.52,
       "nameOrig":"C1376293938",
       "oldbalanceOrig":0.0,
       "newbalanceOrig":0.0,
       "nameDest":"C182325611",
       "oldbalanceDest":1569390.8999999999,
       "newbalanceDest":1875213.4199999999}

test3={
  "step": [
    0,234
  ],
  "type": [
    "string","CASH_OUT"
  ],
  "amount": [
    0,305822.52
  ],
  "nameOrig": [
    "string","C1376293938"
  ],
  "oldbalanceOrig": [
    0,0.0
  ],
  "newbalanceOrig": [
    0,0.0
  ],
  "nameDest": [
    "string","C182325611"
  ],
  "oldbalanceDest": [
    0,1569390.8999999999
  ],
  "newbalanceDest": [
    0,1875213.4199999999
  ]
}
keys = {"request": test1}
keys2 = {"item": test2}
#prediction = r.get("http://127.0.0.1:4000/'chaka'")
prediction = r.post("http://127.0.0.1:4000/predict-fraud",  json=test3)
results=prediction.json()
print(results )