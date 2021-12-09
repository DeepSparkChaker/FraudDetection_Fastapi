# TL; DR:

<<<<<<< HEAD
In this article, we are going to build a # FraudDetection_Fastapi
this api will detect fraud 
Due to rapid growth in field of cashless or digital transactions, credit cards are widely used in all around the world. Credit cards providers are issuing thousands of cards to their customers. Providers have to ensure all the credit card users should be genuine and real. Any mistake in issuing a card can be reason of financial crises. Due to rapid growth in cashless transaction, the chances of number of fraudulent transactions can also increasing. A Fraud transaction can be identified by analyzing various behaviors of credit card customers from previous transaction history datasets. If any deviation is noticed in spending behavior from available patterns, it is possibly of fraudulent transaction. Data mining and machine learning techniques are widely used in credit card fraud detection. I am presenting review of various data mining and machine learning methods which are widely used for credit card fraud detections and complete this project end to end from Data Understanding to deploy Model via API .

In this section we overview our selected method for engineering our solution. CRISP-DM stands for Cross-Industry Standard Process for Data Mining. It is an open standard guide that describes common approaches that are used by data mining experts. CRISP-DM includes descriptions of the typical phases of a project, including tasks details and provides an overview of the data mining lifecycle. The lifecycle model consists of six phases with arrows indicating the most important and frequent dependencies between phases.

1-You will find my complete work for Ml "Fraud detection" by following those notebooks :

https://www.kaggle.com/bannourchaker/frauddetection-part1-eda

https://www.kaggle.com/bannourchaker/frauddetection-part2-preparation

https://www.kaggle.com/bannourchaker/frauddetection-part3-modeling1-cross-validation

https://www.kaggle.com/bannourchaker/frauddetection-part3-modeling2-selectbestmodel

https://www.kaggle.com/bannourchaker/frauddetection-part3-modeling3-tuning

https://www.kaggle.com/bannourchaker/frauddetection-part4-evaluation

https://www.kaggle.com/bannourchaker/frauddetection-part4-explainai

2- Deploy the models via api :

-Baseline pipeline : https://github.com/DeepSparkChaker/FraudDetection_Fastapi

-Advanced pipeline: https://github.com/DeepSparkChaker/FraudDetection_Fastapi_VF
I hope it's a good and useful guide.

=======
In this article, we are going to build a # FraudDetection_Fastapi
this api will detect fraud 

>>>>>>> 

 Using FastAPI create REST API calls to predict from the model, and finally containerize it using Docker 😃
I want to emphasize the usage of FastAPI and how rapidly this framework is a game-changer for building easy to go and much faster API calls for a machine learning pipeline.
Traditionally, we have been using Flask Microservices for building REST API calls but the process involves a bit of nitty-gritty to understand the framework and implement it.
On the other end, I found FastAPI to be pretty user-friendly and very easy to pick up and implement type of framework.
And finally from one game-changer to another, Docker
As a data scientist: our role is vague and it keeps on changing year in year out. Some skillset gets added, some get extinct and obsolete, but Docker has made its mark as one of the very important and most sought out skills in the market. Docker gives us the ability to containerize a solution with all its binding software and requirements.
# The Data

We have used a Fraud problem : https://www.kaggle.com/bannourchaker/frauddetection  Dataset for the purpose of building the model.
The dataset comprises 6341907 Transactions  and is a binary classification problem with the target variable being a Fraud: Fraud or not Fraud.
# Project 
|
|--- model

|    |______ model.pkl

|
|--- app

|    |_______ main.py

|

|--- Dockerfile

#  FastAPI


FastAPI is a modern, high-performance, batteries-included Python web framework that's perfect for building RESTful APIs. It can handle both synchronous and asynchronous requests and has built-in support for data validation, JSON serialization, authentication and authorization, and OpenAPI.

Highlights:

    Heavily inspired by Flask, it has a lightweight microframework feel with support for Flask-like route decorators.
    It takes advantage of Python type hints for parameter declaration which enables data validation (via pydantic) and OpenAPI/Swagger documentation.
    Built on top of Starlette, it supports the development of asynchronous APIs.
    It's fast. Since async is much more efficient than the traditional synchronous threading model, it can compete with Node and Go with regards to performance.


# Docker

Finally, to wrap it all up, we create a Dockerfile :

We have attached a docker container (tiangolo/uvicorn-gunicorn-fastapi) which is made public on docker-hub, which makes quick work of creating a docker image on our own functionalities.

To create a docker image and deploy it, we run the following commands, and voila!

docker build -t api .

docker run -d -p 5000:5000 api

Why Docker? We want to minimize the differences between the production and development environments.
This is especially important with this project, since it relies on a number of data science dependencies that have very specific system requirements.

<<<<<<< HEAD
#
http://192.168.99.100:5000/docs#/default/predict_fraud_predict_fraud_post

=======
#
http://192.168.99.100:5000/docs#/default/predict_fraud_predict_fraud_post

Conclusion:

After going through the process of working around FastAPI and Docker, I feel this skillset is a necessary repertoire in a data scientist's toolkit.

The process of building around our model and deploying it has become easier and much more accessible than it was before.
# Reference:
end to end Ml :
https://neptune.ai/blog/how-to-monitor-your-models-in-production-guide
https://pythonawesome.com/a-complete-end-to-end-machine-learning-portal-that-covers-processes-starting-from-model-training-to-the-model-predicting-results-using-fastapi/

https://medium.com/analytics-vidhya/fundamentals-of-mlops-part-4-tracking-with-mlflow-deployment-with-fastapi-61614115436

Ml Monitoring : 
https://neptune.ai/blog/how-to-monitor-your-models-in-production-guide
https://neptune.ai/blog/best-ml-experiment-tracking-tools


https://www.kaggle.com/bannourchaker/frauddetection/code?datasetId=1719423&sortBy=dateRun&tab=profile

https://www.youtube.com/watch?v=7t2alSnE2-I

https://dev.to/kushalvala/tensorflow-model-deployment-using-fastapi-docker-4183

https://nickc1.github.io/api,/scikit-learn/2019/01/10/scikit-fastapi.html

https://github.com/akshaykhatale/fastapi-heroku-docker

https://www.freecodecamp.org/news/how-to-deploy-an-nlp-model-with-fastapi/

https://github.com/kaustubhgupta/FastAPI-KivyMD-App-Demo

https://www.analyticsvidhya.com/blog/2021/06/deploying-ml-models-as-api-using-fastapi-and-heroku/



# FraudDetection_V2_Fastapi
# FraudDetection_Fastapi_V2
