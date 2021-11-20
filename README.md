# TL; DR:

<<<<<<< HEAD
In this article, we are going to build a # FraudDetection_Fastapi
this api will detect fraud 

=======
In this article, we are going to build a # FraudDetection_Fastapi
this api will detect fraud 

>>>>>>> 726fe96c67bafaeb984f22813a40b7e9bbb0d08c
 Using FastAPI create REST API calls to predict from the model, and finally containerize it using Docker 😃
I want to emphasize the usage of FastAPI and how rapidly this framework is a game-changer for building easy to go and much faster API calls for a machine learning pipeline.
Traditionally, we have been using Flask Microservices for building REST API calls but the process involves a bit of nitty-gritty to understand the framework and implement it.
On the other end, I found FastAPI to be pretty user-friendly and very easy to pick up and implement type of framework.
And finally from one game-changer to another, Docker
As a data scientist: our role is vague and it keeps on changing year in year out. Some skillset gets added, some get extinct and obsolete, but Docker has made its mark as one of the very important and most sought out skills in the market. Docker gives us the ability to containerize a solution with all its binding software and requirements.
# The Data

We have used a text classification problem : IMDb Dataset for the purpose of building the model.
The dataset comprises 50,000 reviews of movies and is a binary classification problem with the target variable being a sentiment: positive or negative.
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

# Docker
Finally, to wrap it all up, we create a Dockerfile :

We have attached a docker container (tiangolo/uvicorn-gunicorn-fastapi) which is made public on docker-hub, which makes quick work of creating a docker image on our own functionalities.

To create a docker image and deploy it, we run the following commands, and voila!

docker build -t api .

docker run -d -p 5000:5000 api

<<<<<<< HEAD
#
http://192.168.99.100:5000/docs#/default/predict_fraud_predict_fraud_post

=======
#
http://192.168.99.100:5000/docs#/default/predict_fraud_predict_fraud_post

>>>>>>> 726fe96c67bafaeb984f22813a40b7e9bbb0d08c
 Conclusion:

After going through the process of working around FastAPI and Docker, I feel this skillset is a necessary repertoire in a data scientist's toolkit. The process of building around our model and deploying it has become easier and much more accessible than it was before.
# Reference:
https://www.youtube.com/watch?v=7t2alSnE2-I

https://dev.to/kushalvala/tensorflow-model-deployment-using-fastapi-docker-4183

https://nickc1.github.io/api,/scikit-learn/2019/01/10/scikit-fastapi.html

https://github.com/akshaykhatale/fastapi-heroku-docker

https://www.freecodecamp.org/news/how-to-deploy-an-nlp-model-with-fastapi/

https://github.com/kaustubhgupta/FastAPI-KivyMD-App-Demo

https://www.analyticsvidhya.com/blog/2021/06/deploying-ml-models-as-api-using-fastapi-and-heroku/



