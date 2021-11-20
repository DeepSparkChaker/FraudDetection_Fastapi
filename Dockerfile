FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7

RUN pip install --upgrade pip



WORKDIR /app

COPY . /app


RUN pip install -r requirements.txt

#RUN python -m nltk.downloader all

EXPOSE 5000

CMD ["python", "main.py"]