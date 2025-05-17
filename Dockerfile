FROM python:3.11

WORKDIR /wealth_estimator

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY ./app ./app
COPY ./data ./data
COPY main.py .

EXPOSE 8585

CMD ["python", "main.py"]
