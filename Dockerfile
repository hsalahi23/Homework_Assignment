FROM python:3.12

WORKDIR /wealth_estimator

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ./app ./app
COPY ./data ./data
COPY main.py .

EXPOSE 8080

CMD ["python", "main.py"]
