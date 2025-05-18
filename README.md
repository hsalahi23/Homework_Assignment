# Wealth Potential Estimator API
A mock API that takes a selfie image and returns:

- An estimated net worth

- The top 3 most similar wealthy individuals and similarity scores

## Quick Start

### Prerequisites

Before you can use this API, make sure you have the following prerequisites installed on your system:

- Python 3.9+
- virtualenv (for creating virtual environments)


#### Create virtual environment 
```bash
cd <path to the project>
python3 -m venv venv
```

#### Activate virtual environment
```bash
source venv/bin/activate
```

### Installation
Install the required Python packages:

```bash
pip install -r requirements.txt
```

## How to Run Locally
1. Run the API server
```bash
python main.py
```
2. Open the Swagger UI
Go to http://localhost:8000/docs

3. Use the /predict endpoint
Upload a selfie image (.jpg, .png)

Receive estimated net worth and similar individuals

## Example API Response

{
  "estimated_net_worth": 3150000.75,
  "similar_individuals": [
    {"name": "Wealthy Person 8", "similarity_score": 0.912},
    {"name": "Wealthy Person 23", "similarity_score": 0.874},
    {"name": "Wealthy Person 68", "similarity_score": 0.823}
  ]
}


## How to Use the /predict Endpoint
Once the API is running (either locally or on your deployed AWS EC2 server), you can make predictions by sending a selfie image via a POST request to the /predict endpoint.

### Endpoint URL
Local: http://0.0.0.0:80/predict

Deployed API: http://52.60.71.197:80/predict

#### Request Format
- Method: POST

- Content-Type: multipart/form-data

- Field: "file" – your image file (selfie)

### Sample Usage in Python
```python
import requests

url = "http://0.0.0.0:80/predict"  # Replace with EC2 URL if deployed
image_path = "path_to_image/selfie.jpg"

with open(image_path, "rb") as image_file:
    files = {"file": (image_path, image_file, "image/jpeg")}
    response = requests.post(url, files=files)

print(response)
if response.status_code == 200:
    print("Response JSON:", response.json())
else:
    print("Error:", response.status_code, response.text)
```

## Overview

### How it works:
- Upload a selfie via the /predict endpoint.

- Face is detected and embedded using FaceNet + MTCNN.

- System compares the face embedding to a mock dataset of wealthy individuals.

- The top 3 most similar individuals are identified.

- The user's estimated net worth is calculated using a weighted average of those individuals' net worths.


## Architecture Decisions
### Face Embedding
- Model: Pre-trained FaceNet from facenet-pytorch

- Detection: MTCNN detects and crops faces reliably

- Embedding Size: 512-dimensional vectors

### Similarity Computation
- Metric: Cosine similarity between embeddings

## Net Worth Estimation
- Takes the top 3 most similar individuals

- Estimates net worth using weighted average based on similarity scores
- Formula: 
```text
estimated = sum(score_i * net_worth_i) / sum(score_i)
```
## Assumptions
Wealthy individuals are defined as people in a mock dataset containing:

- A unique name

- 512-dimensional face embedding

- A fictional net worth (in millions USD dollar)

- Selfie quality and face detection are expected to be decent (no multi-face support).

**NOTE**: The model has no real predictive power — this is a mock, demonstration-only tool.



## Limitations & Disclaimer
This API is not predictive. it’s a technical demo with made-up data.

Intended for Spring Financial inc. homework assignment.

## Author

- [Hamid Salahi](https://www.linkedin.com/in/hamidreza-salahi/)