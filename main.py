import uvicorn

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.embeddings import extract_embedding_from_bytes
from app.similarity import find_similar_wealthy_individuals, estimate_net_worth_from_similars


app = FastAPI(title="Wealth Potential Estimator API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "Wealth Potential Estimator API is up and running!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")

    image_bytes = await file.read()

    try:
        embedding = extract_embedding_from_bytes(image_bytes)
    except ValueError as ve:
        return JSONResponse(
            status_code=400,
            content={"detail": str(ve)}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

    similar_people = find_similar_wealthy_individuals(embedding, top_k=3)
    estimated_net_worth = estimate_net_worth_from_similars(similar_people)

    response = {
        "estimated_net_worth_in_million": round(estimated_net_worth, 0),
        "similar_individuals": [
            {"name": name, "similarity_score": round(similarity, 4)}
            for name, similarity in similar_people
        ],
    }
    return response

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=80, proxy_headers=True)