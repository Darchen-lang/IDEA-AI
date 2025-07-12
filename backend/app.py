from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware 
import uvicorn
import os 
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
from pydantic import BaseModel

from predict_ideai import IdeaFeasibilityPredictor


from predict_ideai import IdeaFeasibilityPredictor 


class IdeaRequest(BaseModel):
    idea_text: str


app = FastAPI(
    title="IdeAI Feasibility Predictor API",
    description="API to predict the feasibility score of an idea and provide suggestions.",
    version="1.0.0"
)


origins = [
    "http://localhost",
    "http://localhost:8001", 
    "file://", 
    "file:///",
      "http://localhost:3000",   
    "http://127.0.0.1:3000",   
    "*"                     
   
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)



predictor: IdeaFeasibilityPredictor = None


@app.on_event("startup")
async def startup_event():
    global predictor
    try:
        model_path = "./trained_model" 
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model directory not found: {model_path}")

        predictor = IdeaFeasibilityPredictor(model_path=model_path)
        print(f"Model and tokenizer loaded from: {model_path} on device: {predictor.device}")
        print("AI model loaded successfully and is ready for predictions!")
    except Exception as e:
        print(f"ERROR: Failed to load AI model. Details: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load AI model. Check server logs. Error: {e}")


@app.post("/predict_feasibility/")
async def predict_feasibility(request: IdeaRequest):
    if predictor is None:
        raise HTTPException(status_code=500, detail="AI model is not loaded. Server startup failed.")

    try:
        
        score = predictor.predict(request.idea_text)
        suggestion = predictor.generate_suggestions(score)

        return {
            "idea_text": request.idea_text,
            "predicted_feasibility_score": score,
            "suggestion": suggestion
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed due to an internal error: {e}")


@app.get("/")
async def read_root():
    return {"message": "Welcome to IdeAI. Use /predict_feasibility/ to get idea scores."}


if __name__ == "__main__":
    print("Starting FastAPI server...")
    print("Access API documentation at: http://127.0.0.1:8000/docs")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)