
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from core.rag_engine import embed_and_store_pdf, query_rag
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "data/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

class Question(BaseModel):
    question: str

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    file_path = os.path.join(UPLOAD_DIR, file.filename)
    try:
        with open(file_path, "wb") as f:
            f.write(await file.read())
        print(f"[INFO] File saved to: {file_path}")

        success = embed_and_store_pdf(file_path)
        if not success:
            print("[ERROR] Failed to embed and store PDF.")
            raise HTTPException(status_code=500, detail="Failed to process PDF.")
        return {"message": "PDF uploaded and processed successfully."}
    except Exception as e:
        print(f"[EXCEPTION] Upload failed: {e}")
        raise HTTPException(status_code=500, detail="Upload failed.")


@app.post("/chat")
async def chat(question: Question):
    answer, sources = query_rag(question.question)
    return {
        "answer": answer,
        "sources": sources
    }
