from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import os
from utils import process_file, create_vectorstore, generate_response

app = FastAPI()

# In-memory storage for vector store
vectorstore = None

# Define request model
class QuestionRequest(BaseModel):
    question: str

@app.post("/upload-pdf/")
async def upload_file(file: UploadFile):
    try:
        # Save uploaded file
        file_path = os.path.join("uploaded_files", file.filename)
        os.makedirs("uploaded_files", exist_ok=True)
        content = await file.read()
        
        with open(file_path, "wb") as f:
            f.write(content)
        
        # Process PDF and create vector store
        global vectorstore
        pages, splits = process_file(file_path)
        vectorstore = create_vectorstore(splits)
        
        return JSONResponse(
            content={
                "message": "PDF processed successfully!",
                "pages": len(pages),
                "chunks": len(splits)
            }
        )
    except Exception as e:
        return JSONResponse(
            content={"error": str(e)},
            status_code=500
        )

@app.post("/ask-question/")
async def ask_question(request: QuestionRequest):
    try:
        global vectorstore
        if not vectorstore:
            return JSONResponse(
                content={"error": "No PDF has been uploaded and processed yet."},
                status_code=400
            )
        
        answer = generate_response(vectorstore, request.question)
        return {"answer": answer}
    except Exception as e:
        return JSONResponse(
            content={"error": str(e)},
            status_code=500
        )
