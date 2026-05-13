from fastapi import APIRouter, HTTPException
from app.schemas import AskRequest, AskResponse
from app.rag_service import ask_question

router = APIRouter()

@router.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    try:
        result = ask_question(req.question)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
def health():
    return {"status": "ok"}