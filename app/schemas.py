from pydantic import BaseModel
from typing import List, Dict

class AskRequest(BaseModel):
    question: str

class AskResponse(BaseModel):
    answer: str
    sources: List[Dict]
    evaluation: Dict