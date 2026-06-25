from pydantic import BaseModel
from typing import List, Union

class JinaRerankerResult(BaseModel):
    index: int
    relevance_score: float
    document: Union[str, dict, None] = None

class JinaRerankerResponse(BaseModel):
    model: str
    usage: dict
    results: List[JinaRerankerResult]

class JinaRerankerRequest(BaseModel):
    query: str
    documents: List[Union[str, dict]]
    model: str | None = None
    top_n: int | None = None
    return_documents: bool = True
    batch_size: int | None = None
