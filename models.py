from pydantic import BaseModel
from typing import Dict, Any, Optional

class AgenticError(BaseModel):
    error: str
    code: str
    message: str
    suggestion: Optional[str] = None
    retry_after: Optional[int] = None
    _links: Optional[Dict] = None

class HATEOASResponse(BaseModel):
    data: Any
    _links: Dict = {}
    warnings: list = []
