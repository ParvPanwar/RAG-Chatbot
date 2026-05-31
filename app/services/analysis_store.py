from uuid import uuid4
from typing import Optional

from app.schemas.analysis import AnalyzeResponse

_analysis_results: dict[str, AnalyzeResponse] = {}


def save_analysis_result(result: AnalyzeResponse) -> str:
    analysis_id = str(uuid4())
    _analysis_results[analysis_id] = result
    return analysis_id


def get_analysis_result(analysis_id: str) -> Optional[AnalyzeResponse]:
    return _analysis_results.get(analysis_id)
