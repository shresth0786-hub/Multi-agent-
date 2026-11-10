from typing import List, Dict, Any, TypedDict, Annotated
import operator

class ResearchState(TypedDict, total=False):
    query: str
    problem: str  # User's specific problem/context - fed to agents for tailored report
    paper_metadata: List[Dict[str, Any]]
    analysis_results: List[Dict[str, Any]]
    comparison_matrix: Dict[str, Any]
    critique_findings: Dict[str, Any]
    final_report: str
    _stats: Dict[str, Any]
