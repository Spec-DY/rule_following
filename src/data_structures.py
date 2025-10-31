"""
Data structures for test results and configurations
"""

import os
import json
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
from datetime import datetime
from enum import Enum


class TestType(Enum):
    SPATIAL = "spatial"
    TEMPORAL = "temporal"


class PieceType(Enum):
    KNIGHT = "knight"
    BISHOP = "bishop"
    ROOK = "rook"


@dataclass
class TestResult:
    """Test result data class"""
    # Basic information
    test_type: str  # "spatial" or "temporal"
    test_layer: int  # 0, 1, 2, 3
    case_id: str

    # Test specific
    piece_type: Optional[str] = None  # for spatial
    rule_type: Optional[str] = None  # for temporal

    # === Recognition verification (NEW) ===
    verification_question: Optional[str] = None
    verification_expected: Optional[str] = None
    verification_response: Optional[str] = None
    verification_passed: Optional[bool] = None

    # Question and answer
    question: str = ""
    expected_answer: str = ""  # "yes", "no", "unknown"
    model_response: str = ""
    correct: bool = False

    # Test 2 specific (Know-Do Gap)
    declarative_question: Optional[str] = None
    declarative_response: Optional[str] = None
    declarative_correct: Optional[bool] = None
    know_do_gap: Optional[bool] = None

    # Test 3 specific (Explicit Rule)
    condition: Optional[str] = None  # "without_rule", "with_rule"

    # Metadata
    image_paths: List[str] = None
    timestamp: str = ""
    model_name: str = ""

    def __post_init__(self):
        if self.image_paths is None:
            self.image_paths = []
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict):
        return cls(**data)


def save_results(results: List[TestResult], output_path: str):
    """Save test results to JSON file"""
    results_dict = [r.to_dict() for r in results]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results_dict, f, indent=2)

    print(f"Results saved to: {output_path}")


def load_results(input_path: str) -> List[TestResult]:
    """Load test results from JSON file"""
    with open(input_path, 'r') as f:
        results_dict = json.load(f)

    return [TestResult.from_dict(r) for r in results_dict]
