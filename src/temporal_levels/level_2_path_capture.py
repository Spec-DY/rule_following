"""
Temporal Level 2: Path Blocked Capture
Tests capture with path blocking for Rook, Bishop, Queen
"""

from typing import List, Dict
from .temporal_level_base import TemporalLevelBase
from .level_2_generator import Level2Generator


class TemporalLevel2(TemporalLevelBase):
    """Level 2: Path blocked capture (Rook, Bishop, Queen)"""

    def __init__(self,
                 base_output_dir: str = "./output/temporal_level_2",
                 n_cases: int = 90,
                 seed: int = 42,
                 auto_timestamp: bool = True,
                 rate_limit_requests: int = 0,
                 rate_limit_pause: int = 0):
        """
        Initialize Temporal Level 2

        Args:
            base_output_dir: Base directory for output files
            n_cases: Total number of test cases
            seed: Random seed for reproducibility
            auto_timestamp: If True, append timestamp to output directory
            rate_limit_requests: Number of requests before pausing
            rate_limit_pause: Seconds to pause
        """
        super().__init__(
            level=2,
            base_output_dir=base_output_dir,
            n_cases=n_cases,
            seed=seed,
            auto_timestamp=auto_timestamp,
            rate_limit_requests=rate_limit_requests,
            rate_limit_pause=rate_limit_pause
        )

    def generate_test_cases(self) -> List[Dict]:
        """Generate test cases automatically"""
        print(
            f"\nGenerating Level 2 test cases (n_cases={self.n_cases}, seed={self.seed})")
        print("=" * 60)

        generator = Level2Generator(seed=self.seed)
        cases = generator.generate_all(n_cases=self.n_cases)

        # Add verification questions to each case
        for case in cases:
            verification_info = self.verification_gen.generate_verification(
                case)
            case.update(verification_info)

        self.test_cases = cases
        return cases
