"""
Temporal Level 5: Castling + 2 Check Rules
Tests castling with 2 out of 3 check-related constraints
"""

from typing import List, Dict
from .temporal_level_base import TemporalLevelBase
from .level_5_generator import Level5Generator


class TemporalLevel5(TemporalLevelBase):
    """Level 5: Castling with 2 check rules"""

    def __init__(self,
                 base_output_dir: str = "./output/temporal_level_5",
                 n_cases: int = 100,
                 seed: int = 42,
                 auto_timestamp: bool = True,
                 rate_limit_requests: int = 0,
                 rate_limit_pause: int = 0):
        """
        Initialize Temporal Level 5

        Args:
            base_output_dir: Base directory for output files
            n_cases: Total number of test cases
            seed: Random seed for reproducibility
            auto_timestamp: If True, append timestamp to output directory
            rate_limit_requests: Number of requests before pausing
            rate_limit_pause: Seconds to pause
        """
        super().__init__(
            level=5,
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
            f"\nGenerating Level 5 test cases (n_cases={self.n_cases}, seed={self.seed})")
        print("=" * 60)

        generator = Level5Generator(seed=self.seed)
        cases = generator.generate_all(n_cases=self.n_cases)

        # Add verification questions to each case
        for case in cases:
            verification_info = self.verification_gen.generate_verification(
                case)
            case.update(verification_info)

        self.test_cases = cases
        return cases
