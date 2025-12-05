"""
Temporal Test 0: Pure Temporal Reasoning
Tests basic temporal understanding without chess rules
With per-case verification
"""

from typing import List, Dict
from .temporal_test_base import TemporalTestBase
from .test_0_generator import TemporalTest0Generator


class TemporalTest0(TemporalTestBase):
    """Test 0: Pure temporal understanding (no chess rules)"""

    def __init__(self,
                 base_output_dir: str = "./output/temporal_test_0",
                 n_cases_per_type: int = 10,
                 seed: int = 42,
                 auto_timestamp: bool = True,
                 rate_limit_requests: int = 0,
                 rate_limit_pause: int = 0):
        """
        Initialize Temporal Test 0

        Args:
            base_output_dir: Base directory for output files
            n_cases_per_type: Number of cases per test type
            seed: Random seed for reproducibility
            auto_timestamp: If True, append timestamp to output directory
        """
        super().__init__(
            test_layer=0,
            base_output_dir=base_output_dir,
            n_cases_per_type=n_cases_per_type,
            seed=seed,
            auto_timestamp=auto_timestamp,
            rate_limit_requests=rate_limit_requests,
            rate_limit_pause=rate_limit_pause
        )

    def generate_test_cases(self) -> List[Dict]:
        """Generate test cases automatically"""
        print(
            f"\nGenerating test cases (n_per_type={self.n_cases_per_type}, seed={self.seed})")
        print("="*60)

        generator = TemporalTest0Generator(seed=self.seed)
        cases = generator.generate_all(n_per_type=self.n_cases_per_type)

        # Add verification questions to each case
        for case in cases:
            verification_info = self.verification_gen.generate_verification(
                case)
            case.update(verification_info)

        self.test_cases = cases
        return cases
