"""
Spatial Test 1: Rule Following Baseline
Tests chess rule application ability (all 6 piece types)
With per-case board recognition verification
"""

from typing import List, Dict
from .spatial_test_base import SpatialTestBase
from .test_1_generator import SpatialTest1Generator


class SpatialTest1(SpatialTestBase):
    """Test 1: Chess rule following (all piece types)"""

    def __init__(self,
                 base_output_dir: str = "./output/spatial_test_1",
                 n_cases_per_type: int = 10,
                 seed: int = 42,
                 auto_timestamp: bool = True,
                 rate_limit_requests: int = 0,
                 rate_limit_pause: int = 0):
        """
        Initialize Spatial Test 1

        Args:
            base_output_dir: Base directory for output files
            n_cases_per_type: Number of cases per piece type
            seed: Random seed for reproducibility
            auto_timestamp: If True, append timestamp to output directory
        """
        super().__init__(
            test_layer=1,
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

        generator = SpatialTest1Generator(seed=self.seed)
        cases = generator.generate_all(n_per_type=self.n_cases_per_type)

        # Add verification questions to each case
        for case in cases:
            verification_info = self.verification_gen.generate_verification(
                case)
            case.update(verification_info)

        self.test_cases = cases
        return cases
