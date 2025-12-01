"""
Condition Test 1
Tests counting how many pieces can attack a target
With per-case verification
"""

from typing import List, Dict
from src.condition.condition_test_base import ConditionTestBase
from src.condition.condition_test_1_generator import ConditionTest1Generator


class ConditionTest1(ConditionTestBase):
    """Test 1: Threat Count - counting attacking pieces"""

    def __init__(self,
                 base_output_dir: str = "./output/condition_test_1",
                 n_cases_per_level: int = 10,
                 seed: int = 42,
                 auto_timestamp: bool = True,
                 rate_limit_requests: int = 0,
                 rate_limit_pause: int = 0):
        """
        Initialize Condition Test 1
        Args:
            base_output_dir: Base directory for output files
            n_cases_per_level: Number of cases per condition level (1-6)
            seed: Random seed for reproducibility
            auto_timestamp: If True, append timestamp to output directory
            rate_limit_requests: Number of requests before pausing
            rate_limit_pause: Seconds to pause
        """
        super().__init__(
            test_layer=1,
            base_output_dir=base_output_dir,
            n_cases_per_level=n_cases_per_level,
            seed=seed,
            auto_timestamp=auto_timestamp,
            rate_limit_requests=rate_limit_requests,
            rate_limit_pause=rate_limit_pause
        )

    def generate_test_cases(self) -> List[Dict]:
        """Generate test cases automatically"""
        print(
            f"\nGenerating test cases (n_per_level={self.n_cases_per_level}, seed={self.seed})")
        print("="*60)

        generator = ConditionTest1Generator(seed=self.seed)
        cases = generator.generate_all(n_per_level=self.n_cases_per_level)

        # Add verification questions to each case
        for case in cases:
            verification_info = self.verification_gen.generate_verification(
                case)
            case.update(verification_info)

        self.test_cases = cases
        return cases
