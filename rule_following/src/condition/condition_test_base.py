"""
Base class for all Condition Tests
Provides common functionality for test execution, verification, and result reporting
"""

import os
from typing import List, Dict, Tuple
from datetime import datetime
from abc import ABC, abstractmethod
from PIL import Image
from src.data_structures import TestResult, save_results, create_summary
from src.board_generator import ChessBoardGenerator
from src.condition.verification_generator import ConditionVerificationGenerator
import time


class ConditionTestBase(ABC):
    """Abstract base class for condition tests"""

    def __init__(self,
                 test_layer: int,
                 base_output_dir: str,
                 n_cases_per_level: int = 10,
                 seed: int = 42,
                 auto_timestamp: bool = True,
                 rate_limit_requests: int = 0,
                 rate_limit_pause: int = 0):
        """
        Initialize Condition Test Base

        Args:
            test_layer: Test layer number (1, 2, 3, ...)
            base_output_dir: Base directory for output files
            n_cases_per_level: Number of cases per condition level
            seed: Random seed for reproducibility
            auto_timestamp: If True, append timestamp to output directory
            rate_limit_requests: Number of requests before pausing
            rate_limit_pause: Seconds to pause
        """
        self.test_layer = test_layer

        self.rate_limit_pause = rate_limit_pause
        self.rate_limit_requests = rate_limit_requests

        if auto_timestamp:
            timestamp = datetime.now().strftime("%m%d_%H%M%S")
            self.output_dir = f"{base_output_dir}_{timestamp}"
        else:
            self.output_dir = base_output_dir

        os.makedirs(self.output_dir, exist_ok=True)

        self.board_gen = ChessBoardGenerator()
        self.test_cases = []
        self.n_cases_per_level = n_cases_per_level
        self.seed = seed

        self.verification_gen = ConditionVerificationGenerator()

    @abstractmethod
    def generate_test_cases(self) -> List[Dict]:
        """
        Generate test cases (must be implemented by subclass)

        Returns:
            List of test case dictionaries
        """
        pass

    def create_test_images(self):
        """Generate images for all test cases"""
        print(f"\nCreating test images...")
        print("="*60)

        for i, case in enumerate(self.test_cases, 1):
            # Create 1 image per test case
            pieces = case.get('pieces', {})
            highlighted = case.get('highlighted_squares', [])

            img = self.board_gen.create_board_with_pieces(
                pieces=pieces,
                highlighted_squares=highlighted
            )

            # Save single image
            img_path = os.path.join(
                self.output_dir,
                f"{case['case_id']}.png"
            )
            img.save(img_path)

            # Store as single-item list for compatibility
            case["image_paths"] = [img_path]

            if i % 10 == 0 or i == len(self.test_cases):
                print(f"  Progress: {i}/{len(self.test_cases)} cases created")

        print(f"✓ All {len(self.test_cases)} test cases created\n")

    def generate_combined_prompt(self, case: Dict) -> str:
        """
        Generate combined prompt with verification question first, then test question
        """
        verification_q = case.get('verification_question', '')
        label = case.get('label', 'This is a chess position.')
        test_q = case['question']

        # Check if this is a counting question (Type 2)
        is_counting = case.get('type') == 'threat_count'

        if is_counting:
            prompt = f"""Look at this chess board image carefully.

{label}

First, to verify you can see the board correctly:
{verification_q}

Now, the main question:
{test_q}

Please answer both questions. Format your response exactly as:
Verification: [your answer - list all pieces and their squares]
Main answer: [number]"""
        else:
            prompt = f"""Look at this chess board image carefully.

{label}

First, to verify you can see the board correctly:
{verification_q}

Now, the main question:
{test_q}

Please answer both questions. Format your response exactly as:
Verification: [your answer - list all pieces and their squares]
Main answer: [yes/no/unknown]"""

        return prompt

    def run_test(self, model_client, save_results_flag: bool = True) -> Tuple[List[TestResult], Dict]:
        """
        Run the test with per-case verification

        Args:
            model_client: Model client for querying
            save_results_flag: Whether to save results to file

        Returns:
            Tuple of (results_list, statistics_dict)
        """
        results = []
        stats = {
            'total': 0,
            'verification_passed': 0,
            'verification_failed': 0,
            'test_correct': 0,
            'test_incorrect': 0,
            'test_correct_given_verified': 0,
        }

        print(f"{'='*60}")
        print(f"Running Condition Test {self.test_layer}")
        print("(Each case includes verification question + test question)")
        print(f"{'='*60}\n")

        for i, case in enumerate(self.test_cases, 1):
            print(f"[{i}/{len(self.test_cases)}] Testing {case['case_id']}...")

            stats['total'] += 1

            prompt = self.generate_combined_prompt(case)

            try:
                # Query model with combined prompt and all 3 images
                response = model_client.query(prompt, case["image_paths"])

                # Parse response
                verification_response, test_response = self._parse_combined_response(
                    response)

                # Check verification
                verification_passed = self.verification_gen.check_verification_answer(
                    verification_response,
                    case
                )

                if verification_passed:
                    stats['verification_passed'] += 1
                    print(f"  ✓ Verification passed")

                    # Extract test answer
                    model_answer = self._extract_answer(test_response)
                    correct = (model_answer.lower() ==
                               case["expected"].lower())

                    if correct:
                        stats['test_correct'] += 1
                        stats['test_correct_given_verified'] += 1
                        print(
                            f"  ✓ Test correct (Expected: {case['expected']}, Got: {model_answer})")
                    else:
                        stats['test_incorrect'] += 1
                        print(
                            f"  ✗ Test incorrect (Expected: {case['expected']}, Got: {model_answer})")

                else:
                    stats['verification_failed'] += 1
                    correct = False
                    model_answer = "N/A (verification failed)"
                    print(f"  ✗ Verification failed")
                    print(
                        f"    Expected keywords: {case.get('verification_keywords', 'N/A')}")
                    print(f"    Got: {verification_response[:100]}...")

                if self.rate_limit_requests > 0 and i % self.rate_limit_requests == 0 and i < len(self.test_cases):
                    print(
                        f"\n  ⏸️  Rate limit: Processed {i} requests, pausing for {self.rate_limit_pause} seconds...")
                    time.sleep(self.rate_limit_pause)
                    print(f"  ▶️  Resuming...\n")

            except Exception as e:
                print(f"  ✗ Error: {e}")
                verification_response = "error"
                test_response = "error"
                verification_passed = False
                model_answer = "error"
                correct = False
                stats['verification_failed'] += 1

            # Record result
            result = TestResult(
                test_type="condition",
                test_layer=self.test_layer,
                case_id=case["case_id"],
                rule_type=case.get('type'),
                verification_question=case.get('verification_question', ''),
                verification_expected=case.get('verification_expected', ''),
                verification_response=verification_response,
                verification_passed=verification_passed,
                question=case["question"],
                expected_answer=case["expected"],
                model_response=test_response,
                correct=correct,
                image_paths=case["image_paths"],
                model_name=model_client.model_name
            )
            results.append(result)

        # Print results
        self._print_results_summary(results, stats)

        if save_results_flag:
            output_file = os.path.join(
                self.output_dir, f"test_{self.test_layer}_results.json")

            # Create custom summary with level breakdown
            summary = self._create_summary_with_levels(results, stats)

            save_results(results, output_file, summary=summary)

        return results, stats

    def _parse_combined_response(self, response: str) -> Tuple[str, str]:
        """
        Parse model response into verification answer and test answer
        """
        lines = response.split('\n')

        verification_response = ""
        test_response = ""

        for line in lines:
            line_lower = line.lower().strip()

            if line_lower.startswith('verification:'):
                verification_response = line.split(':', 1)[1].strip()
            elif line_lower.startswith('main answer:') or line_lower.startswith('main:'):
                test_response = line.split(':', 1)[1].strip()

        # If parsing failed, try to extract from full response
        if not verification_response or not test_response:
            non_empty = [l.strip() for l in lines if l.strip()]
            if len(non_empty) >= 2:
                verification_response = non_empty[0]
                test_response = non_empty[1]
            else:
                verification_response = response[:len(response)//2]
                test_response = response[len(response)//2:]

        return verification_response, test_response

    def _extract_answer(self, response: str) -> str:
        """Extract answer from model response (supports both yes/no and numbers)"""
        response_lower = response.lower().strip()

        # Try to extract a number first (for Type 2)
        import re
        number_match = re.search(r'\b(\d+)\b', response_lower[:30])
        if number_match:
            return number_match.group(1)

        # Fallback to yes/no/unknown (for Type 1)
        if "yes" in response_lower[:20]:
            return "yes"
        elif "no" in response_lower[:20]:
            return "no"
        else:
            return "unknown"

    def _print_results_summary(self, results: List[TestResult], stats: Dict):
        """Print detailed results summary"""
        if not results:
            return

        print(f"\n{'='*60}")
        print(f"RESULTS SUMMARY")
        print(f"{'='*60}")

        # Verification statistics
        verification_rate = stats['verification_passed'] / \
            stats['total'] if stats['total'] > 0 else 0
        print(f"\nBoard Recognition:")
        print(
            f"  Verified correctly: {stats['verification_passed']}/{stats['total']} ({verification_rate:.1%})")
        print(
            f"  Failed to recognize: {stats['verification_failed']}/{stats['total']} ({1-verification_rate:.1%})")

        # Test accuracy (only among verified cases)
        if stats['verification_passed'] > 0:
            accuracy_given_verified = stats['test_correct_given_verified'] / \
                stats['verification_passed']
            print(f"\nTest Accuracy (among recognized cases):")
            print(
                f"  Correct: {stats['test_correct_given_verified']}/{stats['verification_passed']} ({accuracy_given_verified:.1%})")
        else:
            print(f"\n⚠️  No cases passed verification!")

        # Overall accuracy (including verification failures)
        overall_accuracy = stats['test_correct'] / \
            stats['total'] if stats['total'] > 0 else 0
        print(f"\nOverall Accuracy (all cases):")
        print(
            f"  Correct: {stats['test_correct']}/{stats['total']} ({overall_accuracy:.1%})")

        # Breakdown by level (only verified cases)
        self._print_level_breakdown(results)

        print(f"{'='*60}\n")

    def _print_level_breakdown(self, results: List[TestResult]):
        """Print accuracy breakdown by condition level"""
        print(f"\nAccuracy by level (verified cases only):")
        level_results = {}

        for result in results:
            if not result.verification_passed:
                continue

            case = next(
                (c for c in self.test_cases if c['case_id'] == result.case_id), None)
            if case:
                level = case.get('level', 0)

                if level not in level_results:
                    level_results[level] = {'correct': 0, 'total': 0}

                level_results[level]['total'] += 1
                if result.correct:
                    level_results[level]['correct'] += 1

        for level in sorted(level_results.keys()):
            stats_item = level_results[level]
            acc = stats_item['correct'] / \
                stats_item['total'] if stats_item['total'] > 0 else 0
            print(
                f"  Level {level} ({level} conditions): {acc:5.1%} ({stats_item['correct']:2d}/{stats_item['total']:2d})")

    def _create_summary_with_levels(self, results: List[TestResult], stats: Dict) -> Dict:
        """
        Create summary with level breakdown

        Args:
            results: List of TestResult objects
            stats: Statistics dictionary

        Returns:
            Summary dictionary
        """
        # Calculate rates
        verification_rate = stats['verification_passed'] / \
            stats['total'] if stats['total'] > 0 else 0
        accuracy_given_verified = stats['test_correct_given_verified'] / \
            stats['verification_passed'] if stats['verification_passed'] > 0 else 0
        overall_accuracy = stats['test_correct'] / \
            stats['total'] if stats['total'] > 0 else 0

        summary = {
            "model_name": results[0].model_name if results else "unknown",
            "test_type": "condition",
            "test_layer": self.test_layer,
            "total_cases": stats['total'],
            "timestamp": datetime.now().isoformat(),
            "board_recognition": {
                "verified_correctly": stats['verification_passed'],
                "failed_to_recognize": stats['verification_failed'],
                "verification_rate": round(verification_rate, 3)
            },
            "test_accuracy": {
                "correct_among_verified": stats['test_correct_given_verified'],
                "total_verified": stats['verification_passed'],
                "accuracy_given_verified": round(accuracy_given_verified, 3),
                "overall_correct": stats['test_correct'],
                "overall_accuracy": round(overall_accuracy, 3)
            }
        }

        # Add level breakdown
        level_breakdown = {}
        for result in results:
            if not result.verification_passed:
                continue

            case = next((c for c in self.test_cases if c.get(
                'case_id') == result.case_id), None)
            if case:
                level = case.get('level', 0)
                if level not in level_breakdown:
                    level_breakdown[level] = {'correct': 0, 'total': 0}
                level_breakdown[level]['total'] += 1
                if result.correct:
                    level_breakdown[level]['correct'] += 1

        # Add accuracy percentages
        for level in level_breakdown:
            stats_item = level_breakdown[level]
            stats_item['accuracy'] = round(
                stats_item['correct'] /
                stats_item['total'] if stats_item['total'] > 0 else 0,
                3
            )

        summary["accuracy_by_level"] = dict(sorted(level_breakdown.items()))

        return summary
