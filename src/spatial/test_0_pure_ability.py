"""
Spatial Test 0: Pure Spatial Reasoning
Tests basic spatial understanding without chess rules
With per-case board recognition verification
"""

import os
from typing import List, Dict, Tuple
from datetime import datetime
from ..data_structures import TestResult, save_results
from ..board_generator import ChessBoardGenerator
from .test_0_generator import SpatialTest0Generator
from .verification_generator import VerificationQuestionGenerator


class SpatialTest0:
    """Test 0: Pure spatial understanding (no chess rules)"""

    def __init__(self,
                 base_output_dir: str = "./output/spatial_test_0",
                 n_cases_per_type: int = 10,
                 seed: int = 42,
                 auto_timestamp: bool = True):
        """
        Initialize Spatial Test 0

        Args:
            base_output_dir: Base directory for output files
            n_cases_per_type: Number of cases per test type
            seed: Random seed for reproducibility
            auto_timestamp: If True, append timestamp to output directory
        """
        if auto_timestamp:
            timestamp = datetime.now().strftime("%m%d_%H%M%S")
            self.output_dir = f"{base_output_dir}_{timestamp}"

        else:
            self.output_dir = base_output_dir

        os.makedirs(self.output_dir, exist_ok=True)

        self.board_gen = ChessBoardGenerator()
        self.test_cases = []
        self.n_cases_per_type = n_cases_per_type
        self.seed = seed

        self.verification_gen = VerificationQuestionGenerator()

    def generate_test_cases(self) -> List[Dict]:
        """Generate test cases automatically"""
        print(
            f"\nGenerating test cases (n_per_type={self.n_cases_per_type}, seed={self.seed})")
        print("="*60)

        generator = SpatialTest0Generator(seed=self.seed)
        cases = generator.generate_all(n_per_type=self.n_cases_per_type)

        # Add verification questions to each case
        for case in cases:
            verification_info = self.verification_gen.generate_verification(
                case)
            case.update(verification_info)

        self.test_cases = cases
        return cases

    def create_test_images(self):
        """Generate images for all test cases"""
        print(f"\nCreating test images...")
        print("="*60)

        for i, case in enumerate(self.test_cases, 1):
            if "pieces" in case:
                img = self.board_gen.create_board_with_pieces(
                    pieces=case["pieces"],
                    highlighted_squares=case["squares"]
                )
            else:
                img = self.board_gen.create_empty_board(
                    highlighted_squares=case["squares"]
                )

            img_path = os.path.join(self.output_dir, f"{case['case_id']}.png")
            img.save(img_path)
            case["image_path"] = img_path

            if i % 10 == 0 or i == len(self.test_cases):
                print(f"  Progress: {i}/{len(self.test_cases)} images created")

        print(f"✓ All {len(self.test_cases)} images created\n")

    def generate_combined_prompt(self, case: Dict) -> str:
        """
        Generate combined prompt with verification question first, then test question

        This ensures the model sees the board correctly before we test its reasoning
        """
        verification_q = case.get('verification_question', '')
        test_q = case['question']

        prompt = f"""Look at this chess board carefully.

First, a simple verification question to make sure you see the board correctly:
{verification_q}

Now, the main question:
{test_q}

Please answer both questions. Format your response as:
Verification: [your answer to verification question]
Main answer: [yes/no/unknown for the main question]"""

        return prompt

    def run_test(self, model_client, save_results_flag: bool = True) -> Tuple[List[TestResult], Dict]:
        """
        Run the test with per-case verification

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
            'test_correct_given_verified': 0,  # Among verified cases
        }

        print(f"{'='*60}")
        print("Running Spatial Test 0: Pure Spatial Reasoning")
        print("(Each case includes verification question + test question)")
        print(f"{'='*60}\n")

        for i, case in enumerate(self.test_cases, 1):
            print(f"[{i}/{len(self.test_cases)}] Testing {case['case_id']}...")

            stats['total'] += 1

            prompt = self.generate_combined_prompt(case)

            try:
                # Query model with combined prompt
                response = model_client.query(prompt, case["image_path"])

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
                        f"    Expected: {case.get('verification_expected', 'N/A')}")
                    print(f"    Got: {verification_response[:50]}...")

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
                test_type="spatial",
                test_layer=0,
                case_id=case["case_id"],
                verification_question=case.get('verification_question', ''),
                verification_expected=case.get('verification_expected', ''),
                verification_response=verification_response,
                verification_passed=verification_passed,
                question=case["question"],
                expected_answer=case["expected"],
                model_response=test_response,
                correct=correct,
                image_paths=[case["image_path"]],
                model_name=model_client.model_name
            )
            results.append(result)

        # Print results
        self._print_results_summary(results, stats)

        if save_results_flag:
            output_file = os.path.join(self.output_dir, "test_0_results.json")
            save_results(results, output_file)

        return results, stats

    def _parse_combined_response(self, response: str) -> Tuple[str, str]:
        """
        Parse model response into verification answer and test answer

        Expected format:
        Verification: [answer]
        Main answer: [answer]
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
            # Fallback: split by newlines and take first two non-empty
            non_empty = [l.strip() for l in lines if l.strip()]
            if len(non_empty) >= 2:
                verification_response = non_empty[0]
                test_response = non_empty[1]
            else:
                verification_response = response[:len(response)//2]
                test_response = response[len(response)//2:]

        return verification_response, test_response

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

        # Breakdown by test type (only verified cases)
        print(f"\nAccuracy by test type (verified cases only):")
        type_results = {}

        for result in results:
            if not result.verification_passed:
                continue  # Skip unverified cases

            case = next(
                (c for c in self.test_cases if c['case_id'] == result.case_id), None)
            if case:
                case_type = case.get('type', 'unknown')
                subtype = case.get('subtype', '')
                key = f"{case_type}" if not subtype else f"{case_type}_{subtype}"

                if key not in type_results:
                    type_results[key] = {'correct': 0, 'total': 0}
                type_results[key]['total'] += 1
                if result.correct:
                    type_results[key]['correct'] += 1

        for case_type in sorted(type_results.keys()):
            stats_item = type_results[case_type]
            acc = stats_item['correct'] / \
                stats_item['total'] if stats_item['total'] > 0 else 0
            print(
                f"  {case_type:20s}: {acc:5.1%} ({stats_item['correct']:2d}/{stats_item['total']:2d})")

        print(f"{'='*60}\n")

    def _extract_answer(self, response: str) -> str:
        """Extract answer from model response"""
        response_lower = response.lower().strip()

        if "yes" in response_lower[:20]:
            return "yes"
        elif "no" in response_lower[:20]:
            return "no"
        else:
            return "unknown"
