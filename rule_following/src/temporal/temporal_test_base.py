"""
Base class for all Temporal Tests
Provides common functionality for test execution, verification, and result reporting
"""

import os
from typing import List, Dict, Tuple
from datetime import datetime
from abc import ABC, abstractmethod
from PIL import Image, ImageDraw, ImageFont
from ..data_structures import TestResult, save_results, create_summary
from ..board_generator import ChessBoardGenerator
from .verification_generator import TemporalVerificationGenerator
import time


class TemporalTestBase(ABC):
    """Abstract base class for temporal tests"""

    def __init__(self,
                 test_layer: int,
                 base_output_dir: str,
                 n_cases_per_type: int = 10,
                 seed: int = 42,
                 auto_timestamp: bool = True,
                 rate_limit_requests: int = 0,
                 rate_limit_pause: int = 0):
        """
        Initialize Temporal Test Base

        Args:
            test_layer: Test layer number (0, 1, 2, 3)
            base_output_dir: Base directory for output files
            n_cases_per_type: Number of cases per test type
            seed: Random seed for reproducibility
            auto_timestamp: If True, append timestamp to output directory
            rate_limit_requests: Number of requests before pausing (None = no limit)
            rate_limit_pause: Seconds to pause (None = no limit)
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
        self.n_cases_per_type = n_cases_per_type
        self.seed = seed

        self.verification_gen = TemporalVerificationGenerator()

    @abstractmethod
    def generate_test_cases(self) -> List[Dict]:
        """
        Generate test cases (must be implemented by subclass)

        Returns:
            List of test case dictionaries
        """
        pass

    def create_test_images(self):
        """Generate images for all test cases with state labels"""
        print(f"\nCreating test images...")
        print("="*60)

        for i, case in enumerate(self.test_cases, 1):
            # Create images for all states in the sequence
            image_paths = []

            for state_idx, state in enumerate(case.get('states', [])):
                pieces = state.get('pieces', {})
                squares = state.get('squares', [])

                if pieces:
                    img = self.board_gen.create_board_with_pieces(
                        pieces=pieces,
                        highlighted_squares=squares
                    )
                else:
                    img = self.board_gen.create_empty_board(
                        highlighted_squares=squares
                    )

                # ✅ Add "State N" label to the image
                img = self._add_state_label(img, state_idx + 1)

                img_path = os.path.join(
                    self.output_dir,
                    f"{case['case_id']}_state_{state_idx+1}.png"
                )
                img.save(img_path)
                image_paths.append(img_path)

            case["image_paths"] = image_paths

            if i % 10 == 0 or i == len(self.test_cases):
                print(f"  Progress: {i}/{len(self.test_cases)} cases created")

        print(f"✓ All {len(self.test_cases)} test cases created\n")

    def _add_state_label(self, img: Image.Image, state_num: int) -> Image.Image:
        """
        Add 'State N' label to the top of the image

        Args:
            img: Original chess board image
            state_num: State number (1, 2, 3, etc.)

        Returns:
            New image with label at the top
        """
        # Create a new image with extra space at the top for label
        label_height = 50
        new_img = Image.new(
            'RGB', (img.width, img.height + label_height), 'white')

        # Paste original image below the label area
        new_img.paste(img, (0, label_height))

        # Draw the label
        draw = ImageDraw.Draw(new_img)

        # Try to load a font, fall back to default if needed
        try:
            font = ImageFont.truetype("arial.ttf", 28)
        except:
            try:
                font = ImageFont.truetype("Arial.ttf", 28)
            except:
                try:
                    # Try system fonts on different platforms
                    font = ImageFont.truetype(
                        "/System/Library/Fonts/Helvetica.ttc", 28)
                except:
                    font = ImageFont.load_default()

        # Draw "State N" text
        text = f"State {state_num}"

        # Get text bounding box for centering
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        x = (new_img.width - text_width) // 2
        y = (label_height - text_height) // 2

        draw.text((x, y), text, fill='black', font=font)

        return new_img

    def generate_combined_prompt(self, case: Dict) -> str:
        """
        Generate combined prompt with verification question first, then test question
        Handles both yes/no questions and multiple choice questions
        Explicitly specifies verification answer format
        """
        verification_q = case.get('verification_question', '')
        label = case.get('label', 'These are chess board states.')
        test_q = case['question']

        # Count the number of states
        num_states = len(case.get('states', []))

        # Create explicit image-to-state mapping
        if num_states == 2:
            image_ref = "Image 1 shows State 1. Image 2 shows State 2."
            verification_format_example = "State 1: White Pawn at e5, Black Pawn at f7; State 2: White Pawn at e5, Black Pawn at f5"
        elif num_states == 3:
            image_ref = "Image 1 shows State 1. Image 2 shows State 2. Image 3 shows State 3."
            verification_format_example = "State 1: White Pawn at e5; State 2: Black Pawn at f5; State 3: White Pawn at f6"
        elif num_states == 4:
            image_ref = "Image 1 shows State 1. Image 2 shows State 2. Image 3 shows State 3. Image 4 shows State 4."
            verification_format_example = "State 1: King at e1, Rook at h1; State 2: King at f1, Rook at h1; State 3: King at g1, Rook at h1; State 4: King at g1, Rook at f1"
        else:
            image_refs = [
                f"Image {i+1} shows State {i+1}" for i in range(num_states)]
            image_ref = ". ".join(image_refs) + "."
            verification_format_example = "State 1: [pieces]; State 2: [pieces]; ..."

        # Check if this is a multiple choice question
        if 'options' in case:
            options = case['options']
            options_text = "\n".join([f"{k}) {v}" for k, v in options.items()])

            prompt = f"""Look at these chess board states carefully.

    {image_ref}

    {label}

    First, a simple verification question to make sure you see the states correctly:
    {verification_q}

    For the verification answer, use this exact format:
    - For each state, list all pieces as: [Color] [Piece Type] at [square]
    - Separate states with semicolons
    - Example: {verification_format_example}

    Now, the main question:
    {test_q}

    Options:
    {options_text}

    Please answer both questions. Format your response exactly as:
    Verification: [your answer using the format above]
    Main answer: [A/B/C/D]"""
        else:
            # Regular yes/no question
            prompt = f"""Look at these chess board states carefully.

    {image_ref}

    {label}

    First, a simple verification question to make sure you see the states correctly:
    {verification_q}

    For the verification answer, use this exact format:
    - For each state, list all pieces as: [Color] [Piece Type] at [square]
    - Separate states with semicolons
    - Example: {verification_format_example}

    Now, the main question:
    {test_q}

    Please answer both questions. Format your response exactly as:
    Verification: [your answer using the format above]
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
        print(f"Running Temporal Test {self.test_layer}")
        print("(Each case includes verification question + test question)")
        print(f"{'='*60}\n")

        for i, case in enumerate(self.test_cases, 1):
            print(f"[{i}/{len(self.test_cases)}] Testing {case['case_id']}...")

            stats['total'] += 1

            prompt = self.generate_combined_prompt(case)

            try:
                # ✅ Query model with combined prompt and ALL images
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
                        f"    Expected: {case.get('verification_expected', 'N/A')}")
                    print(f"    Got: {verification_response[:50]}...")

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
                test_type="temporal",
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
            summary = create_summary(results, stats, self.test_cases)
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
        """Extract answer from model response"""
        response_lower = response.lower().strip()

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

        # Breakdown by type (only verified cases)
        self._print_type_breakdown(results)

        print(f"{'='*60}\n")

    def _print_type_breakdown(self, results: List[TestResult]):
        """Print accuracy breakdown by type"""
        print(f"\nAccuracy by type (verified cases only):")
        type_results = {}

        for result in results:
            if not result.verification_passed:
                continue

            case = next(
                (c for c in self.test_cases if c['case_id'] == result.case_id), None)
            if case:
                case_type = case.get('type', 'unknown')
                subtype = case.get('subtype', '')

                # Overall by type
                if case_type not in type_results:
                    type_results[case_type] = {
                        'correct': 0, 'total': 0, 'subtypes': {}}
                type_results[case_type]['total'] += 1
                if result.correct:
                    type_results[case_type]['correct'] += 1

                # By subtype
                if subtype:
                    if subtype not in type_results[case_type]['subtypes']:
                        type_results[case_type]['subtypes'][subtype] = {
                            'correct': 0, 'total': 0}
                    type_results[case_type]['subtypes'][subtype]['total'] += 1
                    if result.correct:
                        type_results[case_type]['subtypes'][subtype]['correct'] += 1

        for case_type in sorted(type_results.keys()):
            stats_item = type_results[case_type]
            acc = stats_item['correct'] / \
                stats_item['total'] if stats_item['total'] > 0 else 0
            print(
                f"  {case_type.capitalize():20s}: {acc:5.1%} ({stats_item['correct']:2d}/{stats_item['total']:2d})")

            # Show subtypes
            for subtype, sub_stats in sorted(stats_item['subtypes'].items()):
                sub_acc = sub_stats['correct'] / \
                    sub_stats['total'] if sub_stats['total'] > 0 else 0
                print(
                    f"    └─ {subtype:23s}: {sub_acc:5.1%} ({sub_stats['correct']:2d}/{sub_stats['total']:2d})")

    def _extract_answer(self, response: str) -> str:
        """
        Extract answer from model response
        Handles both yes/no/unknown and A/B/C/D formats
        """
        response_lower = response.lower().strip()

        # Check for multiple choice answers first
        if response_lower.startswith(('a)', 'a.', 'a ', 'a:')):
            return "A"
        elif response_lower.startswith(('b)', 'b.', 'b ', 'b:')):
            return "B"
        elif response_lower.startswith(('c)', 'c.', 'c ', 'c:')):
            return "C"
        elif response_lower.startswith(('d)', 'd.', 'd ', 'd:')):
            return "D"

        # Check for single letter answers
        first_word = response_lower.split(
        )[0] if response_lower.split() else ""
        if first_word in ['a', 'b', 'c', 'd']:
            return first_word.upper()

        # Check for yes/no/unknown
        if "yes" in response_lower[:20]:
            return "yes"
        elif "no" in response_lower[:20]:
            return "no"
        else:
            return "unknown"
