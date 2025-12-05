"""
Automated test case generator for Temporal Test 0
Pure temporal reasoning without chess rules - automatically generated
"""

import random
from typing import List, Dict


class TemporalTest0Generator:
    """Automatically generate temporal reasoning test cases"""

    def __init__(self, seed: int = 42):
        """
        Initialize generator

        Args:
            seed: Random seed for reproducibility
        """
        random.seed(seed)
        self.files = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
        self.ranks = ['1', '2', '3', '4', '5', '6', '7', '8']

    def _random_square(self) -> str:
        """Generate random square"""
        file = random.choice(self.files)
        rank = random.choice(self.ranks)
        return file + rank

    def _random_piece(self) -> str:
        """Generate random piece symbol"""
        pieces = ['P', 'N', 'B', 'R', 'Q', 'K', 'p', 'n', 'b', 'r', 'q', 'k']
        return random.choice(pieces)

    def _get_different_square(self, square: str) -> str:
        """Get a different random square"""
        new_square = self._random_square()
        while new_square == square:
            new_square = self._random_square()
        return new_square

    # ============= Type 1: Movement Detection =============

    def generate_movement_detection_tests(self, n_positive: int = 5, n_negative: int = 5) -> List[Dict]:
        """
        Test: Can the model detect position changes?
        Pure temporal ability - no chess rules involved

        Args:
            n_positive: Number of positive cases (piece moved to target)
            n_negative: Number of negative cases (piece moved elsewhere)
        """
        cases = []

        # Positive cases: piece did move from A to B
        for i in range(n_positive):
            piece = self._random_piece()
            start_sq = self._random_square()
            end_sq = self._get_different_square(start_sq)

            question = f"""Between these two states, did the piece move from {start_sq} to {end_sq}?
- Answer 'yes' if the piece moved from {start_sq} to {end_sq}
- Answer 'no' if the piece moved to a different location
- Answer 'unknown' if you cannot determine"""

            cases.append({
                "case_id": f"movement_pos_{i+1}",
                "type": "movement_detection",
                "subtype": "moved_to_target",
                "states": [
                    {"pieces": {start_sq: piece}, "squares": []},
                    {"pieces": {end_sq: piece}, "squares": []}
                ],
                "label": "These are two consecutive board states.",
                "question": question,
                "expected": "yes",
                "reasoning": f"Piece moved from {start_sq} to {end_sq}"
            })

        # Negative cases: piece moved from A to C (not B)
        for i in range(n_negative):
            piece = self._random_piece()
            start_sq = self._random_square()
            actual_end_sq = self._get_different_square(start_sq)
            asked_sq = self._get_different_square(start_sq)
            while asked_sq == actual_end_sq:
                asked_sq = self._get_different_square(start_sq)

            question = f"""Between these two states, did the piece move from {start_sq} to {asked_sq}?
- Answer 'yes' if the piece moved from {start_sq} to {asked_sq}
- Answer 'no' if the piece moved to a different location
- Answer 'unknown' if you cannot determine"""

            cases.append({
                "case_id": f"movement_neg_{i+1}",
                "type": "movement_detection",
                "subtype": "moved_elsewhere",
                "states": [
                    {"pieces": {start_sq: piece}, "squares": []},
                    {"pieces": {actual_end_sq: piece}, "squares": []}
                ],
                "label": "These are two consecutive board states.",
                "question": question,
                "expected": "no",
                "reasoning": f"Piece moved to {actual_end_sq}, not {asked_sq}"
            })

        return cases

    # ============= Type 2: Sequence Order =============

    def generate_sequence_order_tests(self, n_positive: int = 5, n_negative: int = 5) -> List[Dict]:
        """
        Test: Can the model understand temporal sequence order?

        Args:
            n_positive: Number of cases asking about correct order
            n_negative: Number of cases asking about incorrect order
        """
        cases = []

        # Positive cases: asking about correct order
        for i in range(n_positive):
            piece = self._random_piece()
            sq_a = self._random_square()
            sq_b = self._get_different_square(sq_a)
            sq_c = self._get_different_square(sq_b)
            while sq_c == sq_a:
                sq_c = self._get_different_square(sq_b)

            question = f"""Did the piece move to {sq_b} before moving to {sq_c}?
- Answer 'yes' if the sequence shows movement to {sq_b} before {sq_c}
- Answer 'no' if this is not the order shown
- Answer 'unknown' if you cannot determine"""

            cases.append({
                "case_id": f"sequence_pos_{i+1}",
                "type": "sequence_order",
                "subtype": "correct_order",
                "states": [
                    {"pieces": {sq_a: piece}, "squares": []},
                    {"pieces": {sq_b: piece}, "squares": []},
                    {"pieces": {sq_c: piece}, "squares": []}
                ],
                "label": "States are shown in chronological order (1 → 2 → 3).",
                "question": question,
                "expected": "yes",
                "reasoning": f"Sequence is {sq_a} → {sq_b} → {sq_c}"
            })

        # Negative cases: asking about wrong order
        for i in range(n_negative):
            piece = self._random_piece()
            sq_a = self._random_square()
            sq_b = self._get_different_square(sq_a)
            sq_c = self._get_different_square(sq_b)
            while sq_c == sq_a:
                sq_c = self._get_different_square(sq_b)

            # Ask if went to C before B (wrong order)
            question = f"""Did the piece move to {sq_c} before moving to {sq_b}?
- Answer 'yes' if the sequence shows movement to {sq_c} before {sq_b}
- Answer 'no' if this is not the order shown
- Answer 'unknown' if you cannot determine"""

            cases.append({
                "case_id": f"sequence_neg_{i+1}",
                "type": "sequence_order",
                "subtype": "wrong_order",
                "states": [
                    {"pieces": {sq_a: piece}, "squares": []},
                    {"pieces": {sq_b: piece}, "squares": []},
                    {"pieces": {sq_c: piece}, "squares": []}
                ],
                "label": "States are shown in chronological order (1 → 2 → 3).",
                "question": question,
                "expected": "no",
                "reasoning": f"Sequence is {sq_a} → {sq_b} → {sq_c}, not to {sq_c} before {sq_b}"
            })

        return cases

    # ============= Type 3: State Comparison =============

    def generate_state_comparison_tests(self, n_positive: int = 5, n_negative: int = 5) -> List[Dict]:
        """
        Test: Can the model compare states across time?
        Specifically: Did the piece return to its starting position?

        Args:
            n_positive: Number of cases where piece returns
            n_negative: Number of cases where piece doesn't return
        """
        cases = []

        # Positive cases: piece returns to starting position
        for i in range(n_positive):
            piece = self._random_piece()
            start_sq = self._random_square()
            middle_sq = self._get_different_square(start_sq)

            question = f"""Did the piece return to its starting position?
- Answer 'yes' if the piece is at the same position in State 3 as in State 1
- Answer 'no' if the piece is at a different position
- Answer 'unknown' if you cannot determine"""

            cases.append({
                "case_id": f"comparison_pos_{i+1}",
                "type": "state_comparison",
                "subtype": "returned",
                "states": [
                    {"pieces": {start_sq: piece}, "squares": []},
                    {"pieces": {middle_sq: piece}, "squares": []},
                    {"pieces": {start_sq: piece}, "squares": []}  # Back to start
                ],
                "label": "States are shown in chronological order (1 → 2 → 3).",
                "question": question,
                "expected": "yes",
                "reasoning": f"Piece started at {start_sq} and returned to {start_sq}"
            })

        # Negative cases: piece doesn't return
        for i in range(n_negative):
            piece = self._random_piece()
            sq_a = self._random_square()
            sq_b = self._get_different_square(sq_a)
            sq_c = self._get_different_square(sq_a)
            while sq_c == sq_b:
                sq_c = self._get_different_square(sq_a)

            question = f"""Did the piece return to its starting position?
- Answer 'yes' if the piece is at the same position in State 3 as in State 1
- Answer 'no' if the piece is at a different position
- Answer 'unknown' if you cannot determine"""

            cases.append({
                "case_id": f"comparison_neg_{i+1}",
                "type": "state_comparison",
                "subtype": "not_returned",
                "states": [
                    {"pieces": {sq_a: piece}, "squares": []},
                    {"pieces": {sq_b: piece}, "squares": []},
                    # Different from start
                    {"pieces": {sq_c: piece}, "squares": []}
                ],
                "label": "States are shown in chronological order (1 → 2 → 3).",
                "question": question,
                "expected": "no",
                "reasoning": f"Piece started at {sq_a} but ended at {sq_c}"
            })

        return cases

    # ============= Type 4: Position Tracking =============

    def generate_position_tracking_tests(self, n_positive: int = 5, n_negative: int = 5) -> List[Dict]:
        """
        Test: Can the model track if a position was visited at any point?

        Args:
            n_positive: Number of cases where position was visited
            n_negative: Number of cases where position was never visited
        """
        cases = []

        # Positive cases: asking about a position that was visited
        for i in range(n_positive):
            piece = self._random_piece()
            sq_a = self._random_square()
            sq_b = self._get_different_square(sq_a)
            sq_c = self._get_different_square(sq_b)
            while sq_c == sq_a:
                sq_c = self._get_different_square(sq_b)

            # Ask about a position that WAS visited (randomly choose State 1, 2, or 3)
            visited_state = random.choice([1, 2, 3])
            ask_sq = [sq_a, sq_b, sq_c][visited_state - 1]

            question = f"""At any point in the sequence, was the piece at {ask_sq}?
- Answer 'yes' if the piece was at {ask_sq} in any of the shown states
- Answer 'no' if the piece was never at {ask_sq}
- Answer 'unknown' if you cannot determine"""

            cases.append({
                "case_id": f"tracking_pos_{i+1}",
                "type": "position_tracking",
                "subtype": "was_there",
                "states": [
                    {"pieces": {sq_a: piece}, "squares": []},
                    {"pieces": {sq_b: piece}, "squares": []},
                    {"pieces": {sq_c: piece}, "squares": []}
                ],
                "label": "States are shown in chronological order (1 → 2 → 3).",
                "question": question,
                "expected": "yes",
                "reasoning": f"Piece was at {ask_sq} in State {visited_state}"
            })

        # Negative cases: asking about a position that was NEVER visited
        for i in range(n_negative):
            piece = self._random_piece()
            sq_a = self._random_square()
            sq_b = self._get_different_square(sq_a)
            sq_c = self._get_different_square(sq_b)
            while sq_c == sq_a:
                sq_c = self._get_different_square(sq_b)

            # Find a square that was never visited
            never_visited = self._random_square()
            while never_visited in [sq_a, sq_b, sq_c]:
                never_visited = self._random_square()

            question = f"""At any point in the sequence, was the piece at {never_visited}?
- Answer 'yes' if the piece was at {never_visited} in any of the shown states
- Answer 'no' if the piece was never at {never_visited}
- Answer 'unknown' if you cannot determine"""

            cases.append({
                "case_id": f"tracking_neg_{i+1}",
                "type": "position_tracking",
                "subtype": "was_not_there",
                "states": [
                    {"pieces": {sq_a: piece}, "squares": []},
                    {"pieces": {sq_b: piece}, "squares": []},
                    {"pieces": {sq_c: piece}, "squares": []}
                ],
                "label": "States are shown in chronological order (1 → 2 → 3).",
                "question": question,
                "expected": "no",
                "reasoning": f"Piece was never at {never_visited} (visited {sq_a}, {sq_b}, {sq_c})"
            })

        return cases

    # ============= Main Generation Method =============

    def generate_all(self, n_per_type: int = 10) -> List[Dict]:
        """
        Generate comprehensive test suite

        Args:
            n_per_type: Number of cases per test type (split into pos/neg)

        Returns:
            List of test case dictionaries
        """
        all_cases = []

        n_pos = n_per_type // 2
        n_neg = n_per_type - n_pos

        print(f"Generating movement detection tests...")
        all_cases.extend(self.generate_movement_detection_tests(n_pos, n_neg))

        print(f"Generating sequence order tests...")
        all_cases.extend(self.generate_sequence_order_tests(n_pos, n_neg))

        print(f"Generating state comparison tests...")
        all_cases.extend(self.generate_state_comparison_tests(n_pos, n_neg))

        print(f"Generating position tracking tests...")
        all_cases.extend(self.generate_position_tracking_tests(n_pos, n_neg))

        print(f"\n✓ Total generated: {len(all_cases)} test cases")

        return all_cases
