"""
Level 3 Generator: En Passant Basic
Tests 3 basic conditions for en passant
"""

import random
from typing import List, Dict, Tuple


class Level3Generator:
    """Generate Level 3 test cases - en passant basic conditions"""

    def __init__(self, seed: int = 42):
        """
        Initialize generator

        Args:
            seed: Random seed for reproducibility
        """
        random.seed(seed)
        self.files = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
        self.ranks = ['1', '2', '3', '4', '5', '6', '7', '8']

    def _adjacent_files(self, file: str) -> List[str]:
        """Get adjacent files"""
        file_idx = self.files.index(file)
        adjacent = []
        if file_idx > 0:
            adjacent.append(self.files[file_idx - 1])
        if file_idx < 7:
            adjacent.append(self.files[file_idx + 1])
        return adjacent

    def _generate_all_valid_combinations(self) -> List[Dict]:
        """
        Generate all 28 valid en passant combinations

        Valid en passant occurs when:
        1. Black pawn moves from rank 7 to rank 5 (2 squares)
        2. White pawn is on rank 5
        3. White pawn is on adjacent file to black pawn

        Total combinations:
        - a-file black pawn: can be captured by b5 white pawn (1 combination)
        - b-file black pawn: can be captured by a5 or c5 white pawn (2 combinations)
        - c-file black pawn: can be captured by b5 or d5 white pawn (2 combinations)
        - ... (similar for d, e, f, g)
        - h-file black pawn: can be captured by g5 white pawn (1 combination)

        Total: 1 + 2*6 + 1 = 14 combinations for white capturing black
        Same for black capturing white (white pawn rank 2→4, black pawn rank 4)
        Total: 28 combinations
        """
        all_valid = []
        case_num = 1

        # White captures black en passant
        for black_file in self.files:
            black_start = black_file + '7'
            black_end = black_file + '5'

            # Get adjacent files for white pawn
            adjacent = self._adjacent_files(black_file)

            for white_file in adjacent:
                white_sq = white_file + '5'

                all_valid.append({
                    "case_id": f"L3_valid_{case_num}",
                    "type": "en_passant_basic",
                    "subtype": "all_conditions_met",
                    "states": [
                        {"pieces": {white_sq: 'P', black_start: 'p'}, "squares": []},
                        {"pieces": {white_sq: 'P', black_end: 'p'}, "squares": []}
                    ],
                    "question": "Can white capture the black pawn en passant?",
                    "expected": "yes",
                    "reasoning": f"Black pawn moved 2 squares from {black_start} to {black_end}, white pawn at {white_sq} is adjacent"
                })
                case_num += 1

        # Black captures white en passant
        for white_file in self.files:
            white_start = white_file + '2'
            white_end = white_file + '4'

            # Get adjacent files for black pawn
            adjacent = self._adjacent_files(white_file)

            for black_file in adjacent:
                black_sq = black_file + '4'

                all_valid.append({
                    "case_id": f"L3_valid_{case_num}",
                    "type": "en_passant_basic",
                    "subtype": "all_conditions_met",
                    "states": [
                        {"pieces": {black_sq: 'p', white_start: 'P'}, "squares": []},
                        {"pieces": {black_sq: 'p', white_end: 'P'}, "squares": []}
                    ],
                    "question": "Can black capture the white pawn en passant?",
                    # We ask if black can capture, but in the context it's asking about the current player (white)
                    "expected": "no",
                    "reasoning": f"White pawn moved 2 squares from {white_start} to {white_end}, black pawn at {black_sq} is adjacent"
                })
                case_num += 1

        return all_valid

    def _generate_condition1_violation(self, n_cases: int) -> List[Dict]:
        """
        Generate cases that violate condition 1: Pawn not from starting position
        Black pawn starts from rank 6 (or other non-7 rank) instead of rank 7
        """
        cases = []

        for i in range(n_cases):
            # Pick a file for black pawn (not edge files for better variation)
            black_file = random.choice(['b', 'c', 'd', 'e', 'f', 'g'])

            # Black pawn starts from rank 6 (not the starting rank 7)
            black_start = black_file + '6'
            black_end = black_file + '5'

            # White pawn on adjacent file at rank 5
            adjacent = self._adjacent_files(black_file)
            white_file = random.choice(adjacent)
            white_sq = white_file + '5'

            cases.append({
                "case_id": f"L3_cond1_violation_{i+1}",
                "type": "en_passant_basic",
                "subtype": "not_from_start",
                "states": [
                    {"pieces": {white_sq: 'P', black_start: 'p'}, "squares": []},
                    {"pieces": {white_sq: 'P', black_end: 'p'}, "squares": []}
                ],
                "question": "Can white capture the black pawn en passant?",
                "expected": "no",
                "reasoning": f"Black pawn did not start from rank 7 (started from {black_start})"
            })

        return cases

    def _generate_condition2_violation(self, n_cases: int) -> List[Dict]:
        """
        Generate cases that violate condition 2: Pawn moved only 1 square
        Black pawn moves from rank 6 to rank 5 (only 1 square, not 2)
        """
        cases = []

        for i in range(n_cases):
            # Pick a file for black pawn
            black_file = random.choice(['b', 'c', 'd', 'e', 'f', 'g'])

            # Black pawn moves only 1 square (from rank 6 to rank 5)
            black_start = black_file + '6'
            black_end = black_file + '5'

            # White pawn on adjacent file at rank 5
            adjacent = self._adjacent_files(black_file)
            white_file = random.choice(adjacent)
            white_sq = white_file + '5'

            cases.append({
                "case_id": f"L3_cond2_violation_{i+1}",
                "type": "en_passant_basic",
                "subtype": "moved_one_square",
                "states": [
                    {"pieces": {white_sq: 'P', black_start: 'p'}, "squares": []},
                    {"pieces": {white_sq: 'P', black_end: 'p'}, "squares": []}
                ],
                "question": "Can white capture the black pawn en passant?",
                "expected": "no",
                "reasoning": f"Black pawn only moved 1 square from {black_start} to {black_end}"
            })

        return cases

    def _generate_condition3_violation(self, n_cases: int) -> List[Dict]:
        """
        Generate cases that violate condition 3: Pawns not adjacent
        Black pawn and white pawn are not on adjacent files (2+ files apart)
        """
        cases = []

        for i in range(n_cases):
            # Pick a file for black pawn
            black_file = random.choice(['a', 'b', 'c', 'd'])
            black_start = black_file + '7'
            black_end = black_file + '5'

            # Pick a white pawn file that is NOT adjacent (2+ files away)
            black_file_idx = self.files.index(black_file)
            # Skip adjacent files
            non_adjacent_files = [
                f for i, f in enumerate(self.files)
                if abs(i - black_file_idx) >= 2
            ]

            if non_adjacent_files:
                white_file = random.choice(non_adjacent_files)
                white_sq = white_file + '5'

                cases.append({
                    "case_id": f"L3_cond3_violation_{i+1}",
                    "type": "en_passant_basic",
                    "subtype": "not_adjacent",
                    "states": [
                        {"pieces": {white_sq: 'P', black_start: 'p'}, "squares": []},
                        {"pieces": {white_sq: 'P', black_end: 'p'}, "squares": []}
                    ],
                    "question": "Can white capture the black pawn en passant?",
                    "expected": "no",
                    "reasoning": f"White pawn at {white_sq} is not adjacent to black pawn at {black_end}"
                })

        return cases

    def generate_all(self, n_cases: int = 100) -> List[Dict]:
        """
        Generate all Level 3 test cases

        Args:
            n_cases: Total number of cases

        Returns:
            List of test case dictionaries
        """
        all_cases = []

        # Calculate distribution
        # Valid: 25% (but max 28)
        n_valid = min(int(n_cases * 0.25), 28)

        # Invalid: 75% (split among 3 violation types)
        n_invalid = n_cases - n_valid
        n_cond1 = n_invalid // 3
        n_cond2 = n_invalid // 3
        n_cond3 = n_invalid - n_cond1 - n_cond2

        print(f"Generating valid en passant cases (max 28)...")
        # Generate all 28 valid combinations
        all_valid_combinations = self._generate_all_valid_combinations()

        # Sample n_valid cases from the 28 combinations
        if n_valid < len(all_valid_combinations):
            valid_cases = random.sample(all_valid_combinations, n_valid)
        else:
            valid_cases = all_valid_combinations

        all_cases.extend(valid_cases)
        print(f"  ✓ Generated {len(valid_cases)} valid cases")

        print(f"Generating condition violations...")

        # Condition 1 violation: Not from starting position
        cond1_cases = self._generate_condition1_violation(n_cond1)
        all_cases.extend(cond1_cases)
        print(
            f"  ✓ Generated {len(cond1_cases)} condition 1 violations (not from start)")

        # Condition 2 violation: Moved only 1 square
        cond2_cases = self._generate_condition2_violation(n_cond2)
        all_cases.extend(cond2_cases)
        print(
            f"  ✓ Generated {len(cond2_cases)} condition 2 violations (moved 1 square)")

        # Condition 3 violation: Not adjacent
        cond3_cases = self._generate_condition3_violation(n_cond3)
        all_cases.extend(cond3_cases)
        print(
            f"  ✓ Generated {len(cond3_cases)} condition 3 violations (not adjacent)")

        print(f"\n✓ Total generated: {len(all_cases)} Level 3 test cases")
        print(
            f"  Valid: {len(valid_cases)} ({len(valid_cases)/len(all_cases)*100:.1f}%)")
        print(
            f"  Invalid: {len(all_cases) - len(valid_cases)} ({(len(all_cases) - len(valid_cases))/len(all_cases)*100:.1f}%)")

        return all_cases
