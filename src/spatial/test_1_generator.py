"""
Automated test case generator for Spatial Test 1
Chess rule following: All 6 piece types (King, Queen, Rook, Bishop, Knight, Pawn)
"""

import random
from typing import List, Dict, Tuple


class SpatialTest1Generator:
    """Generate chess rule following test cases for all piece types"""

    def __init__(self, seed: int = 42):
        """
        Initialize generator

        Args:
            seed: Random seed for reproducibility
        """
        random.seed(seed)
        self.files = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
        self.ranks = ['1', '2', '3', '4', '5', '6', '7', '8']

    def _square_to_coords(self, square: str) -> Tuple[int, int]:
        """Convert square name to coordinates (0-7, 0-7)"""
        file = ord(square[0]) - ord('a')
        rank = int(square[1]) - 1
        return (file, rank)

    def _coords_to_square(self, file: int, rank: int) -> str:
        """Convert coordinates to square name"""
        if 0 <= file <= 7 and 0 <= rank <= 7:
            return chr(ord('a') + file) + str(rank + 1)
        return None

    def _random_square(self) -> str:
        """Generate random square"""
        file = random.choice(self.files)
        rank = random.choice(self.ranks)
        return file + rank

    def _is_valid_knight_move(self, from_sq: str, to_sq: str) -> bool:
        """Check if move is valid L-shape knight move"""
        f1, r1 = self._square_to_coords(from_sq)
        f2, r2 = self._square_to_coords(to_sq)

        df = abs(f2 - f1)
        dr = abs(r2 - r1)

        return (df == 2 and dr == 1) or (df == 1 and dr == 2)

    def _is_on_diagonal(self, from_sq: str, to_sq: str) -> bool:
        """Check if two squares are on same diagonal"""
        f1, r1 = self._square_to_coords(from_sq)
        f2, r2 = self._square_to_coords(to_sq)

        return abs(f2 - f1) == abs(r2 - r1) and from_sq != to_sq

    def _is_on_straight_line(self, from_sq: str, to_sq: str) -> bool:
        """Check if two squares are on same rank or file"""
        f1, r1 = self._square_to_coords(from_sq)
        f2, r2 = self._square_to_coords(to_sq)

        return (f1 == f2 or r1 == r2) and from_sq != to_sq

    def _get_squares_between(self, from_sq: str, to_sq: str) -> List[str]:
        """Get all squares between two squares (for blocking check)"""
        f1, r1 = self._square_to_coords(from_sq)
        f2, r2 = self._square_to_coords(to_sq)

        squares = []

        # Calculate direction
        df = 0 if f1 == f2 else (1 if f2 > f1 else -1)
        dr = 0 if r1 == r2 else (1 if r2 > r1 else -1)

        # Move step by step
        f, r = f1 + df, r1 + dr
        while (f, r) != (f2, r2):
            sq = self._coords_to_square(f, r)
            if sq:
                squares.append(sq)
            f += df
            r += dr

        return squares

    # ============= King Tests =============

    def generate_king_tests(self, n_per_type: int = 10) -> List[Dict]:
        """Generate king movement tests - can move one square in any direction"""
        cases = []
        n_pos = n_per_type // 2
        n_neg = n_per_type - n_pos

        # Positive cases: valid one-square moves
        attempts = 0
        while len([c for c in cases if c['expected'] == 'yes']) < n_pos and attempts < 1000:
            from_sq = self._random_square()
            f1, r1 = self._square_to_coords(from_sq)

            # Pick a random adjacent square
            directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1),
                          (0, 1), (1, -1), (1, 0), (1, 1)]
            df, dr = random.choice(directions)
            to_sq = self._coords_to_square(f1 + df, r1 + dr)

            if to_sq:
                cases.append({
                    'case_id': f'king_pos_{len([c for c in cases if c["expected"] == "yes"])+1}',
                    'type': 'king',
                    'subtype': 'valid_move',
                    'pieces': {from_sq: 'K'},
                    'squares': [to_sq],
                    'question': f'Can this king move to highlighted square?',
                    'expected': 'yes',
                    'reasoning': f'King can move one square in any direction'
                })
            attempts += 1

        # Negative cases: moves more than one square
        attempts = 0
        while len([c for c in cases if c['expected'] == 'no']) < n_neg and attempts < 1000:
            from_sq = self._random_square()
            to_sq = self._random_square()

            f1, r1 = self._square_to_coords(from_sq)
            f2, r2 = self._square_to_coords(to_sq)

            if abs(f2 - f1) > 1 or abs(r2 - r1) > 1:
                cases.append({
                    'case_id': f'king_neg_{len([c for c in cases if c["expected"] == "no"])+1}',
                    'type': 'king',
                    'subtype': 'too_far',
                    'pieces': {from_sq: 'K'},
                    'squares': [to_sq],
                    'question': f'Can this king move to highlighted square?',
                    'expected': 'no',
                    'reasoning': 'King can only move one square at a time'
                })
            attempts += 1

        return cases

    # ============= Queen Tests =============

    def generate_queen_tests(self, n_per_type: int = 10) -> List[Dict]:
        """Generate queen movement tests - combines rook and bishop movement"""
        cases = []
        n_clear = n_per_type // 3
        n_blocked = n_per_type // 3
        n_invalid = n_per_type - n_clear - n_blocked

        # Type 1: Clear path (diagonal or straight)
        attempts = 0
        while len([c for c in cases if c.get('subtype') == 'clear_path']) < n_clear and attempts < 1000:
            from_sq = self._random_square()
            to_sq = self._random_square()

            if self._is_on_diagonal(from_sq, to_sq) or self._is_on_straight_line(from_sq, to_sq):
                path_type = 'diagonal' if self._is_on_diagonal(
                    from_sq, to_sq) else 'straight'
                cases.append({
                    'case_id': f'queen_clear_{len([c for c in cases if c.get("subtype") == "clear_path"])+1}',
                    'type': 'queen',
                    'subtype': 'clear_path',
                    'pieces': {from_sq: 'Q'},
                    'squares': [to_sq],
                    'question': f'Can this queen move to highlighted square?',
                    'expected': 'yes',
                    'reasoning': f'Queen can move on clear {path_type} path'
                })
            attempts += 1

        # Type 2: Blocked path
        attempts = 0
        while len([c for c in cases if c.get('subtype') == 'blocked_path']) < n_blocked and attempts < 1000:
            from_sq = self._random_square()
            to_sq = self._random_square()

            if self._is_on_diagonal(from_sq, to_sq) or self._is_on_straight_line(from_sq, to_sq):
                between = self._get_squares_between(from_sq, to_sq)
                if between:
                    blocking_sq = random.choice(between)
                    cases.append({
                        'case_id': f'queen_blocked_{len([c for c in cases if c.get("subtype") == "blocked_path"])+1}',
                        'type': 'queen',
                        'subtype': 'blocked_path',
                        'pieces': {
                            from_sq: 'Q',
                            blocking_sq: 'P'
                        },
                        'squares': [to_sq],
                        'question': f'Can this queen move to highlighted square?',
                        'expected': 'no',
                        'reasoning': f'Path blocked by piece at {blocking_sq}'
                    })
            attempts += 1

        # Type 3: Invalid move (not diagonal or straight)
        attempts = 0
        while len([c for c in cases if c.get('subtype') == 'invalid_move']) < n_invalid and attempts < 1000:
            from_sq = self._random_square()
            to_sq = self._random_square()

            if not self._is_on_diagonal(from_sq, to_sq) and not self._is_on_straight_line(from_sq, to_sq) and from_sq != to_sq:
                cases.append({
                    'case_id': f'queen_invalid_{len([c for c in cases if c.get("subtype") == "invalid_move"])+1}',
                    'type': 'queen',
                    'subtype': 'invalid_move',
                    'pieces': {from_sq: 'Q'},
                    'squares': [to_sq],
                    'question': f'Can this queen move to highlighted square?',
                    'expected': 'no',
                    'reasoning': 'Queen only moves diagonally or straight'
                })
            attempts += 1

        return cases

    # ============= Rook Tests =============

    def generate_rook_tests(self, n_per_type: int = 10) -> List[Dict]:
        """Generate rook movement tests"""
        cases = []
        n_clear = n_per_type // 3
        n_blocked = n_per_type // 3
        n_invalid = n_per_type - n_clear - n_blocked

        # Type 1: Clear straight path (positive)
        attempts = 0
        while len([c for c in cases if c.get('subtype') == 'clear_path']) < n_clear and attempts < 1000:
            from_sq = self._random_square()
            to_sq = self._random_square()

            if self._is_on_straight_line(from_sq, to_sq):
                cases.append({
                    'case_id': f'rook_clear_{len([c for c in cases if c.get("subtype") == "clear_path"])+1}',
                    'type': 'rook',
                    'subtype': 'clear_path',
                    'pieces': {from_sq: 'R'},
                    'squares': [to_sq],
                    'question': f'Can this rook move to highlighted square?',
                    'expected': 'yes',
                    'reasoning': 'Rook on clear straight path'
                })
            attempts += 1

        # Type 2: Blocked straight path (negative)
        attempts = 0
        while len([c for c in cases if c.get('subtype') == 'blocked_path']) < n_blocked and attempts < 1000:
            from_sq = self._random_square()
            to_sq = self._random_square()

            if self._is_on_straight_line(from_sq, to_sq):
                between = self._get_squares_between(from_sq, to_sq)
                if between:
                    blocking_sq = random.choice(between)
                    cases.append({
                        'case_id': f'rook_blocked_{len([c for c in cases if c.get("subtype") == "blocked_path"])+1}',
                        'type': 'rook',
                        'subtype': 'blocked_path',
                        'pieces': {
                            from_sq: 'R',
                            blocking_sq: 'P'
                        },
                        'squares': [to_sq],
                        'question': f'Can this rook move to highlighted square?',
                        'expected': 'no',
                        'reasoning': f'Path blocked by piece at {blocking_sq}'
                    })
            attempts += 1

        # Type 3: Not on straight line (negative)
        attempts = 0
        while len([c for c in cases if c.get('subtype') == 'not_straight']) < n_invalid and attempts < 1000:
            from_sq = self._random_square()
            to_sq = self._random_square()

            if not self._is_on_straight_line(from_sq, to_sq) and from_sq != to_sq:
                cases.append({
                    'case_id': f'rook_invalid_{len([c for c in cases if c.get("subtype") == "not_straight"])+1}',
                    'type': 'rook',
                    'subtype': 'not_straight',
                    'pieces': {from_sq: 'R'},
                    'squares': [to_sq],
                    'question': f'Can this rook move to highlighted square?',
                    'expected': 'no',
                    'reasoning': 'Not on straight line - rook only moves straight'
                })
            attempts += 1

        return cases

    # ============= Bishop Tests =============

    def generate_bishop_tests(self, n_per_type: int = 10) -> List[Dict]:
        """Generate bishop movement tests"""
        cases = []
        n_clear = n_per_type // 3
        n_blocked = n_per_type // 3
        n_invalid = n_per_type - n_clear - n_blocked

        # Type 1: Clear diagonal path (positive)
        attempts = 0
        while len([c for c in cases if c.get('subtype') == 'clear_path']) < n_clear and attempts < 1000:
            from_sq = self._random_square()
            to_sq = self._random_square()

            if self._is_on_diagonal(from_sq, to_sq):
                cases.append({
                    'case_id': f'bishop_clear_{len([c for c in cases if c.get("subtype") == "clear_path"])+1}',
                    'type': 'bishop',
                    'subtype': 'clear_path',
                    'pieces': {from_sq: 'B'},
                    'squares': [to_sq],
                    'question': f'Can this bishop move to highlighted square?',
                    'expected': 'yes',
                    'reasoning': 'Bishop on clear diagonal path'
                })
            attempts += 1

        # Type 2: Blocked diagonal path (negative)
        attempts = 0
        while len([c for c in cases if c.get('subtype') == 'blocked_path']) < n_blocked and attempts < 1000:
            from_sq = self._random_square()
            to_sq = self._random_square()

            if self._is_on_diagonal(from_sq, to_sq):
                between = self._get_squares_between(from_sq, to_sq)
                if between:
                    blocking_sq = random.choice(between)
                    cases.append({
                        'case_id': f'bishop_blocked_{len([c for c in cases if c.get("subtype") == "blocked_path"])+1}',
                        'type': 'bishop',
                        'subtype': 'blocked_path',
                        'pieces': {
                            from_sq: 'B',
                            blocking_sq: 'P'
                        },
                        'squares': [to_sq],
                        'question': f'Can this bishop move to highlighted square?',
                        'expected': 'no',
                        'reasoning': f'Path blocked by piece at {blocking_sq}'
                    })
            attempts += 1

        # Type 3: Not on diagonal (negative)
        attempts = 0
        while len([c for c in cases if c.get('subtype') == 'not_diagonal']) < n_invalid and attempts < 1000:
            from_sq = self._random_square()
            to_sq = self._random_square()

            if not self._is_on_diagonal(from_sq, to_sq) and from_sq != to_sq:
                cases.append({
                    'case_id': f'bishop_invalid_{len([c for c in cases if c.get("subtype") == "not_diagonal"])+1}',
                    'type': 'bishop',
                    'subtype': 'not_diagonal',
                    'pieces': {from_sq: 'B'},
                    'squares': [to_sq],
                    'question': f'Can this bishop move to highlighted square?',
                    'expected': 'no',
                    'reasoning': 'Not on diagonal - bishop only moves diagonally'
                })
            attempts += 1

        return cases

    # ============= Knight Tests =============

    def generate_knight_tests(self, n_per_type: int = 10) -> List[Dict]:
        """Generate knight movement tests"""
        cases = []
        n_pos = n_per_type // 2
        n_neg = n_per_type - n_pos

        # Positive cases: valid L-shape moves
        attempts = 0
        while len([c for c in cases if c['expected'] == 'yes']) < n_pos and attempts < 1000:
            from_sq = self._random_square()
            to_sq = self._random_square()

            if self._is_valid_knight_move(from_sq, to_sq):
                cases.append({
                    'case_id': f'knight_pos_{len([c for c in cases if c["expected"] == "yes"])+1}',
                    'type': 'knight',
                    'subtype': 'valid_move',
                    'pieces': {from_sq: 'N'},
                    'squares': [to_sq],
                    'question': f'Can this knight move to highlighted square?',
                    'expected': 'yes',
                    'reasoning': f'Valid L-shape move from {from_sq} to {to_sq}'
                })
            attempts += 1

        # Negative cases: invalid moves
        attempts = 0
        while len([c for c in cases if c['expected'] == 'no']) < n_neg and attempts < 1000:
            from_sq = self._random_square()
            to_sq = self._random_square()

            if not self._is_valid_knight_move(from_sq, to_sq) and from_sq != to_sq:
                f1, r1 = self._square_to_coords(from_sq)
                f2, r2 = self._square_to_coords(to_sq)

                subtype = 'invalid_move'
                if f1 == f2 or r1 == r2:
                    subtype = 'straight_line'
                elif abs(f2 - f1) == abs(r2 - r1):
                    subtype = 'diagonal'

                cases.append({
                    'case_id': f'knight_neg_{len([c for c in cases if c["expected"] == "no"])+1}',
                    'type': 'knight',
                    'subtype': subtype,
                    'pieces': {from_sq: 'N'},
                    'squares': [to_sq],
                    'question': f'Can this knight move to highlighted square?',
                    'expected': 'no',
                    'reasoning': f'Invalid knight move: {subtype}'
                })
            attempts += 1

        return cases

    # ============= Pawn Tests =============

    def generate_pawn_tests(self, n_per_type: int = 10) -> List[Dict]:
        """Generate pawn movement tests - forward one square (or two from start)"""
        cases = []
        n_pos = n_per_type // 2
        n_neg = n_per_type - n_pos

        # Positive cases: valid forward moves
        pos_count = 0
        attempts = 0
        while pos_count < n_pos and attempts < 1000:
            # White pawn (moves up, rank increases)
            from_rank = random.choice(['2', '3', '4', '5', '6', '7'])
            from_file = random.choice(self.files)
            from_sq = from_file + from_rank

            # Can move one square forward
            if random.random() < 0.7:  # 70% one square
                to_sq = from_file + str(int(from_rank) + 1)
                reasoning = 'Pawn can move one square forward'
            else:  # 30% two squares from starting position
                if from_rank == '2':
                    to_sq = from_file + '4'
                    reasoning = 'Pawn can move two squares from starting position'
                else:
                    to_sq = from_file + str(int(from_rank) + 1)
                    reasoning = 'Pawn can move one square forward'

            # Make sure to_sq is valid
            if to_sq[1] in self.ranks:
                cases.append({
                    'case_id': f'pawn_pos_{pos_count+1}',
                    'type': 'pawn',
                    'subtype': 'valid_forward',
                    'pieces': {from_sq: 'P'},
                    'squares': [to_sq],
                    'question': f'Can this pawn move to highlighted square?',
                    'expected': 'yes',
                    'reasoning': reasoning
                })
                pos_count += 1
            attempts += 1

        # Negative cases: invalid moves
        neg_count = 0
        attempts = 0
        while neg_count < n_neg and attempts < 1000:
            from_rank = random.choice(['2', '3', '4', '5', '6', '7'])
            from_file = random.choice(self.files)
            from_sq = from_file + from_rank

            # Generate invalid moves
            invalid_type = random.choice(
                ['backward', 'sideways', 'diagonal_no_capture', 'too_far'])

            if invalid_type == 'backward':
                to_sq = from_file + str(int(from_rank) - 1)
                reasoning = 'Pawn cannot move backward'
            elif invalid_type == 'sideways':
                new_file = random.choice(
                    [f for f in self.files if f != from_file])
                to_sq = new_file + from_rank
                reasoning = 'Pawn cannot move sideways'
            elif invalid_type == 'diagonal_no_capture':
                # Diagonal move without capture piece
                file_idx = self.files.index(from_file)
                if file_idx > 0:
                    new_file = self.files[file_idx - 1]
                else:
                    new_file = self.files[file_idx + 1]
                to_sq = new_file + str(int(from_rank) + 1)
                reasoning = 'Pawn can only move diagonally when capturing'
            else:  # too_far
                if from_rank != '2':
                    to_sq = from_file + str(int(from_rank) + 2)
                    reasoning = 'Pawn can only move two squares from starting position'
                else:
                    to_sq = from_file + str(int(from_rank) + 3)
                    reasoning = 'Pawn cannot move three squares'

            # Make sure to_sq is valid
            if to_sq[1] in self.ranks:
                cases.append({
                    'case_id': f'pawn_neg_{neg_count+1}',
                    'type': 'pawn',
                    'subtype': invalid_type,
                    'pieces': {from_sq: 'P'},
                    'squares': [to_sq],
                    'question': f'Can this pawn move to highlighted square?',
                    'expected': 'no',
                    'reasoning': reasoning
                })
                neg_count += 1
            attempts += 1

        return cases

    # ============= Main Generation Method =============

    def generate_all(self, n_per_type: int = 10) -> List[Dict]:
        """
        Generate comprehensive Test 1 suite for all 6 piece types

        Args:
            n_per_type: Number of cases per piece type

        Returns:
            List of test case dictionaries
        """
        all_cases = []

        print(f"Generating King tests...")
        king_cases = self.generate_king_tests(n_per_type=n_per_type)
        all_cases.extend(king_cases)
        print(f"  Generated {len(king_cases)} king test cases")

        print(f"Generating Queen tests...")
        queen_cases = self.generate_queen_tests(n_per_type=n_per_type)
        all_cases.extend(queen_cases)
        print(f"  Generated {len(queen_cases)} queen test cases")

        print(f"Generating Rook tests...")
        rook_cases = self.generate_rook_tests(n_per_type=n_per_type)
        all_cases.extend(rook_cases)
        print(f"  Generated {len(rook_cases)} rook test cases")

        print(f"Generating Bishop tests...")
        bishop_cases = self.generate_bishop_tests(n_per_type=n_per_type)
        all_cases.extend(bishop_cases)
        print(f"  Generated {len(bishop_cases)} bishop test cases")

        print(f"Generating Knight tests...")
        knight_cases = self.generate_knight_tests(n_per_type=n_per_type)
        all_cases.extend(knight_cases)
        print(f"  Generated {len(knight_cases)} knight test cases")

        print(f"Generating Pawn tests...")
        pawn_cases = self.generate_pawn_tests(n_per_type=n_per_type)
        all_cases.extend(pawn_cases)
        print(f"  Generated {len(pawn_cases)} pawn test cases")

        print(f"\n✓ Total generated: {len(all_cases)} test cases")

        return all_cases
