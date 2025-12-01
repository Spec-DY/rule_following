"""
Condition Test 1 Generator
Generates test cases where models must count how many pieces can attack a target.
"""

import random
from typing import List, Dict, Tuple
from dataclasses import dataclass


@dataclass
class ConditionTest1Case:
    """Single test case for Condition Test 1"""
    level: int  # 1-6, number of potential attackers
    target_piece: str  # e.g., "p" for black pawn
    target_square: str  # e.g., "e5"
    attacker_pieces: Dict[str, str]  # {square: piece}
    # Number of pieces that can actually attack (0 to level)
    correct_answer: int
    test_id: str

    def to_dict(self):
        # Create pieces dict: target + attackers
        pieces = {self.target_square: self.target_piece}
        pieces.update(self.attacker_pieces)

        # Determine piece description
        piece_type = self.target_piece.upper()
        piece_color = "White" if self.target_piece.isupper() else "Black"
        attacker_color = "Black" if piece_color == "White" else "White"

        piece_names = {
            'P': 'Pawn', 'N': 'Knight', 'B': 'Bishop',
            'R': 'Rook', 'Q': 'Queen', 'K': 'King'
        }
        piece_name = piece_names.get(piece_type, 'Piece')

        return {
            "case_id": self.test_id,
            "type": "threat_count",
            "level": self.level,
            "target_piece": self.target_piece,
            "target_square": self.target_square,
            "pieces": pieces,
            "highlighted_squares": [],
            "label": f"This chess position shows a {piece_color} {piece_name} on {self.target_square}.",
            "question": f"How many {attacker_color} pieces can attack the {piece_color} {piece_name} on {self.target_square}?",
            "expected": str(self.correct_answer),
        }


class ConditionTest1Generator:
    """
    Generates test cases for Condition Test 1: Threat Count
    Tests counting how many pieces can attack a target.
    """

    FILES = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    RANKS = ['1', '2', '3', '4', '5', '6', '7', '8']

    def __init__(self, seed: int = 42):
        """Initialize generator with random seed"""
        self.seed = seed
        random.seed(seed)
        self.test_counter = 0

    def generate_all(self, n_per_level: int = 10) -> List[Dict]:
        """
        Generate all test cases for levels 1-6

        Args:
            n_per_level: Number of test cases per level

        Returns:
            List of all generated test case dictionaries
        """
        all_cases = []

        for level in range(1, 7):  # Levels 1-6
            print(f"  Generating Level {level} cases...")
            cases = self.generate_level_cases(level, n_per_level)
            all_cases.extend(cases)

        print(f"\n✓ Total generated: {len(all_cases)} test cases")
        return all_cases

    def generate_level_cases(self, level: int, n_cases: int) -> List[Dict]:
        """
        Generate test cases for a specific level
        Ensures diverse distribution of answers (0 to level)

        Args:
            level: Number of potential attackers (1-6)
            n_cases: Number of cases to generate

        Returns:
            List of test case dictionaries
        """
        cases = []

        # Create distribution across all possible answers (0 to level)
        # For example, Level 3 should have cases with answers: 0, 1, 2, 3
        possible_answers = list(range(level + 1))  # [0, 1, 2, ..., level]

        # Distribute cases across answers as evenly as possible
        cases_per_answer = n_cases // len(possible_answers)
        remainder = n_cases % len(possible_answers)

        for answer_idx, target_count in enumerate(possible_answers):
            # Add extra case to some answers to handle remainder
            n_for_this_answer = cases_per_answer + \
                (1 if answer_idx < remainder else 0)

            for _ in range(n_for_this_answer):
                case = self._generate_single_case(level, target_count)
                if case:
                    cases.append(case.to_dict())

        return cases

    def _generate_single_case(self, level: int, target_count: int) -> ConditionTest1Case:
        """
        Generate a single test case with exactly target_count attackers

        Args:
            level: Total number of potential attackers
            target_count: Number of attackers that should actually be able to attack

        Returns:
            A single test case
        """
        max_attempts = 100

        for attempt in range(max_attempts):
            # Step 1: Choose target piece and square
            target_color = random.choice(['white', 'black'])
            target_piece_type = random.choice(['P', 'N', 'B', 'R', 'Q', 'K'])
            target_square = self._random_square()

            # Format piece
            if target_color == 'white':
                target_piece = target_piece_type
            else:
                target_piece = target_piece_type.lower()

            # Step 2: Generate attacking pieces
            attacker_color = 'black' if target_color == 'white' else 'white'
            occupied_squares = {target_square}

            attacker_pieces = {}
            attacking_pieces_list = self._select_attacker_types(level)

            # Decide which pieces should be able to attack
            # Randomly select target_count pieces to be actual attackers
            actual_attacker_indices = random.sample(range(level), target_count)

            # Place each piece
            for piece_idx, piece_type in enumerate(attacking_pieces_list):
                should_attack = piece_idx in actual_attacker_indices

                square = self._place_piece(
                    piece_type,
                    target_square,
                    occupied_squares,
                    can_attack=should_attack
                )

                if square:
                    occupied_squares.add(square)
                    # Format piece
                    if attacker_color == 'white':
                        attacker_pieces[square] = piece_type
                    else:
                        attacker_pieces[square] = piece_type.lower()

            # Verify we placed all pieces and answer is correct
            if len(attacker_pieces) == level:
                actual_count = self._count_attackers(
                    target_square, attacker_pieces, occupied_squares)

                if actual_count == target_count:
                    self.test_counter += 1
                    return ConditionTest1Case(
                        level=level,
                        target_piece=target_piece,
                        target_square=target_square,
                        attacker_pieces=attacker_pieces,
                        correct_answer=target_count,
                        test_id=f"cond1_L{level}_{self.test_counter:03d}"
                    )

        print(
            f"Warning: Failed to generate case for level {level}, count {target_count} after {max_attempts} attempts")
        return None

    def _select_attacker_types(self, level: int) -> List[str]:
        """Select which piece types will be potential attackers"""
        piece_types = ['R', 'B', 'N', 'Q', 'K', 'P']

        if level == 6:
            return random.sample(piece_types, 6)
        else:
            return random.sample(piece_types, level)

    def _place_piece(self, piece_type: str, target_square: str,
                     occupied: set, can_attack: bool) -> str:
        """
        Place a piece that either can or cannot attack the target

        Args:
            piece_type: Type of piece (R/B/N/Q/K/P)
            target_square: Target square
            occupied: Set of already occupied squares
            can_attack: If True, place where it CAN attack; if False, place where it CANNOT

        Returns:
            Square where piece was placed, or None if failed
        """
        available_squares = [
            sq for sq in self._all_squares() if sq not in occupied
        ]

        random.shuffle(available_squares)

        for square in available_squares:
            if self._can_piece_attack(piece_type, square, target_square, occupied):
                if can_attack:
                    return square
            else:
                if not can_attack:
                    return square

        return None

    def _count_attackers(self, target_square: str, attacker_pieces: Dict[str, str],
                         occupied: set) -> int:
        """
        Count how many pieces can actually attack the target

        Returns:
            Number of attackers
        """
        count = 0
        for attacker_square, piece in attacker_pieces.items():
            if self._can_piece_attack(piece, attacker_square, target_square, occupied):
                count += 1
        return count

    def _can_piece_attack(self, piece_type: str, from_square: str,
                          to_square: str, occupied: set) -> bool:
        """Check if a piece can attack a target square"""
        if from_square == to_square:
            return False

        piece_upper = piece_type.upper()

        if piece_upper == 'R':
            return self._rook_can_attack(from_square, to_square, occupied)
        elif piece_upper == 'B':
            return self._bishop_can_attack(from_square, to_square, occupied)
        elif piece_upper == 'N':
            return self._knight_can_attack(from_square, to_square)
        elif piece_upper == 'Q':
            return (self._rook_can_attack(from_square, to_square, occupied) or
                    self._bishop_can_attack(from_square, to_square, occupied))
        elif piece_upper == 'K':
            return self._king_can_attack(from_square, to_square)
        elif piece_upper == 'P':
            return self._pawn_can_attack(piece_type, from_square, to_square)

        return False

    def _rook_can_attack(self, from_sq: str, to_sq: str, occupied: set) -> bool:
        """Check if rook can attack (same rank or file, no obstacles)"""
        from_file, from_rank = from_sq[0], int(from_sq[1])
        to_file, to_rank = to_sq[0], int(to_sq[1])

        if from_file != to_file and from_rank != to_rank:
            return False

        if from_file == to_file:  # Vertical
            start, end = sorted([from_rank, to_rank])
            for rank in range(start + 1, end):
                if f"{from_file}{rank}" in occupied:
                    return False
        else:  # Horizontal
            start_idx = self.FILES.index(from_file)
            end_idx = self.FILES.index(to_file)
            start, end = sorted([start_idx, end_idx])
            for file_idx in range(start + 1, end):
                if f"{self.FILES[file_idx]}{from_rank}" in occupied:
                    return False

        return True

    def _bishop_can_attack(self, from_sq: str, to_sq: str, occupied: set) -> bool:
        """Check if bishop can attack (diagonal, no obstacles)"""
        from_file_idx = self.FILES.index(from_sq[0])
        from_rank = int(from_sq[1])
        to_file_idx = self.FILES.index(to_sq[0])
        to_rank = int(to_sq[1])

        if abs(from_file_idx - to_file_idx) != abs(from_rank - to_rank):
            return False

        file_step = 1 if to_file_idx > from_file_idx else -1
        rank_step = 1 if to_rank > from_rank else -1

        current_file = from_file_idx + file_step
        current_rank = from_rank + rank_step

        while current_file != to_file_idx:
            sq = f"{self.FILES[current_file]}{current_rank}"
            if sq in occupied:
                return False
            current_file += file_step
            current_rank += rank_step

        return True

    def _knight_can_attack(self, from_sq: str, to_sq: str) -> bool:
        """Check if knight can attack (L-shape)"""
        from_file_idx = self.FILES.index(from_sq[0])
        from_rank = int(from_sq[1])
        to_file_idx = self.FILES.index(to_sq[0])
        to_rank = int(to_sq[1])

        file_diff = abs(from_file_idx - to_file_idx)
        rank_diff = abs(from_rank - to_rank)

        return (file_diff == 2 and rank_diff == 1) or (file_diff == 1 and rank_diff == 2)

    def _king_can_attack(self, from_sq: str, to_sq: str) -> bool:
        """Check if king can attack (one square any direction)"""
        from_file_idx = self.FILES.index(from_sq[0])
        from_rank = int(from_sq[1])
        to_file_idx = self.FILES.index(to_sq[0])
        to_rank = int(to_sq[1])

        file_diff = abs(from_file_idx - to_file_idx)
        rank_diff = abs(from_rank - to_rank)

        return file_diff <= 1 and rank_diff <= 1 and (file_diff + rank_diff) > 0

    def _pawn_can_attack(self, piece: str, from_sq: str, to_sq: str) -> bool:
        """Check if pawn can attack (diagonal forward one square)"""
        from_file_idx = self.FILES.index(from_sq[0])
        from_rank = int(from_sq[1])
        to_file_idx = self.FILES.index(to_sq[0])
        to_rank = int(to_sq[1])

        file_diff = abs(from_file_idx - to_file_idx)

        if file_diff != 1:
            return False

        # White pawn attacks upward, black pawn attacks downward
        if piece.isupper():  # White pawn
            return to_rank == from_rank + 1
        else:  # Black pawn
            return to_rank == from_rank - 1

    def _random_square(self) -> str:
        """Generate a random square"""
        file = random.choice(self.FILES)
        rank = random.choice(self.RANKS)
        return f"{file}{rank}"

    def _all_squares(self) -> List[str]:
        """Get all squares on the board"""
        return [f"{file}{rank}" for file in self.FILES for rank in self.RANKS]
