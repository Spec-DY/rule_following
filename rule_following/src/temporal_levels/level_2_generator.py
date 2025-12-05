"""
Level 2 Generator: Path Blocked Capture
Tests capture with path blocking for Rook, Bishop, Queen
"""

import random
from typing import List, Dict, Tuple


class Level2Generator:
    """Generate Level 2 test cases - path blocked capture"""

    def __init__(self, seed: int = 42):
        """
        Initialize generator

        Args:
            seed: Random seed for reproducibility
        """
        random.seed(seed)
        self.files = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
        self.ranks = ['1', '2', '3', '4', '5', '6', '7', '8']

        # Piece types to test (Knight doesn't get blocked)
        self.piece_types = ['rook', 'bishop', 'queen']

    def _random_square(self) -> str:
        """Generate random square"""
        return random.choice(self.files) + random.choice(self.ranks)

    def _square_to_coords(self, square: str) -> Tuple[int, int]:
        """Convert square name to coordinates (0-7, 0-7)"""
        file = ord(square[0]) - ord('a')
        rank = int(square[1]) - 1
        return (file, rank)

    def _coords_to_square(self, file: int, rank: int) -> str:
        """Convert coordinates to square name"""
        if 0 <= file < 8 and 0 <= rank < 8:
            return chr(ord('a') + file) + str(rank + 1)
        return None

    def _get_random_piece_color(self) -> str:
        """Get random piece color"""
        return random.choice(['white', 'black'])

    def _piece_symbol(self, piece_type: str, color: str) -> str:
        """Get piece symbol"""
        symbols = {
            'rook': 'R' if color == 'white' else 'r',
            'bishop': 'B' if color == 'white' else 'b',
            'queen': 'Q' if color == 'white' else 'q',
            'knight': 'N' if color == 'white' else 'n',
            'pawn': 'P' if color == 'white' else 'p',
        }
        return symbols.get(piece_type, 'P' if color == 'white' else 'p')

    def _get_opposite_color(self, color: str) -> str:
        """Get opposite color"""
        return 'black' if color == 'white' else 'white'

    def _is_on_path(self, start: str, end: str, check: str, piece_type: str) -> bool:
        """
        Check if a square is on the path between start and end

        Args:
            start: Starting square
            end: Ending square
            check: Square to check
            piece_type: Type of piece (determines valid paths)

        Returns:
            True if check is on the path between start and end
        """
        if check == start or check == end:
            return False

        start_f, start_r = self._square_to_coords(start)
        end_f, end_r = self._square_to_coords(end)
        check_f, check_r = self._square_to_coords(check)

        # For rook: straight line (horizontal or vertical)
        if piece_type == 'rook':
            # Same file (vertical)
            if start_f == end_f == check_f:
                min_r = min(start_r, end_r)
                max_r = max(start_r, end_r)
                return min_r < check_r < max_r
            # Same rank (horizontal)
            elif start_r == end_r == check_r:
                min_f = min(start_f, end_f)
                max_f = max(start_f, end_f)
                return min_f < check_f < max_f
            return False

        # For bishop: diagonal
        elif piece_type == 'bishop':
            # Check if all three points are on same diagonal
            if abs(start_f - end_f) != abs(start_r - end_r):
                return False
            if abs(start_f - check_f) != abs(start_r - check_r):
                return False
            if abs(end_f - check_f) != abs(end_r - check_r):
                return False

            # Check if check is between start and end
            if start_f < end_f:
                return start_f < check_f < end_f
            else:
                return end_f < check_f < start_f

        # For queen: can move like rook or bishop
        else:
            # Try rook movement
            if start_f == end_f or start_r == end_r:
                return self._is_on_path(start, end, check, 'rook')
            # Try bishop movement
            else:
                return self._is_on_path(start, end, check, 'bishop')

    # ===== ROOK CASES =====

    def _generate_rook_capture_case(self, case_type: str, case_num: int) -> Dict:
        """Generate a rook capture test case"""
        attacker_color = self._get_random_piece_color()
        target_color = self._get_opposite_color(attacker_color)

        max_attempts = 100
        for attempt in range(max_attempts):
            # Pick random start square for attacker
            start_sq = self._random_square()
            start_f, start_r = self._square_to_coords(start_sq)

            # Pick target square (straight line from start)
            move_type = random.choice(['horizontal', 'vertical'])
            if move_type == 'horizontal':
                # Same rank, different file
                distance = random.randint(3, 6)
                direction = random.choice([-1, 1])
                target_f = start_f + direction * distance
                target_r = start_r
            else:  # vertical
                # Same file, different rank
                distance = random.randint(3, 6)
                direction = random.choice([-1, 1])
                target_f = start_f
                target_r = start_r + direction * distance

            if not (0 <= target_f < 8 and 0 <= target_r < 8):
                continue

            target_sq = self._coords_to_square(target_f, target_r)

            if case_type == 'valid':
                # Blocker NOT on path
                blocker_sq = self._get_square_not_on_path(
                    start_sq, target_sq, 'rook')
                if blocker_sq:
                    return self._create_capture_case(
                        'rook', attacker_color, start_sq, target_sq, blocker_sq,
                        f"L2_rook_valid_{case_num}", "valid", "yes",
                        "Path is clear, capture is legal"
                    )

            elif case_type == 'blocked':
                # Blocker ON path
                blocker_sq = self._get_square_on_path(
                    start_sq, target_sq, 'rook')
                if blocker_sq:
                    return self._create_capture_case(
                        'rook', attacker_color, start_sq, target_sq, blocker_sq,
                        f"L2_rook_blocked_{case_num}", "blocked", "no",
                        f"Path is blocked by piece at {blocker_sq}"
                    )

            else:  # invalid movement pattern
                # Pick a square not in straight line (diagonal move)
                distance = random.randint(2, 4)
                dir_f = random.choice([-1, 1])
                dir_r = random.choice([-1, 1])
                target_f = start_f + dir_f * distance
                target_r = start_r + dir_r * distance

                if 0 <= target_f < 8 and 0 <= target_r < 8:
                    target_sq = self._coords_to_square(target_f, target_r)
                    blocker_sq = self._random_square()
                    while blocker_sq == start_sq or blocker_sq == target_sq:
                        blocker_sq = self._random_square()

                    return self._create_capture_case(
                        'rook', attacker_color, start_sq, target_sq, blocker_sq,
                        f"L2_rook_invalid_{case_num}", "invalid_pattern", "no",
                        "Rook cannot move diagonally"
                    )

        # Fallback if we couldn't generate after max attempts
        print(
            f"  Warning: Could not generate rook {case_type} case after {max_attempts} attempts")
        return None

    # ===== BISHOP CASES =====

    def _generate_bishop_capture_case(self, case_type: str, case_num: int) -> Dict:
        """Generate a bishop capture test case"""
        attacker_color = self._get_random_piece_color()
        target_color = self._get_opposite_color(attacker_color)

        max_attempts = 100
        for attempt in range(max_attempts):
            # Pick random start square for attacker
            start_sq = self._random_square()
            start_f, start_r = self._square_to_coords(start_sq)

            if case_type == 'invalid':
                # Generate straight line move (invalid for bishop)
                move_type = random.choice(['horizontal', 'vertical'])
                distance = random.randint(2, 5)
                direction = random.choice([-1, 1])

                if move_type == 'horizontal':
                    target_f = start_f + direction * distance
                    target_r = start_r
                else:
                    target_f = start_f
                    target_r = start_r + direction * distance

                if 0 <= target_f < 8 and 0 <= target_r < 8:
                    target_sq = self._coords_to_square(target_f, target_r)
                    blocker_sq = self._random_square()
                    while blocker_sq == start_sq or blocker_sq == target_sq:
                        blocker_sq = self._random_square()

                    return self._create_capture_case(
                        'bishop', attacker_color, start_sq, target_sq, blocker_sq,
                        f"L2_bishop_invalid_{case_num}", "invalid_pattern", "no",
                        "Bishop cannot move in straight line"
                    )
            else:
                # Pick target square (diagonal from start)
                distance = random.randint(3, 5)
                dir_f = random.choice([-1, 1])
                dir_r = random.choice([-1, 1])
                target_f = start_f + dir_f * distance
                target_r = start_r + dir_r * distance

                if not (0 <= target_f < 8 and 0 <= target_r < 8):
                    continue

                target_sq = self._coords_to_square(target_f, target_r)

                if case_type == 'valid':
                    # Blocker NOT on path
                    blocker_sq = self._get_square_not_on_path(
                        start_sq, target_sq, 'bishop')
                    if blocker_sq:
                        return self._create_capture_case(
                            'bishop', attacker_color, start_sq, target_sq, blocker_sq,
                            f"L2_bishop_valid_{case_num}", "valid", "yes",
                            "Path is clear, capture is legal"
                        )

                elif case_type == 'blocked':
                    # Blocker ON path
                    blocker_sq = self._get_square_on_path(
                        start_sq, target_sq, 'bishop')
                    if blocker_sq:
                        return self._create_capture_case(
                            'bishop', attacker_color, start_sq, target_sq, blocker_sq,
                            f"L2_bishop_blocked_{case_num}", "blocked", "no",
                            f"Path is blocked by piece at {blocker_sq}"
                        )

        # Fallback
        print(
            f"  Warning: Could not generate bishop {case_type} case after {max_attempts} attempts")
        return None

    # ===== QUEEN CASES =====

    def _generate_queen_capture_case(self, case_type: str, case_num: int) -> Dict:
        """Generate a queen capture test case"""
        attacker_color = self._get_random_piece_color()

        max_attempts = 100
        for attempt in range(max_attempts):
            start_sq = self._random_square()
            start_f, start_r = self._square_to_coords(start_sq)

            if case_type == 'invalid':
                # Generate L-shape move (invalid for queen)
                l_moves = [
                    (2, 1), (2, -1), (-2, 1), (-2, -1),
                    (1, 2), (1, -2), (-1, 2), (-1, -2)
                ]
                df, dr = random.choice(l_moves)
                target_f = start_f + df
                target_r = start_r + dr

                if 0 <= target_f < 8 and 0 <= target_r < 8:
                    target_sq = self._coords_to_square(target_f, target_r)
                    blocker_sq = self._random_square()
                    while blocker_sq == start_sq or blocker_sq == target_sq:
                        blocker_sq = self._random_square()

                    return self._create_capture_case(
                        'queen', attacker_color, start_sq, target_sq, blocker_sq,
                        f"L2_queen_invalid_{case_num}", "invalid_pattern", "no",
                        "Queen cannot move in L-shape"
                    )
            else:
                # Queen can move like rook or bishop
                move_like = random.choice(['rook', 'bishop'])

                if move_like == 'rook':
                    # Straight line movement
                    move_type = random.choice(['horizontal', 'vertical'])
                    distance = random.randint(3, 6)
                    direction = random.choice([-1, 1])

                    if move_type == 'horizontal':
                        target_f = start_f + direction * distance
                        target_r = start_r
                    else:
                        target_f = start_f
                        target_r = start_r + direction * distance
                else:
                    # Diagonal movement
                    distance = random.randint(3, 5)
                    dir_f = random.choice([-1, 1])
                    dir_r = random.choice([-1, 1])
                    target_f = start_f + dir_f * distance
                    target_r = start_r + dir_r * distance

                if not (0 <= target_f < 8 and 0 <= target_r < 8):
                    continue

                target_sq = self._coords_to_square(target_f, target_r)

                if case_type == 'valid':
                    blocker_sq = self._get_square_not_on_path(
                        start_sq, target_sq, 'queen')
                    if blocker_sq:
                        return self._create_capture_case(
                            'queen', attacker_color, start_sq, target_sq, blocker_sq,
                            f"L2_queen_valid_{case_num}", "valid", "yes",
                            "Path is clear, capture is legal"
                        )

                elif case_type == 'blocked':
                    blocker_sq = self._get_square_on_path(
                        start_sq, target_sq, 'queen')
                    if blocker_sq:
                        return self._create_capture_case(
                            'queen', attacker_color, start_sq, target_sq, blocker_sq,
                            f"L2_queen_blocked_{case_num}", "blocked", "no",
                            f"Path is blocked by piece at {blocker_sq}"
                        )

        # Fallback
        print(
            f"  Warning: Could not generate queen {case_type} case after {max_attempts} attempts")
        return None

    # ===== HELPER METHODS =====

    def _get_square_on_path(self, start: str, end: str, piece_type: str) -> str:
        """Get a square that IS on the path between start and end"""
        start_f, start_r = self._square_to_coords(start)
        end_f, end_r = self._square_to_coords(end)

        if piece_type in ['rook', 'queen']:
            # Try straight line
            if start_f == end_f:  # Vertical
                min_r, max_r = sorted([start_r, end_r])
                if max_r - min_r > 1:
                    middle_r = random.randint(min_r + 1, max_r - 1)
                    return self._coords_to_square(start_f, middle_r)
            elif start_r == end_r:  # Horizontal
                min_f, max_f = sorted([start_f, end_f])
                if max_f - min_f > 1:
                    middle_f = random.randint(min_f + 1, max_f - 1)
                    return self._coords_to_square(middle_f, start_r)

        if piece_type in ['bishop', 'queen']:
            # Try diagonal
            if abs(end_f - start_f) == abs(end_r - start_r) and abs(end_f - start_f) > 1:
                dir_f = 1 if end_f > start_f else -1
                dir_r = 1 if end_r > start_r else -1
                distance = abs(end_f - start_f)
                middle_dist = random.randint(1, distance - 1)
                middle_f = start_f + dir_f * middle_dist
                middle_r = start_r + dir_r * middle_dist
                return self._coords_to_square(middle_f, middle_r)

        return None

    def _get_square_not_on_path(self, start: str, end: str, piece_type: str) -> str:
        """Get a square that is NOT on the path between start and end"""
        for _ in range(50):  # Try up to 50 times
            blocker = self._random_square()
            if blocker != start and blocker != end:
                if not self._is_on_path(start, end, blocker, piece_type):
                    return blocker
        return None

    def _create_capture_case(self, piece_type: str, attacker_color: str,
                             start_sq: str, target_sq: str, blocker_sq: str,
                             case_id: str, subtype: str, expected: str,
                             reasoning: str) -> Dict:
        """Create a capture test case"""
        attacker_symbol = self._piece_symbol(piece_type, attacker_color)
        target_symbol = self._piece_symbol(
            'pawn', self._get_opposite_color(attacker_color))
        blocker_symbol = self._piece_symbol(
            random.choice(['knight', 'pawn']),
            random.choice(['white', 'black'])
        )

        return {
            "case_id": case_id,
            "type": "path_blocked_capture",
            "subtype": subtype,
            "piece_type": piece_type,
            "attacker_color": attacker_color,
            "states": [
                {
                    "pieces": {
                        start_sq: attacker_symbol,
                        target_sq: target_symbol,
                        blocker_sq: blocker_symbol
                    },
                    "squares": []
                },
                {
                    "pieces": {
                        target_sq: attacker_symbol,
                        blocker_sq: blocker_symbol
                    },
                    "squares": []
                }
            ],
            "question": "Could this capture happen according to chess rules?",
            "expected": expected,
            "reasoning": reasoning
        }

    # ===== MAIN GENERATION =====

    def generate_all(self, n_cases: int = 90) -> List[Dict]:
        """
        Generate all Level 2 test cases

        Args:
            n_cases: Total number of cases

        Returns:
            List of test case dictionaries
        """
        all_cases = []

        # Distribute cases evenly across 3 piece types
        cases_per_piece = n_cases // 3
        remainder = n_cases % 3

        for idx, piece_type in enumerate(self.piece_types):
            n_piece_cases = cases_per_piece + (1 if idx < remainder else 0)

            # 33% valid, 33% blocked, 34% invalid
            n_valid = n_piece_cases // 3
            n_blocked = n_piece_cases // 3
            n_invalid = n_piece_cases - n_valid - n_blocked

            print(
                f"Generating {piece_type} tests: {n_valid} valid + {n_blocked} blocked + {n_invalid} invalid...")

            # Generate valid cases
            valid_count = 0
            for i in range(n_valid):
                if piece_type == 'rook':
                    case = self._generate_rook_capture_case(
                        'valid', valid_count+1)
                elif piece_type == 'bishop':
                    case = self._generate_bishop_capture_case(
                        'valid', valid_count+1)
                else:
                    case = self._generate_queen_capture_case(
                        'valid', valid_count+1)

                if case:
                    all_cases.append(case)
                    valid_count += 1

            # Generate blocked cases
            blocked_count = 0
            for i in range(n_blocked):
                if piece_type == 'rook':
                    case = self._generate_rook_capture_case(
                        'blocked', blocked_count+1)
                elif piece_type == 'bishop':
                    case = self._generate_bishop_capture_case(
                        'blocked', blocked_count+1)
                else:
                    case = self._generate_queen_capture_case(
                        'blocked', blocked_count+1)

                if case:
                    all_cases.append(case)
                    blocked_count += 1

            # Generate invalid cases
            invalid_count = 0
            for i in range(n_invalid):
                if piece_type == 'rook':
                    case = self._generate_rook_capture_case(
                        'invalid', invalid_count+1)
                elif piece_type == 'bishop':
                    case = self._generate_bishop_capture_case(
                        'invalid', invalid_count+1)
                else:
                    case = self._generate_queen_capture_case(
                        'invalid', invalid_count+1)

                if case:
                    all_cases.append(case)
                    invalid_count += 1

            print(
                f"  ✓ Generated {valid_count} valid, {blocked_count} blocked, {invalid_count} invalid cases")

        print(f"\n✓ Total generated: {len(all_cases)} Level 2 test cases")
        return all_cases
