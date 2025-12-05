"""
Level 5 Generator: Castling + 2 Check Rules
Tests castling with 2 out of 3 check-related constraints
"""

import random
from typing import List, Dict, Tuple, Set
from collections import defaultdict


class Level5Generator:
    """Generate Level 5 test cases - castling with 2 check rules"""

    # Maximum pieces per color in standard chess
    PIECE_LIMITS = {
        'king': 1,
        'queen': 1,
        'rook': 2,
        'bishop': 2,
        'knight': 2
    }

    def __init__(self, seed: int = 42):
        """
        Initialize generator

        Args:
            seed: Random seed for reproducibility
        """
        random.seed(seed)
        self.files = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
        self.ranks = ['1', '2', '3', '4', '5', '6', '7', '8']

        # Castling configurations with path squares that must be empty
        self.castling_configs = {
            'white_kingside': {
                'king_start': 'e1', 'king_end': 'g1',
                'rook_start': 'h1', 'rook_end': 'f1',
                'through_sq': 'f1', 'color': 'white',
                'king_symbol': 'K', 'rook_symbol': 'R',
                'path_squares': ['f1', 'g1']
            },
            'white_queenside': {
                'king_start': 'e1', 'king_end': 'c1',
                'rook_start': 'a1', 'rook_end': 'd1',
                'through_sq': 'd1', 'color': 'white',
                'king_symbol': 'K', 'rook_symbol': 'R',
                'path_squares': ['b1', 'c1', 'd1']
            },
            'black_kingside': {
                'king_start': 'e8', 'king_end': 'g8',
                'rook_start': 'h8', 'rook_end': 'f8',
                'through_sq': 'f8', 'color': 'black',
                'king_symbol': 'k', 'rook_symbol': 'r',
                'path_squares': ['f8', 'g8']
            },
            'black_queenside': {
                'king_start': 'e8', 'king_end': 'c8',
                'rook_start': 'a8', 'rook_end': 'd8',
                'through_sq': 'd8', 'color': 'black',
                'king_symbol': 'k', 'rook_symbol': 'r',
                'path_squares': ['b8', 'c8', 'd8']
            }
        }

        # Check rule combinations (3 choose 2)
        self.check_combinations = [
            ['in', 'through'],
            ['in', 'into'],
            ['through', 'into']
        ]

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

    def _get_piece_type_from_symbol(self, symbol: str) -> Tuple[str, str]:
        """Get piece type and color from symbol"""
        symbol_map = {
            'K': ('king', 'white'), 'k': ('king', 'black'),
            'Q': ('queen', 'white'), 'q': ('queen', 'black'),
            'R': ('rook', 'white'), 'r': ('rook', 'black'),
            'B': ('bishop', 'white'), 'b': ('bishop', 'black'),
            'N': ('knight', 'white'), 'n': ('knight', 'black'),
            'P': ('pawn', 'white'), 'p': ('pawn', 'black')
        }
        return symbol_map.get(symbol, (None, None))

    def _can_add_piece(self, piece_type: str, color: str, piece_counts: Dict) -> bool:
        """Check if we can add another piece of this type and color"""
        key = (piece_type, color)
        current = piece_counts.get(key, 0)
        limit = self.PIECE_LIMITS.get(piece_type, 2)
        return current < limit

    def _add_piece_to_counts(self, piece_type: str, color: str, piece_counts: Dict):
        """Add a piece to the count tracker"""
        key = (piece_type, color)
        piece_counts[key] = piece_counts.get(key, 0) + 1

    def _get_attacker_positions(self, target_sq: str, attacker_type: str,
                                forbidden: Set[str]) -> List[str]:
        """
        Get valid positions for an attacker to attack target square
        """
        target_f, target_r = self._square_to_coords(target_sq)
        positions = []

        if attacker_type == 'rook' or attacker_type == 'queen':
            for f in range(8):
                if f != target_f:
                    sq = self._coords_to_square(f, target_r)
                    if sq and sq not in forbidden:
                        positions.append(sq)
            for r in range(8):
                if r != target_r:
                    sq = self._coords_to_square(target_f, r)
                    if sq and sq not in forbidden:
                        positions.append(sq)

        if attacker_type == 'bishop' or attacker_type == 'queen':
            for d in range(1, 8):
                for df, dr in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
                    sq = self._coords_to_square(
                        target_f + df * d, target_r + dr * d)
                    if sq and sq not in forbidden:
                        positions.append(sq)

        if attacker_type == 'knight':
            for df, dr in [(2, 1), (2, -1), (-2, 1), (-2, -1),
                           (1, 2), (1, -2), (-1, 2), (-1, -2)]:
                sq = self._coords_to_square(target_f + df, target_r + dr)
                if sq and sq not in forbidden:
                    positions.append(sq)

        return positions

    def _get_attacker_symbol(self, attacker_type: str, color: str) -> str:
        """Get piece symbol for attacker"""
        symbols = {
            ('rook', 'white'): 'R', ('rook', 'black'): 'r',
            ('bishop', 'white'): 'B', ('bishop', 'black'): 'b',
            ('knight', 'white'): 'N', ('knight', 'black'): 'n',
            ('queen', 'white'): 'Q', ('queen', 'black'): 'q'
        }
        return symbols.get((attacker_type, color), 'p')

    def _can_attack(self, from_sq: str, to_sq: str, piece_type: str) -> bool:
        """Check if a piece can attack from from_sq to to_sq"""
        from_f, from_r = self._square_to_coords(from_sq)
        to_f, to_r = self._square_to_coords(to_sq)

        df = abs(to_f - from_f)
        dr = abs(to_r - from_r)

        if piece_type == 'rook':
            return (df == 0 and dr > 0) or (dr == 0 and df > 0)
        elif piece_type == 'bishop':
            return df == dr and df > 0
        elif piece_type == 'queen':
            return (df == 0 and dr > 0) or (dr == 0 and df > 0) or (df == dr and df > 0)
        elif piece_type == 'knight':
            return (df == 2 and dr == 1) or (df == 1 and dr == 2)
        return False

    def _get_non_attacking_square(self, critical_squares: List[str],
                                  forbidden: Set[str], piece_type: str,
                                  piece_color: str) -> str:
        """
        Get a square where a piece does NOT attack any critical squares

        Returns:
            Valid square or None
        """
        for _ in range(100):
            sq = self._random_square()
            if sq in forbidden:
                continue

            attacks_critical = False
            for critical_sq in critical_squares:
                if self._can_attack(sq, critical_sq, piece_type):
                    attacks_critical = True
                    break

            if not attacks_critical:
                return sq

        return None

    def _generate_valid_case(self, case_num: int) -> Dict:
        """Generate a valid castling case"""
        castling_type = random.choice(list(self.castling_configs.keys()))
        config = self.castling_configs[castling_type]
        check_combo = random.choice(self.check_combinations)

        # Initialize piece counts with castling pieces
        piece_counts = {}
        self._add_piece_to_counts('king', config['color'], piece_counts)
        self._add_piece_to_counts('rook', config['color'], piece_counts)

        critical_squares = [config['king_start'],
                            config['through_sq'], config['king_end']]

        occupied = {config['king_start'], config['rook_start']}
        path_squares = set(config['path_squares'])
        forbidden = occupied | path_squares

        extra_pieces = {}
        for i in range(2):
            # Try to find a valid piece type and color
            for _ in range(50):
                piece_type = random.choice(['knight', 'bishop'])
                piece_color = random.choice(['white', 'black'])

                if not self._can_add_piece(piece_type, piece_color, piece_counts):
                    continue

                sq = self._get_non_attacking_square(critical_squares, forbidden,
                                                    piece_type, piece_color)
                if sq:
                    symbol = self._get_attacker_symbol(piece_type, piece_color)
                    extra_pieces[sq] = symbol
                    forbidden.add(sq)
                    self._add_piece_to_counts(
                        piece_type, piece_color, piece_counts)
                    break

        state1_pieces = {
            config['king_start']: config['king_symbol'],
            config['rook_start']: config['rook_symbol'],
            **extra_pieces
        }

        state2_pieces = {
            config['king_end']: config['king_symbol'],
            config['rook_end']: config['rook_symbol'],
            **extra_pieces
        }

        return {
            "case_id": f"L5_valid_{case_num}",
            "type": "castling_with_constraints",
            "subtype": "valid",
            "castling_type": castling_type,
            "check_rules_tested": check_combo,
            "states": [
                {"pieces": state1_pieces, "squares": []},
                {"pieces": state2_pieces, "squares": []}
            ],
            "question": f"Is this castling move legal?",
            "expected": "yes",
            "reasoning": "All castling conditions met, no check violations"
        }

    def _generate_invalid_case(self, case_num: int, check_combo: List[str],
                               violation_type: str) -> Dict:
        """
        Generate an invalid castling case
        """
        castling_type = random.choice(list(self.castling_configs.keys()))
        config = self.castling_configs[castling_type]

        # Initialize piece counts
        piece_counts = {}
        self._add_piece_to_counts('king', config['color'], piece_counts)
        self._add_piece_to_counts('rook', config['color'], piece_counts)

        occupied = {config['king_start'], config['rook_start']}
        path_squares = set(config['path_squares'])
        forbidden = occupied | path_squares

        # Determine attack targets
        attack_targets = []
        violation_details = []

        if violation_type == 'first' or violation_type == 'both':
            rule = check_combo[0]
            if rule == 'in':
                attack_targets.append(config['king_start'])
                violation_details.append('in_check')
            elif rule == 'through':
                attack_targets.append(config['through_sq'])
                violation_details.append('through_check')
            elif rule == 'into':
                attack_targets.append(config['king_end'])
                violation_details.append('into_check')

        if violation_type == 'second' or violation_type == 'both':
            rule = check_combo[1]
            if rule == 'in':
                attack_targets.append(config['king_start'])
                violation_details.append('in_check')
            elif rule == 'through':
                attack_targets.append(config['through_sq'])
                violation_details.append('through_check')
            elif rule == 'into':
                attack_targets.append(config['king_end'])
                violation_details.append('into_check')

        attack_targets = list(set(attack_targets))
        violation_details = list(set(violation_details))

        # Attacker must be opposite color
        attacker_color = 'black' if config['color'] == 'white' else 'white'

        extra_pieces = {}

        # Place attackers
        for target in attack_targets:
            for _ in range(50):
                attacker_type = random.choice(
                    ['rook', 'bishop', 'knight', 'queen'])

                if not self._can_add_piece(attacker_type, attacker_color, piece_counts):
                    continue

                positions = self._get_attacker_positions(
                    target, attacker_type, forbidden)
                if positions:
                    pos = random.choice(positions)
                    symbol = self._get_attacker_symbol(
                        attacker_type, attacker_color)
                    extra_pieces[pos] = symbol
                    forbidden.add(pos)
                    self._add_piece_to_counts(
                        attacker_type, attacker_color, piece_counts)
                    break

        # Fill remaining slots with non-attacking pieces
        while len(extra_pieces) < 2:
            for _ in range(50):
                piece_type = random.choice(['knight', 'bishop'])
                piece_color = random.choice(['white', 'black'])

                if not self._can_add_piece(piece_type, piece_color, piece_counts):
                    continue

                all_critical = [config['king_start'],
                                config['through_sq'], config['king_end']]
                non_target_critical = [
                    sq for sq in all_critical if sq not in attack_targets]

                sq = self._get_non_attacking_square(non_target_critical, forbidden,
                                                    piece_type, piece_color)
                if sq:
                    symbol = self._get_attacker_symbol(piece_type, piece_color)
                    extra_pieces[sq] = symbol
                    forbidden.add(sq)
                    self._add_piece_to_counts(
                        piece_type, piece_color, piece_counts)
                    break
            else:
                break  # Couldn't find a valid piece after many attempts

        state1_pieces = {
            config['king_start']: config['king_symbol'],
            config['rook_start']: config['rook_symbol'],
            **extra_pieces
        }

        state2_pieces = {
            config['king_end']: config['king_symbol'],
            config['rook_end']: config['rook_symbol'],
            **extra_pieces
        }

        reasoning = f"Violates: {', '.join(violation_details)}"

        return {
            "case_id": f"L5_invalid_{case_num}",
            "type": "castling_with_constraints",
            "subtype": "invalid",
            "castling_type": castling_type,
            "check_rules_tested": check_combo,
            "violation_details": violation_details,
            "states": [
                {"pieces": state1_pieces, "squares": []},
                {"pieces": state2_pieces, "squares": []}
            ],
            "question": f"Is this castling move legal?",
            "expected": "no",
            "reasoning": reasoning
        }

    def generate_all(self, n_cases: int = 100) -> List[Dict]:
        """
        Generate all Level 5 test cases
        """
        all_cases = []

        n_valid = int(n_cases * 0.20)
        n_invalid = n_cases - n_valid

        print(f"Generating valid castling cases...")
        for i in range(n_valid):
            case = self._generate_valid_case(i + 1)
            if case:
                all_cases.append(case)
        print(f"  ✓ Generated {n_valid} valid cases")

        print(f"Generating invalid castling cases...")

        cases_per_combo = n_invalid // 3
        remainder = n_invalid % 3

        invalid_count = 0
        for combo_idx, check_combo in enumerate(self.check_combinations):
            n_combo_cases = cases_per_combo + \
                (1 if combo_idx < remainder else 0)

            n_first = n_combo_cases // 3
            n_second = n_combo_cases // 3
            n_both = n_combo_cases - n_first - n_second

            combo_name = f"[{check_combo[0]}, {check_combo[1]}]"

            for i in range(n_first):
                case = self._generate_invalid_case(
                    invalid_count + 1, check_combo, 'first')
                if case:
                    all_cases.append(case)
                    invalid_count += 1

            for i in range(n_second):
                case = self._generate_invalid_case(
                    invalid_count + 1, check_combo, 'second')
                if case:
                    all_cases.append(case)
                    invalid_count += 1

            for i in range(n_both):
                case = self._generate_invalid_case(
                    invalid_count + 1, check_combo, 'both')
                if case:
                    all_cases.append(case)
                    invalid_count += 1

            print(f"  ✓ Generated cases for combo {combo_name}")

        print(f"\n✓ Total generated: {len(all_cases)} Level 5 test cases")
        print(f"  Valid: {n_valid} ({n_valid/len(all_cases)*100:.1f}%)")
        print(
            f"  Invalid: {invalid_count} ({invalid_count/len(all_cases)*100:.1f}%)")

        return all_cases
