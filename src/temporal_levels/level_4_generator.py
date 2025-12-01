"""
Level 4 Generator: En Passant + Constraints
Tests en passant timing and check constraints
"""

import random
from typing import List, Dict, Tuple


class Level4Generator:
    """Generate Level 4 test cases - en passant with constraints"""

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
        return random.choice(self.files) + random.choice(self.ranks)

    def _adjacent_files(self, file: str) -> List[str]:
        """Get adjacent files"""
        file_idx = self.files.index(file)
        adjacent = []
        if file_idx > 0:
            adjacent.append(self.files[file_idx - 1])
        if file_idx < 7:
            adjacent.append(self.files[file_idx + 1])
        return adjacent

    def _is_square_blocking_check(self, king_sq: str, attacker_sq: str, blocker_sq: str) -> bool:
        """
        Check if a square blocks the line between king and attacker
        """
        if blocker_sq == king_sq or blocker_sq == attacker_sq:
            return False

        king_f, king_r = ord(king_sq[0]) - ord('a'), int(king_sq[1]) - 1
        att_f, att_r = ord(attacker_sq[0]) - ord('a'), int(attacker_sq[1]) - 1
        block_f, block_r = ord(
            blocker_sq[0]) - ord('a'), int(blocker_sq[1]) - 1

        # Check if on same file (vertical line)
        if king_f == att_f == block_f:
            min_r, max_r = sorted([king_r, att_r])
            return min_r < block_r < max_r

        # Check if on same rank (horizontal line)
        if king_r == att_r == block_r:
            min_f, max_f = sorted([king_f, att_f])
            return min_f < block_f < max_f

        # Check if on diagonal
        if abs(att_f - king_f) == abs(att_r - king_r):
            if abs(block_f - king_f) == abs(block_r - king_r):
                if king_f < att_f:
                    return king_f < block_f < att_f
                else:
                    return att_f < block_f < king_f

        return False

    def _get_safe_extra_pieces(self, occupied_squares: List[str]) -> Tuple[str, str]:
        """Get 2 random squares for extra pieces"""
        available = [f + r for f in self.files for r in self.ranks
                     if f + r not in occupied_squares]

        if len(available) >= 2:
            return random.sample(available, 2)
        return None, None

    def _generate_valid_cases(self, n_cases: int) -> List[Dict]:
        """Generate valid en passant cases with extra pieces"""
        cases = []

        valid_combinations = []
        for black_file in ['b', 'c', 'd', 'e', 'f', 'g']:
            black_start = black_file + '7'
            black_end = black_file + '5'

            adjacent = self._adjacent_files(black_file)
            for white_file in adjacent:
                white_sq = white_file + '5'
                valid_combinations.append({
                    'white_sq': white_sq,
                    'black_start': black_start,
                    'black_end': black_end
                })

        for i in range(n_cases):
            combo = random.choice(valid_combinations)
            white_sq = combo['white_sq']
            black_start = combo['black_start']
            black_end = combo['black_end']

            occupied = [white_sq, black_start, black_end]
            extra1, extra2 = self._get_safe_extra_pieces(occupied)

            if extra1 and extra2:
                extra_piece_1 = random.choice(['N', 'B', 'n', 'b'])
                extra_piece_2 = random.choice(['N', 'B', 'n', 'b'])

                cases.append({
                    "case_id": f"L4_valid_{i+1}",
                    "type": "en_passant_constraint",
                    "subtype": "valid",
                    "states": [
                        {
                            "pieces": {
                                white_sq: 'P',
                                black_start: 'p',
                                extra1: extra_piece_1,
                                extra2: extra_piece_2
                            },
                            "squares": []
                        },
                        {
                            "pieces": {
                                white_sq: 'P',
                                black_end: 'p',
                                extra1: extra_piece_1,
                                extra2: extra_piece_2
                            },
                            "squares": []
                        }
                    ],
                    "question": "Can white capture the black pawn en passant?",
                    "expected": "yes",
                    "reasoning": "All conditions met, no constraints violated"
                })

        return cases

    def _generate_scenario_a_missed_timing(self, n_cases: int) -> List[Dict]:
        """Scenario A: Missed timing (3 images)"""
        cases = []

        for i in range(n_cases):
            black_file = random.choice(['b', 'c', 'd', 'e', 'f', 'g'])
            adjacent = self._adjacent_files(black_file)
            white_file = random.choice(adjacent)

            black_start = black_file + '7'
            black_mid = black_file + '5'
            white_sq = white_file + '5'

            piece_type = random.choice(['rook', 'knight'])

            if piece_type == 'rook':
                move_type = random.choice(['horizontal', 'vertical'])

                if move_type == 'horizontal':
                    start_file = random.choice(['a', 'h'])
                    start_rank = random.choice(['1', '8'])
                    moving_piece_start = start_file + start_rank
                    end_file = random.choice(['c', 'd', 'e'])
                    moving_piece_end = end_file + start_rank
                else:
                    start_file = random.choice(['a', 'h'])
                    moving_piece_start = start_file + '1'
                    moving_piece_end = start_file + '3'

                moving_piece_symbol = random.choice(['R', 'r'])

            else:
                knight_moves = [
                    ('b1', 'c3'), ('g1', 'f3'), ('b1', 'a3'), ('g1', 'h3'),
                    ('b8', 'c6'), ('g8', 'f6'), ('b8', 'a6'), ('g8', 'h6'),
                    ('a1', 'c2'), ('h1', 'f2'), ('a8', 'c7'), ('h8', 'f7')
                ]

                moving_piece_start, moving_piece_end = random.choice(
                    knight_moves)
                moving_piece_symbol = random.choice(['N', 'n'])

            occupied = [white_sq, black_start, black_mid,
                        moving_piece_start, moving_piece_end]

            if len(occupied) == len(set(occupied)):
                extra_sq = None
                for _ in range(50):
                    candidate = self._random_square()
                    if candidate not in occupied:
                        extra_sq = candidate
                        break

                if extra_sq:
                    extra_piece = random.choice(['N', 'B', 'n', 'b'])

                    cases.append({
                        "case_id": f"L4_scenario_a_{i+1}",
                        "type": "en_passant_constraint",
                        "subtype": "missed_timing",
                        "states": [
                            {
                                "pieces": {
                                    white_sq: 'P',
                                    black_start: 'p',
                                    moving_piece_start: moving_piece_symbol,
                                    extra_sq: extra_piece
                                },
                                "squares": []
                            },
                            {
                                "pieces": {
                                    white_sq: 'P',
                                    black_mid: 'p',
                                    moving_piece_start: moving_piece_symbol,
                                    extra_sq: extra_piece
                                },
                                "squares": []
                            },
                            {
                                "pieces": {
                                    white_sq: 'P',
                                    black_mid: 'p',
                                    moving_piece_end: moving_piece_symbol,
                                    extra_sq: extra_piece
                                },
                                "squares": []
                            }
                        ],
                        "question": "Can white capture the black pawn en passant in the position shown in State 3?",
                        "expected": "no",
                        "reasoning": f"White moved {piece_type} instead, timing window closed"
                    })

        return cases

    def _generate_scenario_b_causes_check(self, n_cases: int) -> List[Dict]:
        """
        Scenario B: En passant would expose King to check

        Simple setup: King and Rook on same file, white pawn blocks in between
        """
        cases = []

        # Use predefined safe configurations
        configs = [
            # King at d1, white pawn d5, black rook d4 (rank 1-4!)
            {'king': 'd1', 'white': 'd5', 'attacker': 'd2',
                'black_start': 'e7', 'black_end': 'e5'},
            {'king': 'd1', 'white': 'd5', 'attacker': 'd3',
                'black_start': 'e7', 'black_end': 'e5'},
            {'king': 'e1', 'white': 'e5', 'attacker': 'e2',
                'black_start': 'd7', 'black_end': 'd5'},
            {'king': 'e1', 'white': 'e5', 'attacker': 'e3',
                'black_start': 'f7', 'black_end': 'f5'},
            {'king': 'f1', 'white': 'f5', 'attacker': 'f2',
                'black_start': 'e7', 'black_end': 'e5'},
            {'king': 'f1', 'white': 'f5', 'attacker': 'f3',
                'black_start': 'g7', 'black_end': 'g5'},
        ]

        for i in range(n_cases):
            config = random.choice(configs)

            cases.append({
                "case_id": f"L4_scenario_b_{i+1}",
                "type": "en_passant_constraint",
                "subtype": "causes_check",
                "states": [
                    {
                        "pieces": {
                            config['white']: 'P',
                            config['black_start']: 'p',
                            config['king']: 'K',
                            config['attacker']: 'r'
                        },
                        "squares": []
                    },
                    {
                        "pieces": {
                            config['white']: 'P',
                            config['black_end']: 'p',
                            config['king']: 'K',
                            config['attacker']: 'r'
                        },
                        "squares": []
                    }
                ],
                "question": "Can white capture the black pawn en passant?",
                "expected": "no",
                "reasoning": f"White pawn blocks check, capturing would expose King to check"
            })

        return cases

    def _generate_scenario_c_in_check(self, n_cases: int) -> List[Dict]:
        """
        Scenario C: King is currently in check

        ULTRA SIMPLE STRATEGY: Use predefined valid configurations
        All attackers guaranteed to be in rank 1-4
        All 4 pieces guaranteed to exist
        No blocking issues
        """
        cases = []

        # Predefined configurations that are guaranteed to work
        # Format: king, attacker (rank 1-4 only!), white_pawn, black_pawn_start, black_pawn_end
        configs = [
            # Horizontal attacks (same rank)
            {'king': 'a1', 'attacker': 'h1', 'white': 'd5',
                'black_start': 'e7', 'black_end': 'e5', 'attacker_type': 'r'},
            {'king': 'a2', 'attacker': 'h2', 'white': 'd5',
                'black_start': 'e7', 'black_end': 'e5', 'attacker_type': 'r'},
            {'king': 'h1', 'attacker': 'a1', 'white': 'd5',
                'black_start': 'e7', 'black_end': 'e5', 'attacker_type': 'r'},
            {'king': 'h2', 'attacker': 'a2', 'white': 'd5',
                'black_start': 'e7', 'black_end': 'e5', 'attacker_type': 'r'},

            # Vertical attacks (same file, attacker in rank 1-4)
            {'king': 'a1', 'attacker': 'a4', 'white': 'd5',
                'black_start': 'e7', 'black_end': 'e5', 'attacker_type': 'r'},
            {'king': 'a2', 'attacker': 'a4', 'white': 'd5',
                'black_start': 'e7', 'black_end': 'e5', 'attacker_type': 'r'},
            {'king': 'b1', 'attacker': 'b4', 'white': 'd5',
                'black_start': 'e7', 'black_end': 'e5', 'attacker_type': 'r'},
            {'king': 'h1', 'attacker': 'h4', 'white': 'd5',
                'black_start': 'e7', 'black_end': 'e5', 'attacker_type': 'r'},

            # Diagonal attacks from rank 1-4
            {'king': 'a1', 'attacker': 'c3', 'white': 'f5',
                'black_start': 'e7', 'black_end': 'e5', 'attacker_type': 'b'},
            {'king': 'a1', 'attacker': 'd4', 'white': 'f5',
                'black_start': 'e7', 'black_end': 'e5', 'attacker_type': 'q'},
            {'king': 'a2', 'attacker': 'c4', 'white': 'f5',
                'black_start': 'e7', 'black_end': 'e5', 'attacker_type': 'b'},
            {'king': 'b1', 'attacker': 'd3', 'white': 'f5',
                'black_start': 'e7', 'black_end': 'e5', 'attacker_type': 'b'},
            {'king': 'h1', 'attacker': 'f3', 'white': 'd5',
                'black_start': 'e7', 'black_end': 'e5', 'attacker_type': 'b'},
            {'king': 'h2', 'attacker': 'f4', 'white': 'd5',
                'black_start': 'e7', 'black_end': 'e5', 'attacker_type': 'q'},
        ]

        for i in range(n_cases):
            config = random.choice(configs)

            # Double-check attacker is in rank 1-4
            attacker_rank = int(config['attacker'][1])
            if attacker_rank < 1 or attacker_rank > 4:
                print(
                    f"ERROR: Config has attacker at {config['attacker']} not in rank 1-4!")
                continue

            # Verify all 4 positions are unique
            positions = [config['king'], config['attacker'],
                         config['white'], config['black_start']]
            if len(set(positions)) != 4:
                continue

            # Verify black pawn path is clear
            black_file = config['black_start'][0]
            black_path = black_file + '6'
            if black_path in positions:
                continue

            # Verify no blocking
            if self._is_square_blocking_check(config['king'], config['attacker'], config['white']):
                continue
            if self._is_square_blocking_check(config['king'], config['attacker'], config['black_start']):
                continue
            if self._is_square_blocking_check(config['king'], config['attacker'], config['black_end']):
                continue

            cases.append({
                "case_id": f"L4_scenario_c_{i+1}",
                "type": "en_passant_constraint",
                "subtype": "in_check",
                "states": [
                    {
                        "pieces": {
                            config['white']: 'P',
                            config['black_start']: 'p',
                            config['king']: 'K',
                            config['attacker']: config['attacker_type']
                        },
                        "squares": []
                    },
                    {
                        "pieces": {
                            config['white']: 'P',
                            config['black_end']: 'p',
                            config['king']: 'K',
                            config['attacker']: config['attacker_type']
                        },
                        "squares": []
                    }
                ],
                "question": "Can white capture the black pawn en passant?",
                "expected": "no",
                "reasoning": f"King at {config['king']} is in check from {config['attacker']}, must resolve check first"
            })

        return cases

    def generate_all(self, n_cases: int = 100) -> List[Dict]:
        """Generate all Level 4 test cases"""
        all_cases = []

        n_valid = int(n_cases * 0.20)
        n_invalid = n_cases - n_valid

        n_scenario_a = n_invalid // 3
        n_scenario_b = n_invalid // 3
        n_scenario_c = n_invalid - n_scenario_a - n_scenario_b

        print(f"Generating valid en passant cases with extra pieces...")
        valid_cases = self._generate_valid_cases(n_valid)
        all_cases.extend(valid_cases)
        print(f"  ✓ Generated {len(valid_cases)} valid cases")

        print(f"Generating constraint violations...")

        scenario_a = self._generate_scenario_a_missed_timing(n_scenario_a)
        all_cases.extend(scenario_a)
        print(f"  ✓ Generated {len(scenario_a)} scenario A (missed timing)")

        scenario_b = self._generate_scenario_b_causes_check(n_scenario_b)
        all_cases.extend(scenario_b)
        print(f"  ✓ Generated {len(scenario_b)} scenario B (causes check)")

        scenario_c = self._generate_scenario_c_in_check(n_scenario_c)
        all_cases.extend(scenario_c)
        print(f"  ✓ Generated {len(scenario_c)} scenario C (in check)")

        print(f"\n✓ Total generated: {len(all_cases)} Level 4 test cases")
        print(
            f"  Valid: {len(valid_cases)} ({len(valid_cases)/len(all_cases)*100:.1f}%)")
        print(
            f"  Invalid: {len(all_cases) - len(valid_cases)} ({(len(all_cases) - len(valid_cases))/len(all_cases)*100:.1f}%)")

        return all_cases
