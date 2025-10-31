"""
Automated test case generator for Spatial Test 0
"""

import random
from typing import List, Dict, Tuple


class SpatialTest0Generator:
    """Automatically generate spatial reasoning test cases"""

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
        return chr(ord('a') + file) + str(rank + 1)

    def _random_square(self) -> str:
        """Generate random square"""
        file = random.choice(self.files)
        rank = random.choice(self.ranks)
        return file + rank

    def _random_square_pair(self) -> Tuple[str, str]:
        """Generate two different random squares"""
        sq1 = self._random_square()
        sq2 = self._random_square()
        while sq1 == sq2:
            sq2 = self._random_square()
        return sq1, sq2

    # ============= Type 1: Same Line =============

    def generate_same_file_tests(self, n_positive: int = 5, n_negative: int = 5) -> List[Dict]:
        """
        Generate same file (vertical line) tests

        Args:
            n_positive: Number of positive cases (same file)
            n_negative: Number of negative cases (different files)
        """
        cases = []

        # Positive cases: same file
        for i in range(n_positive):
            file = random.choice(self.files)
            rank1, rank2 = random.sample(self.ranks, 2)
            sq1, sq2 = file + rank1, file + rank2

            cases.append({
                "case_id": f"same_file_pos_{i+1}",
                "type": "same_line",
                "subtype": "same_file",
                "squares": [sq1, sq2],
                "question": f"Are the highlighted squares on the same file (vertical line)?",
                "expected": "yes",
                "reasoning": f"Both on file {file}"
            })

        # Negative cases: different files
        for i in range(n_negative):
            sq1, sq2 = self._random_square_pair()
            # Ensure different files
            while sq1[0] == sq2[0]:
                sq2 = self._random_square()

            cases.append({
                "case_id": f"same_file_neg_{i+1}",
                "type": "same_line",
                "subtype": "same_file",
                "squares": [sq1, sq2],
                "question": f"Are the highlighted squares on the same file (vertical line)?",
                "expected": "no",
                "reasoning": f"Different files: {sq1[0]} vs {sq2[0]}"
            })

        return cases

    def generate_same_rank_tests(self, n_positive: int = 5, n_negative: int = 5) -> List[Dict]:
        """
        Generate same rank (horizontal line) tests

        Args:
            n_positive: Number of positive cases (same rank)
            n_negative: Number of negative cases (different ranks)
        """
        cases = []

        # Positive cases
        for i in range(n_positive):
            rank = random.choice(self.ranks)
            file1, file2 = random.sample(self.files, 2)
            sq1, sq2 = file1 + rank, file2 + rank

            cases.append({
                "case_id": f"same_rank_pos_{i+1}",
                "type": "same_line",
                "subtype": "same_rank",
                "squares": [sq1, sq2],
                "question": f"Are the highlighted squares on the same rank (horizontal line)?",
                "expected": "yes",
                "reasoning": f"Both on rank {rank}"
            })

        # Negative cases
        for i in range(n_negative):
            sq1, sq2 = self._random_square_pair()
            while sq1[1] == sq2[1]:  # Ensure different ranks
                sq2 = self._random_square()

            cases.append({
                "case_id": f"same_rank_neg_{i+1}",
                "type": "same_line",
                "subtype": "same_rank",
                "squares": [sq1, sq2],
                "question": f"Are the highlighted squares on the same rank (horizontal line)?",
                "expected": "no",
                "reasoning": f"Different ranks: {sq1[1]} vs {sq2[1]}"
            })

        return cases

    # ============= Type 2: Diagonal =============

    def _on_same_diagonal(self, sq1: str, sq2: str) -> bool:
        """Check if two squares are on the same diagonal"""
        f1, r1 = self._square_to_coords(sq1)
        f2, r2 = self._square_to_coords(sq2)

        # Same diagonal if |delta_file| == |delta_rank|
        return abs(f1 - f2) == abs(r1 - r2) and sq1 != sq2

    def generate_diagonal_tests(self, n_positive: int = 5, n_negative: int = 5) -> List[Dict]:
        """
        Generate diagonal tests

        Args:
            n_positive: Number of positive cases (on same diagonal)
            n_negative: Number of negative cases (not on same diagonal)
        """
        cases = []

        # Positive cases: on same diagonal
        attempts = 0
        while len(cases) < n_positive and attempts < 1000:
            sq1, sq2 = self._random_square_pair()
            if self._on_same_diagonal(sq1, sq2):
                cases.append({
                    "case_id": f"diagonal_pos_{len(cases)+1}",
                    "type": "diagonal",
                    "squares": [sq1, sq2],
                    "question": f"Are the highlighted squares on the same diagonal?",
                    "expected": "yes",
                    "reasoning": "On same diagonal"
                })
            attempts += 1

        # Negative cases: not on same diagonal
        attempts = 0
        neg_count = 0
        while neg_count < n_negative and attempts < 1000:
            sq1, sq2 = self._random_square_pair()
            if not self._on_same_diagonal(sq1, sq2):
                cases.append({
                    "case_id": f"diagonal_neg_{neg_count+1}",
                    "type": "diagonal",
                    "squares": [sq1, sq2],
                    "question": f"Are the highlighted squares on the same diagonal?",
                    "expected": "no",
                    "reasoning": "Not on same diagonal"
                })
                neg_count += 1
            attempts += 1

        return cases

    # ============= Type 3: Relative Position =============

    def _get_direction(self, from_sq: str, to_sq: str) -> str:
        """Get direction from one square to another"""
        f1, r1 = self._square_to_coords(from_sq)
        f2, r2 = self._square_to_coords(to_sq)

        df = f2 - f1  # positive = east, negative = west
        dr = r2 - r1  # positive = north, negative = south

        if df == 0 and dr > 0:
            return "north"
        elif df > 0 and dr > 0:
            return "northeast"
        elif df > 0 and dr == 0:
            return "east"
        elif df > 0 and dr < 0:
            return "southeast"
        elif df == 0 and dr < 0:
            return "south"
        elif df < 0 and dr < 0:
            return "southwest"
        elif df < 0 and dr == 0:
            return "west"
        elif df < 0 and dr > 0:
            return "northwest"
        else:
            return "same"

    def _has_component_of(self, actual_dir: str, target_dir: str) -> bool:
        """
        Check if actual direction contains a component of target direction

        Examples:
            - target: "south", actual: "south" → True
            - target: "south", actual: "southwest" → True (has south component)
            - target: "south", actual: "north" → False (no south component)
            - target: "northeast", actual: "northeast" → True
            - target: "northeast", actual: "north" → False (northeast is composite)
        """
        # Define which directions contain each component
        north_dirs = ["north", "northeast", "northwest"]
        south_dirs = ["south", "southeast", "southwest"]
        east_dirs = ["east", "northeast", "southeast"]
        west_dirs = ["west", "northwest", "southwest"]

        direction_map = {
            "north": north_dirs,
            "south": south_dirs,
            "east": east_dirs,
            "west": west_dirs,
            # Composite directions only match themselves
            "northeast": ["northeast"],
            "southeast": ["southeast"],
            "southwest": ["southwest"],
            "northwest": ["northwest"],
        }

        # Check if actual_dir is in the target direction's component list
        return actual_dir in direction_map.get(target_dir, [])

    def generate_direction_tests(self, n_per_direction: int = 2) -> List[Dict]:
        """
        Generate relative position tests

        Args:
            n_per_direction: Number of cases per direction (both positive and negative)
        """
        cases = []
        directions = ["north", "northeast", "east", "southeast",
                      "south", "southwest", "west", "northwest"]

        for direction in directions:
            # Positive cases - must be exact match
            pos_count = 0
            attempts = 0
            while pos_count < n_per_direction and attempts < 200:
                sq1, sq2 = self._random_square_pair()
                actual_dir = self._get_direction(sq1, sq2)

                if actual_dir == direction:
                    cases.append({
                        "case_id": f"dir_{direction}_pos_{pos_count+1}",
                        "type": "relative_position",
                        "squares": [sq1, sq2],
                        "question": f"Is {sq2} {direction} of {sq1}?",
                        "expected": "yes",
                        "reasoning": f"{sq2} is indeed {direction} of {sq1}"
                    })
                    pos_count += 1
                attempts += 1

            # Negative cases - must NOT have any component of target direction
            neg_count = 0
            attempts = 0
            while neg_count < n_per_direction and attempts < 200:
                sq1, sq2 = self._random_square_pair()
                actual_dir = self._get_direction(sq1, sq2)

                # Use the new method: actual_dir must NOT contain target direction component
                if actual_dir != "same" and not self._has_component_of(actual_dir, direction):
                    cases.append({
                        "case_id": f"dir_{direction}_neg_{neg_count+1}",
                        "type": "relative_position",
                        "squares": [sq1, sq2],
                        "question": f"Is {sq2} {direction} of {sq1}?",
                        "expected": "no",
                        "reasoning": f"{sq2} is {actual_dir} of {sq1}, no {direction} component"
                    })
                    neg_count += 1
                attempts += 1

        return cases

    # ============= Type 4: Distance =============

    def _distance(self, sq1: str, sq2: str) -> float:
        """Calculate Euclidean distance between squares"""
        f1, r1 = self._square_to_coords(sq1)
        f2, r2 = self._square_to_coords(sq2)
        return ((f2-f1)**2 + (r2-r1)**2) ** 0.5

    def generate_distance_tests(self, n_positive: int = 5, n_negative: int = 5) -> List[Dict]:
        """
        Generate distance comparison tests

        Args:
            n_positive: Number of positive cases (closer)
            n_negative: Number of negative cases (not closer)
        """
        cases = []

        # Generate random triples and check distances
        attempts = 0
        pos_count = 0
        neg_count = 0

        while (pos_count < n_positive or neg_count < n_negative) and attempts < 2000:
            # Pick 3 different squares
            squares = set()
            while len(squares) < 3:
                squares.add(self._random_square())
            sq1, sq2, sq3 = list(squares)

            d12 = self._distance(sq1, sq2)
            d13 = self._distance(sq1, sq3)

            is_closer = d12 < d13

            if pos_count < n_positive and is_closer:
                cases.append({
                    "case_id": f"distance_pos_{pos_count+1}",
                    "type": "distance",
                    "squares": [sq1, sq2, sq3],
                    "question": f"Is {sq2} closer to {sq1} than {sq3} is to {sq1}?",
                    "expected": "yes",
                    "reasoning": f"Distance {sq1}-{sq2}: {d12:.2f}, {sq1}-{sq3}: {d13:.2f}"
                })
                pos_count += 1
            elif neg_count < n_negative and not is_closer:
                cases.append({
                    "case_id": f"distance_neg_{neg_count+1}",
                    "type": "distance",
                    "squares": [sq1, sq2, sq3],
                    "question": f"Is {sq2} closer to {sq1} than {sq3} is to {sq1}?",
                    "expected": "no",
                    "reasoning": f"Distance {sq1}-{sq2}: {d12:.2f}, {sq1}-{sq3}: {d13:.2f}"
                })
                neg_count += 1

            attempts += 1

        return cases

    # ============= Type 5: Path Clear =============

    def generate_path_clear_tests(self, n_positive: int = 5, n_negative: int = 5) -> List[Dict]:
        """
        Generate path clearance tests

        Args:
            n_positive: Number of positive cases (clear path)
            n_negative: Number of negative cases (blocked path)
        """
        cases = []

        # Positive cases: clear path
        for i in range(n_positive):
            # Pick a file or rank
            if random.choice([True, False]):  # File (vertical)
                file = random.choice(self.files)
                ranks = sorted(random.sample([int(r) for r in self.ranks], 2))
                rank1, rank2 = ranks[0], ranks[1]

                # Place pieces outside the path
                pieces = {}
                for _ in range(random.randint(1, 3)):
                    piece_file = random.choice(
                        [f for f in self.files if f != file])
                    piece_rank = str(random.randint(1, 8))
                    pieces[piece_file + piece_rank] = "P"

                sq1 = file + str(rank1)
                sq2 = file + str(rank2)

            else:  # Rank (horizontal)
                rank = random.choice(self.ranks)
                files = sorted(random.sample(self.files, 2))

                pieces = {}
                for _ in range(random.randint(1, 3)):
                    piece_file = random.choice(self.files)
                    piece_rank = random.choice(
                        [r for r in self.ranks if r != rank])
                    pieces[piece_file + piece_rank] = "P"

                sq1 = files[0] + rank
                sq2 = files[1] + rank

            cases.append({
                "case_id": f"path_clear_pos_{i+1}",
                "type": "path_clear",
                "pieces": pieces,
                "squares": [sq1, sq2],
                "question": f"Is the path from {sq1} to {sq2} clear (no pieces blocking)?",
                "expected": "yes",
                "reasoning": "No pieces block the path"
            })

        # Negative cases: blocked path
        for i in range(n_negative):
            if random.choice([True, False]):  # File (vertical)
                file = random.choice(self.files)
                ranks = sorted(random.sample([int(r) for r in self.ranks], 2))
                rank1, rank2 = ranks[0], ranks[1]

                # Place blocking piece
                if rank2 - rank1 > 1:
                    blocking_rank = random.randint(rank1 + 1, rank2 - 1)
                else:
                    # If adjacent, place on one of them
                    blocking_rank = rank1

                pieces = {file + str(blocking_rank): "P"}

                sq1 = file + str(rank1)
                sq2 = file + str(rank2)

            else:  # Rank (horizontal)
                rank = random.choice(self.ranks)
                files = sorted(random.sample(self.files, 2))
                file_idx1 = self.files.index(files[0])
                file_idx2 = self.files.index(files[1])

                if file_idx2 - file_idx1 > 1:
                    blocking_idx = random.randint(file_idx1 + 1, file_idx2 - 1)
                else:
                    blocking_idx = file_idx1

                blocking_file = self.files[blocking_idx]
                pieces = {blocking_file + rank: "P"}

                sq1 = files[0] + rank
                sq2 = files[1] + rank

            cases.append({
                "case_id": f"path_clear_neg_{i+1}",
                "type": "path_clear",
                "pieces": pieces,
                "squares": [sq1, sq2],
                "question": f"Is the path from {sq1} to {sq2} clear (no pieces blocking)?",
                "expected": "no",
                "reasoning": f"Path blocked by piece at {list(pieces.keys())[0]}"
            })

        return cases

    # ============= Main Generation Method =============

    def generate_all(self, n_per_type: int = 10) -> List[Dict]:
        """
        Generate comprehensive test suite

        Args:
            n_per_type: Number of cases per test type (will be split into positive/negative)

        Returns:
            List of test case dictionaries
        """
        all_cases = []

        n_pos = n_per_type // 2
        n_neg = n_per_type - n_pos

        print(f"Generating same file tests...")
        all_cases.extend(self.generate_same_file_tests(n_pos, n_neg))

        print(f"Generating same rank tests...")
        all_cases.extend(self.generate_same_rank_tests(n_pos, n_neg))

        print(f"Generating diagonal tests...")
        all_cases.extend(self.generate_diagonal_tests(n_pos, n_neg))

        print(f"Generating direction tests...")
        # For directions, use fewer per direction
        all_cases.extend(self.generate_direction_tests(
            n_per_direction=max(1, n_per_type // 8)))

        print(f"Generating distance tests...")
        all_cases.extend(self.generate_distance_tests(n_pos, n_neg))

        print(f"Generating path clear tests...")
        all_cases.extend(self.generate_path_clear_tests(n_pos, n_neg))

        print(f"\n✔ Total generated: {len(all_cases)} test cases")

        return all_cases
