import numpy as np
from PIL import Image, ImageDraw, ImageFont
import random
import json
from pathlib import Path


class TicTacToeResolutionTestGenerator:
    """Generate Tic-Tac-Toe board images with varying resolutions to test preprocessing artifacts."""

    def __init__(self, output_dir="tictactoe_resolution_tests", patch_size=16):
        self.output_dir = Path(output_dir)
        self.patch_size = patch_size
        self.board_size = 3  # 3x3 Tic-Tac-Toe

        # Core parameter: board to image ratio
        self.board_to_image_ratio = 0.7  # Slightly smaller since it's only 3×3

        # Resolution configurations (same as Gomoku)
        self.resolution_groups = {
            "divisible": {
                "resolutions": [1024, 512, 384],
                "description": "Image sizes divisible by patch size (16)",
            },
            "non_divisible": {
                "resolutions": [1010, 510, 370],
                "description": "Image sizes NOT divisible by 16 (require padding)",
            },
        }

        # Create output directories
        for group in ["divisible", "non_divisible"]:
            for res in self.resolution_groups[group]["resolutions"]:
                (self.output_dir / group / f"{res}x{res}").mkdir(
                    parents=True, exist_ok=True
                )

        (self.output_dir / "debug").mkdir(parents=True, exist_ok=True)

        # Colors
        self.colors = {
            "background": (245, 245, 240),
            "board": (220, 179, 92),
            "line": (0, 0, 0),
            "X": (220, 20, 60),  # Red X
            "O": (30, 144, 255),  # Blue O
            "coordinate": (50, 50, 50),
        }

    def _calculate_dimensions(self, image_size: int) -> dict:
        """Calculate all dimensions for a given image size."""
        board_size_px = int(image_size * self.board_to_image_ratio)
        border = (image_size - board_size_px) // 2
        cell_size = board_size_px / 3  # 3×3 board

        # Calculate token grid info
        is_divisible = (image_size % self.patch_size) == 0
        if is_divisible:
            token_grid = image_size // self.patch_size
            padded_to = image_size
        else:
            padded_to = (
                (image_size + self.patch_size - 1) // self.patch_size
            ) * self.patch_size
            token_grid = padded_to // self.patch_size

        return {
            "image_size": image_size,
            "board_size_px": board_size_px,
            "border": border,
            "cell_size": cell_size,
            "is_divisible": is_divisible,
            "padded_to": padded_to,
            "token_grid": f"{token_grid}×{token_grid}",
            "board_to_image_ratio": self.board_to_image_ratio,
        }

    def generate_random_board(self, density="high") -> np.ndarray:
        """Generate random Tic-Tac-Toe board state.

        Args:
            density: "high" generates dense boards (7-9 pieces)
        """
        board = np.zeros((3, 3), dtype=int)

        # For 3×3, we want dense boards to avoid empty-cell bias
        density_ranges = {
            "medium": (4, 6),
            "high": (7, 9),
        }
        min_pieces, max_pieces = density_ranges.get(density, (7, 9))
        num_pieces = random.randint(min_pieces, max_pieces)

        all_positions = [(i, j) for i in range(3) for j in range(3)]
        selected_positions = random.sample(all_positions, num_pieces)

        for idx, (row, col) in enumerate(selected_positions):
            board[row, col] = 1 if idx % 2 == 0 else 2  # 1=X, 2=O

        return board

    def render_board_image(
        self, board: np.ndarray, dimensions: dict, show_debug_info: bool = False
    ) -> Image.Image:
        """Render Tic-Tac-Toe board at specified resolution.

        Args:
            board: 3x3 numpy array (0=empty, 1=X, 2=O)
            dimensions: Dict from _calculate_dimensions()
            show_debug_info: If True, overlay resolution info for debugging
        """
        image_size = dimensions["image_size"]
        board_size_px = dimensions["board_size_px"]
        border = dimensions["border"]
        cell_size = dimensions["cell_size"]

        # Create canvas
        img = Image.new("RGB", (image_size, image_size), self.colors["background"])
        draw = ImageDraw.Draw(img)

        # Board background
        board_margin = max(10, int(cell_size * 0.1))
        draw.rectangle(
            [
                border - board_margin,
                border - board_margin,
                border + board_size_px + board_margin,
                border + board_size_px + board_margin,
            ],
            fill=self.colors["board"],
        )

        # Grid lines (scale with resolution)
        line_width = max(2, int(image_size / 256))
        for i in range(4):
            pos = border + i * cell_size

            # Horizontal
            draw.line(
                [(border, pos), (border + board_size_px, pos)],
                fill=self.colors["line"],
                width=line_width,
            )
            # Vertical
            draw.line(
                [(pos, border), (pos, border + board_size_px)],
                fill=self.colors["line"],
                width=line_width,
            )

        # Coordinate labels
        font_size = max(12, int(cell_size * 0.15))
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            font = ImageFont.load_default()

        label_offset = max(15, int(cell_size * 0.25))
        for i in range(3):
            # Row labels (0, 1, 2)
            row_label = str(i)
            y_pos = border + (i + 0.5) * cell_size
            draw.text(
                (border - label_offset, y_pos - font_size // 2),
                row_label,
                fill=self.colors["coordinate"],
                font=font,
            )

            # Column labels (0, 1, 2)
            col_label = str(i)
            x_pos = border + (i + 0.5) * cell_size
            bbox = draw.textbbox((0, 0), col_label, font=font)
            text_width = bbox[2] - bbox[0]
            draw.text(
                (x_pos - text_width / 2, border - label_offset),
                col_label,
                fill=self.colors["coordinate"],
                font=font,
            )

        # Draw X and O symbols
        symbol_size = cell_size * 0.5
        symbol_width = max(4, int(image_size / 128))

        for i in range(3):
            for j in range(3):
                if board[i, j] != 0:
                    cx = border + (j + 0.5) * cell_size
                    cy = border + (i + 0.5) * cell_size

                    if board[i, j] == 1:  # X
                        offset = symbol_size / 2
                        draw.line(
                            [(cx - offset, cy - offset), (cx + offset, cy + offset)],
                            fill=self.colors["X"],
                            width=symbol_width,
                        )
                        draw.line(
                            [(cx - offset, cy + offset), (cx + offset, cy - offset)],
                            fill=self.colors["X"],
                            width=symbol_width,
                        )
                    else:  # O
                        offset = symbol_size / 2
                        draw.ellipse(
                            [cx - offset, cy - offset, cx + offset, cy + offset],
                            outline=self.colors["O"],
                            width=symbol_width,
                        )

        # Optional debug overlay
        if show_debug_info:
            overlay = Image.new("RGBA", (image_size, image_size), (0, 0, 0, 0))
            overlay_draw = ImageDraw.Draw(overlay)

            info_lines = [
                f"Tic-Tac-Toe 3×3",
                f"Resolution: {image_size}×{image_size}",
                f"Divisible by {self.patch_size}: {dimensions['is_divisible']}",
                f"Padded to: {dimensions['padded_to']}×{dimensions['padded_to']}",
                f"Token grid: {dimensions['token_grid']}",
                f"Board size: {board_size_px}px",
                f"Cell size: {cell_size:.2f}px",
            ]

            try:
                info_font = ImageFont.truetype("arial.ttf", 14)
            except:
                info_font = ImageFont.load_default()

            box_height = len(info_lines) * 20 + 10
            overlay_draw.rectangle(
                [(5, 5), (300, box_height)], fill=(255, 255, 255, 220)
            )

            for idx, line in enumerate(info_lines):
                overlay_draw.text(
                    (10, 10 + idx * 20), line, fill=(0, 0, 0, 255), font=info_font
                )

            img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")

        return img

    def generate_test_suite(self, n_samples_per_resolution: int = 10):
        """Generate complete test suite across all resolutions."""
        print(f"Generating Tic-Tac-Toe Resolution Test Suite")
        print(f"  Board size: 3×3")
        print(f"  Patch size: {self.patch_size}×{self.patch_size}px")
        print(f"  Board to image ratio: {self.board_to_image_ratio}")
        print(f"  Samples per resolution: {n_samples_per_resolution}")
        print()

        # Test metadata
        test_metadata = {
            "patch_size": self.patch_size,
            "board_game_size": self.board_size,
            "board_to_image_ratio": self.board_to_image_ratio,
            "resolution_groups": self.resolution_groups,
            "density_info": "high (7-9 pieces per board)",
            "test_cases": [],
        }

        # Generate HIGH DENSITY board states
        print(f"Generating {n_samples_per_resolution} HIGH DENSITY board states...")
        board_states = []
        for i in range(n_samples_per_resolution):
            board = self.generate_random_board("high")  # 7-9 pieces
            board_states.append(board)

        # Calculate statistics
        piece_counts = [np.sum(b > 0) for b in board_states]
        avg_pieces = np.mean(piece_counts)
        avg_density = avg_pieces / 9  # 3×3 = 9 cells
        empty_baseline = 1 - avg_density

        print(f"  ✓ Generated {len(board_states)} board states")
        print(f"  ✓ Average pieces: {avg_pieces:.1f} ({avg_density:.1%} density)")
        print(
            f"  ✓ Empty baseline: {empty_baseline:.1%} (accuracy if model guesses all empty)"
        )
        print(
            f"  ✓ Models must exceed {empty_baseline:.1%} to show real piece detection\n"
        )

        test_metadata["density_statistics"] = {
            "average_pieces": float(avg_pieces),
            "average_density": float(avg_density),
            "empty_baseline": float(empty_baseline),
        }

        # Generate images for each resolution
        for group_name, group_info in self.resolution_groups.items():
            print(f"Processing {group_name.upper()} group:")
            print(f"  Description: {group_info['description']}\n")

            for resolution in group_info["resolutions"]:
                print(f"  Generating {resolution}×{resolution} images...")

                dimensions = self._calculate_dimensions(resolution)

                for sample_idx, board in enumerate(board_states):
                    # Render image
                    img = self.render_board_image(
                        board, dimensions, show_debug_info=False
                    )

                    # Save image
                    filename = (
                        f"tictactoe_{group_name}_{resolution}_{sample_idx:03d}.png"
                    )
                    filepath = (
                        self.output_dir
                        / group_name
                        / f"{resolution}x{resolution}"
                        / filename
                    )
                    img.save(filepath)

                    # Create test case metadata
                    test_case = {
                        "test_id": f"tictactoe_resolution_{group_name}_{resolution}_{sample_idx:03d}",
                        "group": group_name,
                        "resolution": resolution,
                        "sample_index": sample_idx,
                        "image_file": str(filepath),
                        "dimensions": dimensions,
                        "ground_truth": board.tolist(),
                        "statistics": {
                            "total_pieces": int(np.sum(board > 0)),
                            "X_count": int(np.sum(board == 1)),
                            "O_count": int(np.sum(board == 2)),
                            "density": float(np.sum(board > 0) / 9),
                        },
                    }

                    # Save individual test JSON
                    test_json = filepath.parent / f"test_{sample_idx:03d}.json"
                    with open(test_json, "w") as f:
                        json.dump(test_case, f, indent=2)

                    test_metadata["test_cases"].append(test_case)

                print(f"    ✓ Generated {n_samples_per_resolution} images\n")

        # Save overall metadata
        metadata_file = self.output_dir / "test_metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(test_metadata, f, indent=2)

        print(f"✅ Test suite generated successfully!")
        print(f"📁 Output directory: {self.output_dir}")
        print(f"📊 Total test cases: {len(test_metadata['test_cases'])}")
        print(f"\n⚠️  Note: HIGH DENSITY boards used")
        print(f"   Empty baseline: ~{empty_baseline:.1%}")
        print(f"   Accuracy below this suggests model is guessing empty")

        return test_metadata

    def generate_debug_samples(self):
        """Generate debug samples with resolution info overlay."""
        print("\nGenerating debug samples with resolution info...")

        # Use a dense pattern for visualization
        board = np.array([[1, 2, 1], [2, 1, 2], [2, 1, 1]])

        for group_name, group_info in self.resolution_groups.items():
            for resolution in group_info["resolutions"]:
                dimensions = self._calculate_dimensions(resolution)
                img = self.render_board_image(board, dimensions, show_debug_info=True)

                filepath = (
                    self.output_dir / "debug" / f"debug_{group_name}_{resolution}.png"
                )
                img.save(filepath)
                print(f"  Saved: {filepath.name}")

        print("✅ Debug samples generated!")

    def analyze_resolutions(self):
        """Print detailed analysis of resolution configurations."""
        print("\n" + "=" * 70)
        print("TIC-TAC-TOE RESOLUTION TEST CONFIGURATION")
        print("=" * 70)
        print(f"Board size: 3×3")
        print(f"Patch size: {self.patch_size}×{self.patch_size}px")
        print(f"Board to image ratio: {self.board_to_image_ratio}")
        print()

        for group_name, group_info in self.resolution_groups.items():
            print(f"[{group_name.upper()} GROUP]")
            print(f"Description: {group_info['description']}")
            print()

            for resolution in group_info["resolutions"]:
                dims = self._calculate_dimensions(resolution)

                print(f"  Resolution: {resolution}×{resolution}px")
                print(f"    Divisible by {self.patch_size}: {dims['is_divisible']}")
                print(f"    Padded to: {dims['padded_to']}×{dims['padded_to']}px")
                print(f"    Token grid: {dims['token_grid']}")
                print(f"    Board size: {dims['board_size_px']}px")
                print(f"    Cell size: {dims['cell_size']:.2f}px")
                print(f"    Border: {dims['border']}px")
                print()

        print("=" * 70)


if __name__ == "__main__":
    # Initialize generator
    generator = TicTacToeResolutionTestGenerator(
        output_dir="tictactoe_resolution_tests", patch_size=16
    )

    # Analyze configurations
    generator.analyze_resolutions()

    # Generate test suite
    test_metadata = generator.generate_test_suite(n_samples_per_resolution=30)

    # Generate debug samples
    generator.generate_debug_samples()

    print("\n🎯 Next steps:")
    print("1. Check debug images to verify resolutions")
    print("2. Run VLM tests using the runner script")
    print("3. Compare divisible vs non-divisible groups")
    print("4. Compare 3×3 Tic-Tac-Toe with 15×15 Gomoku results")
