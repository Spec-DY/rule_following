import numpy as np
from PIL import Image, ImageDraw, ImageFont
import random
import json
from pathlib import Path


class GomokuPatchTestGenerator:
    """Generate Gomoku board images with controlled patch alignment for VLM testing."""

    def __init__(self, output_dir="gomoku_patch_tests", patch_size=16):
        self.output_dir = Path(output_dir)
        self.patch_size = patch_size  # 16x16 patches
        self.board_size = 15  # Standard Gomoku board (15x15 intersections)

        # Fixed dimensions
        self.total_image_size = 512  # 32 x 16 = 512 pixels (32x32 patches)
        self.board_pixel_size = 448  # 28 x 16 = 448 pixels (28x28 patches)

        # Create output directories
        self.conditions = [
            "aligned",
            "offset_quarter",
            "offset_half",
            "offset_three_quarter",
        ]
        for condition in self.conditions:
            (self.output_dir / condition).mkdir(parents=True, exist_ok=True)
        (self.output_dir / "debug").mkdir(parents=True, exist_ok=True)

        # Colors
        self.colors = {
            "background": (245, 245, 240),  # Light beige
            "board": (220, 179, 92),  # Wood color
            "line": (0, 0, 0),  # Black lines
            "black_stone": (20, 20, 20),  # Black stones
            "white_stone": (245, 245, 245),  # White stones
            "coordinate": (50, 50, 50),  # Dark gray for labels
        }

        # Define offset conditions for 16x16 patches
        # Board starts at different positions to create different alignments
        self.offset_conditions = {
            "aligned": {
                "offset": 0,  # Start at pixel 32 (2 patches in)
                "description": "Board intersections aligned with patch centers",
            },
            "offset_quarter": {
                "offset": 4,  # Start at pixel 36 (shift by 1/4 patch)
                "description": "Board offset by quarter patch (4px)",
            },
            "offset_half": {
                "offset": 8,  # Start at pixel 40 (shift by 1/2 patch)
                "description": "Board intersections at patch boundaries",
            },
            "offset_three_quarter": {
                "offset": 12,  # Start at pixel 44 (shift by 3/4 patch)
                "description": "Board offset by three-quarter patch (12px)",
            },
        }

        # Base position for aligned case (leaves room on all sides)
        self.base_board_position = 32  # 2 patches from edge (32px)

        # System instruction
        self.system_instruction = (
            "You are looking at a 15x15 Gomoku board image. "
            "The board has 15 rows (labeled A-O from top to bottom) and 15 columns (labeled 0-14 from left to right).\n\n"
            "Task: Convert what you see into a 15x15 number matrix.\n"
            "Rules:\n"
            "- Empty intersection = 0\n"
            "- Black stone (dark circle) = 1\n"
            "- White stone (light circle) = 2\n\n"
            "CRITICAL: Output EXACTLY in this format:\n"
            "Game State: [[row_A_all_15_numbers], [row_B_all_15_numbers], ..., [row_O_all_15_numbers]]\n\n"
            "Remember:\n"
            "1. Start from row A (top) and go to row O (bottom)\n"
            "2. Each row must have EXACTLY 15 numbers\n"
            "3. Total output must be EXACTLY 15 rows\n"
            "4. Look carefully at each intersection to identify stones"
        )

    def generate_random_board(self, density="medium") -> np.ndarray:
        """Generate random Gomoku board state with controlled density."""
        board = np.zeros((self.board_size, self.board_size), dtype=int)

        density_ranges = {"low": (10, 20), "medium": (25, 40), "high": (45, 60)}
        min_stones, max_stones = density_ranges.get(density, (25, 40))
        num_stones = random.randint(min_stones, max_stones)

        all_positions = [
            (i, j) for i in range(self.board_size) for j in range(self.board_size)
        ]
        selected_positions = random.sample(all_positions, num_stones)

        for idx, (row, col) in enumerate(selected_positions):
            board[row, col] = 1 if idx % 2 == 0 else 2

        return board

    def generate_pattern_board(self, pattern_type: str) -> np.ndarray:
        """Generate specific test patterns."""
        board = np.zeros((self.board_size, self.board_size), dtype=int)

        if pattern_type == "corners":
            # Stones in corners
            corners = [(0, 0), (0, 14), (14, 0), (14, 14)]
            for idx, (r, c) in enumerate(corners):
                board[r, c] = 1 if idx % 2 == 0 else 2

        elif pattern_type == "center":
            # Center cluster
            center = 7
            for dr in range(-2, 3):
                for dc in range(-2, 3):
                    if abs(dr) + abs(dc) <= 3:
                        board[center + dr, center + dc] = 1 if (dr + dc) % 2 == 0 else 2

        elif pattern_type == "grid":
            # Regular grid pattern (good for testing systematic effects)
            for i in range(0, 15, 2):
                for j in range(0, 15, 2):
                    board[i, j] = 1 if (i + j) % 4 == 0 else 2

        return board

    def render_board_image(
        self, board: np.ndarray, condition: str, show_patch_overlay: bool = False
    ) -> Image.Image:
        """Render Gomoku board with controlled offset for patch alignment testing.

        Args:
            board: 15x15 numpy array of board state
            condition: One of "aligned", "offset_quarter", "offset_half", "offset_three_quarter"
            show_patch_overlay: If True, overlay patch grid for debugging
        """
        # Create fixed-size canvas
        img = Image.new(
            "RGB",
            (self.total_image_size, self.total_image_size),
            self.colors["background"],
        )
        draw = ImageDraw.Draw(img)

        # Calculate board position based on condition
        offset = self.offset_conditions[condition]["offset"]
        board_start_x = self.base_board_position + offset
        board_start_y = self.base_board_position + offset

        # Cell size (distance between intersections)
        cell_size = self.board_pixel_size / (self.board_size - 1)  # 448/14 = 32 pixels

        # Draw board background
        board_margin = 15  # Extra margin around grid lines
        draw.rectangle(
            [
                board_start_x - board_margin,
                board_start_y - board_margin,
                board_start_x + self.board_pixel_size + board_margin,
                board_start_y + self.board_pixel_size + board_margin,
            ],
            fill=self.colors["board"],
        )

        # Draw grid lines
        for i in range(self.board_size):
            # Position of this line
            x = board_start_x + i * cell_size
            y = board_start_y + i * cell_size

            # Horizontal lines
            draw.line(
                [(board_start_x, y), (board_start_x + self.board_pixel_size, y)],
                fill=self.colors["line"],
                width=1,
            )
            # Vertical lines
            draw.line(
                [(x, board_start_y), (x, board_start_y + self.board_pixel_size)],
                fill=self.colors["line"],
                width=1,
            )

        # Draw star points (traditional Gomoku markers)
        star_points = [
            (3, 3),
            (3, 7),
            (3, 11),
            (7, 3),
            (7, 7),
            (7, 11),
            (11, 3),
            (11, 7),
            (11, 11),
        ]
        for row, col in star_points:
            x = board_start_x + col * cell_size
            y = board_start_y + row * cell_size
            draw.ellipse([x - 3, y - 3, x + 3, y + 3], fill=self.colors["line"])

        # Draw coordinates
        try:
            font = ImageFont.truetype("arial.ttf", 12)
        except:
            font = ImageFont.load_default()

        for i in range(self.board_size):
            # Row labels (A-O) on the left
            row_label = chr(ord("A") + i)
            y_pos = board_start_y + i * cell_size
            draw.text(
                (board_start_x - 20, y_pos - 6),
                row_label,
                fill=self.colors["coordinate"],
                font=font,
            )

            # Column labels (0-14) on top
            col_label = str(i)
            x_pos = board_start_x + i * cell_size
            text_bbox = draw.textbbox((0, 0), col_label, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            draw.text(
                (x_pos - text_width / 2, board_start_y - 20),
                col_label,
                fill=self.colors["coordinate"],
                font=font,
            )

        # Draw stones
        stone_radius = cell_size * 0.4  # Slightly smaller than half cell
        for row in range(self.board_size):
            for col in range(self.board_size):
                if board[row, col] != 0:
                    x = board_start_x + col * cell_size
                    y = board_start_y + row * cell_size

                    if board[row, col] == 1:  # Black stone
                        draw.ellipse(
                            [
                                x - stone_radius,
                                y - stone_radius,
                                x + stone_radius,
                                y + stone_radius,
                            ],
                            fill=self.colors["black_stone"],
                        )
                        # Small highlight for 3D effect
                        highlight_r = stone_radius * 0.15
                        highlight_offset = stone_radius * 0.3
                        draw.ellipse(
                            [
                                x - highlight_offset - highlight_r,
                                y - highlight_offset - highlight_r,
                                x - highlight_offset + highlight_r,
                                y - highlight_offset + highlight_r,
                            ],
                            fill=(60, 60, 60),
                        )
                    else:  # White stone
                        draw.ellipse(
                            [
                                x - stone_radius,
                                y - stone_radius,
                                x + stone_radius,
                                y + stone_radius,
                            ],
                            fill=self.colors["white_stone"],
                            outline=(200, 200, 200),
                            width=1,
                        )
                        # Highlight
                        highlight_r = stone_radius * 0.25
                        highlight_offset = stone_radius * 0.25
                        draw.ellipse(
                            [
                                x - highlight_offset - highlight_r,
                                y - highlight_offset - highlight_r,
                                x - highlight_offset + highlight_r,
                                y - highlight_offset + highlight_r,
                            ],
                            fill=(255, 255, 255),
                        )

        # Optional: Add patch grid overlay for debugging
        if show_patch_overlay:
            overlay = Image.new(
                "RGBA", (self.total_image_size, self.total_image_size), (0, 0, 0, 0)
            )
            overlay_draw = ImageDraw.Draw(overlay)

            # Draw 16x16 patch grid
            for i in range(0, self.total_image_size + 1, self.patch_size):
                # Vertical lines
                overlay_draw.line(
                    [(i, 0), (i, self.total_image_size)], fill=(255, 0, 0, 60), width=1
                )
                # Horizontal lines
                overlay_draw.line(
                    [(0, i), (self.total_image_size, i)], fill=(255, 0, 0, 60), width=1
                )

            # Highlight every 4th line (64x64 blocks) for visual clarity
            for i in range(0, self.total_image_size + 1, self.patch_size * 4):
                overlay_draw.line(
                    [(i, 0), (i, self.total_image_size)], fill=(0, 0, 255, 100), width=2
                )
                overlay_draw.line(
                    [(0, i), (self.total_image_size, i)], fill=(0, 0, 255, 100), width=2
                )

            # Add condition info
            try:
                font = ImageFont.truetype("arial.ttf", 14)
            except:
                font = ImageFont.load_default()

            info_text = f"Condition: {condition} | Offset: {offset}px | Board start: ({board_start_x}, {board_start_y})"
            overlay_draw.rectangle([(5, 5), (400, 25)], fill=(255, 255, 255, 200))
            overlay_draw.text((10, 8), info_text, fill=(0, 0, 0, 255), font=font)

            # Mark some key intersection positions for analysis
            sample_positions = [(0, 0), (7, 7), (14, 14)]  # First, middle, last
            for row, col in sample_positions:
                x = board_start_x + col * cell_size
                y = board_start_y + row * cell_size
                # Which patch does this intersection fall into?
                patch_x = x // self.patch_size
                patch_y = y // self.patch_size
                offset_in_patch_x = x % self.patch_size
                offset_in_patch_y = y % self.patch_size

                # Mark the intersection
                overlay_draw.ellipse(
                    [x - 5, y - 5, x + 5, y + 5], outline=(0, 255, 0, 200), width=2
                )

                # Add label
                label = f"({row},{col})\nP({patch_x},{patch_y})\nOff({offset_in_patch_x},{offset_in_patch_y})"
                overlay_draw.text(
                    (x + 10, y - 10), label, fill=(0, 255, 0, 200), font=font
                )

            # Composite overlay
            img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")

        return img

    def generate_test_suite(self, n_samples_per_condition: int = 10):
        """Generate test suite with multiple samples per alignment condition."""
        print(f"Generating Gomoku Patch Alignment Test Suite")
        print(f"  Total image size: {self.total_image_size}×{self.total_image_size}px")
        print(f"  Board size: {self.board_pixel_size}×{self.board_pixel_size}px")
        print(f"  Patch size: {self.patch_size}×{self.patch_size}px")
        print(f"  Samples per condition: {n_samples_per_condition}")
        print(f"  Total samples: {n_samples_per_condition * len(self.conditions)}\n")

        test_metadata = {
            "image_size": self.total_image_size,
            "board_pixel_size": self.board_pixel_size,
            "patch_size": self.patch_size,
            "board_game_size": self.board_size,
            "conditions": self.offset_conditions,
            "test_cases": [],
        }

        # Generate same board states for fair comparison
        board_states = []
        for i in range(n_samples_per_condition):
            if i < n_samples_per_condition // 3:
                board = self.generate_random_board("low")
            elif i < 2 * n_samples_per_condition // 3:
                board = self.generate_random_board("medium")
            else:
                board = self.generate_random_board("high")
            board_states.append(board)

        # Generate images for each condition using same board states
        for condition in self.conditions:
            print(f"Generating {condition} samples...")

            for sample_idx, board in enumerate(board_states):
                # Render image
                img = self.render_board_image(
                    board, condition, show_patch_overlay=False
                )

                # Save image
                filename = f"gomoku_{condition}_{sample_idx:03d}.png"
                filepath = self.output_dir / condition / filename
                img.save(filepath)

                # Create test case
                test_case = {
                    "test_id": f"gomoku_patch_{condition}_{sample_idx:03d}",
                    "condition": condition,
                    "offset": self.offset_conditions[condition]["offset"],
                    "sample_index": sample_idx,
                    "image_file": str(filepath),
                    "ground_truth": board.tolist(),
                    "prompt": self.system_instruction,
                    "statistics": {
                        "total_stones": int(np.sum(board > 0)),
                        "black_stones": int(np.sum(board == 1)),
                        "white_stones": int(np.sum(board == 2)),
                        "density": float(np.sum(board > 0) / (self.board_size**2)),
                    },
                }

                # Save individual test case JSON
                test_json = self.output_dir / condition / f"test_{sample_idx:03d}.json"
                with open(test_json, "w") as f:
                    json.dump(test_case, f, indent=2)

                test_metadata["test_cases"].append(test_case)

            print(f"  ✓ Generated {n_samples_per_condition} {condition} samples")

        # Save overall metadata
        metadata_file = self.output_dir / "test_metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(test_metadata, f, indent=2)

        print(f"\n✅ Test suite generated successfully!")
        print(f"📁 Output directory: {self.output_dir}")

        return test_metadata

    def generate_debug_samples(self):
        """Generate debug samples with patch grid overlay."""
        print("\nGenerating debug samples with patch grid overlay...")

        # Use a simple pattern for clear visualization
        board = self.generate_pattern_board("grid")

        for condition in self.conditions:
            img = self.render_board_image(board, condition, show_patch_overlay=True)

            filepath = self.output_dir / "debug" / f"debug_{condition}_overlay.png"
            img.save(filepath)
            print(f"  Saved: {filepath.name}")

        print("✅ Debug samples generated!")

    def analyze_alignment(self):
        """Print detailed analysis of patch alignment for each condition."""
        print("\n" + "=" * 60)
        print("PATCH ALIGNMENT ANALYSIS")
        print("=" * 60)
        print(f"Image size: {self.total_image_size}×{self.total_image_size}px")
        print(
            f"Total patches: {self.total_image_size//self.patch_size}×{self.total_image_size//self.patch_size}"
        )
        print(f"Board size: {self.board_pixel_size}×{self.board_pixel_size}px")
        print(
            f"Board patches: {self.board_pixel_size//self.patch_size}×{self.board_pixel_size//self.patch_size}"
        )
        print(f"Cell size: {self.board_pixel_size/(self.board_size-1):.1f}px")

        cell_size = self.board_pixel_size / (self.board_size - 1)

        for condition, params in self.offset_conditions.items():
            offset = params["offset"]
            board_start = self.base_board_position + offset

            print(f"\n[{condition.upper()}]")
            print(f"  Offset: {offset}px")
            print(f"  Board position: ({board_start}, {board_start})")
            print(f"  Description: {params['description']}")

            # Analyze key intersections
            print(f"  Key intersections (relative to 16×16 patch grid):")
            test_points = [
                (0, 0, "Top-left"),
                (7, 7, "Center"),
                (14, 14, "Bottom-right"),
            ]

            for row, col, label in test_points:
                x = board_start + col * cell_size
                y = board_start + row * cell_size
                patch_x = int(x // self.patch_size)
                patch_y = int(y // self.patch_size)
                offset_x = x % self.patch_size
                offset_y = y % self.patch_size

                print(
                    f"    {label} ({row},{col}): "
                    f"Pixel({x:.0f},{y:.0f}) → "
                    f"Patch({patch_x},{patch_y}) + "
                    f"Offset({offset_x:.0f},{offset_y:.0f})"
                )

        print("\n" + "=" * 60)


# Example usage
if __name__ == "__main__":
    # Initialize generator
    generator = GomokuPatchTestGenerator(output_dir="gomoku_patch_tests", patch_size=16)

    # Analyze alignment details
    generator.analyze_alignment()

    # Generate main test suite
    test_metadata = generator.generate_test_suite(n_samples_per_condition=10)

    # Generate debug samples with overlay
    generator.generate_debug_samples()

    print("\n🎯 Next steps:")
    print("1. Check debug images to verify patch alignment")
    print("2. Run VLM tests on all conditions")
    print("3. Compare accuracy across alignment conditions")
