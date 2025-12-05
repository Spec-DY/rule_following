import numpy as np
from PIL import Image, ImageDraw, ImageFont
import random
import json
from pathlib import Path


class GomokuDensityDiagnosticTest:
    """
    Diagnostic test to isolate density effects on VLM perception.

    Design:
    - Single visual style (3D rendered with PNG assets)
    - Three density levels: Low (20-30%), Medium (40-50%), High (60-70%)
    - Same board size (15×15), resolution (1024×1024)
    - Focus: Does density affect stone detection capability?
    """

    def __init__(self, output_dir="gomoku_density_test", board_size=15):
        self.output_dir = Path(output_dir)
        self.board_size = board_size

        # Fixed parameters (controlled)
        self.resolution = 1024
        self.board_to_image_ratio = 0.75

        # Asset paths - adjusted for new structure (from tests/density/ to ../assets/)
        script_dir = Path(__file__).parent
        assets_dir = script_dir.parent.parent / "assets"
        self.assets = {
            "wood_texture": str(assets_dir / "wood_texture.jpg"),
            "black_stone": str(assets_dir / "black_stone.png"),
            "white_stone": str(assets_dir / "white_stone.png"),
        }

        # Density configurations (experimental variable)
        self.density_levels = {
            "low": {
                "range": (
                    int(board_size * board_size * 0.20),
                    int(board_size * board_size * 0.30),
                ),
                "description": "20-30% occupancy",
            },
            "medium": {
                "range": (
                    int(board_size * board_size * 0.40),
                    int(board_size * board_size * 0.50),
                ),
                "description": "40-50% occupancy",
            },
            "high": {
                "range": (
                    int(board_size * board_size * 0.60),
                    int(board_size * board_size * 0.70),
                ),
                "description": "60-70% occupancy",
            },
        }

        # Star points for traditional layout
        self.star_points = [
            (1, 1),
            (1, 7),
            (1, 13),
            (7, 1),
            (7, 7),
            (7, 13),
            (13, 1),
            (13, 7),
            (13, 13),
        ]

        self._load_assets()

        # Create directory structure
        for density_name in self.density_levels.keys():
            (self.output_dir / density_name).mkdir(parents=True, exist_ok=True)
        (self.output_dir / "debug").mkdir(parents=True, exist_ok=True)

    def _load_assets(self):
        """Load PNG assets."""
        print("Loading PNG assets...")

        try:
            self.wood_texture = Image.open(self.assets["wood_texture"]).convert("RGB")
            print(f"  ✓ Wood texture: {self.wood_texture.size}")
        except FileNotFoundError:
            print(f"  ✗ Wood texture not found, using fallback")
            self.wood_texture = None

        try:
            self.black_stone = Image.open(self.assets["black_stone"]).convert("RGBA")
            print(f"  ✓ Black stone: {self.black_stone.size}")
        except FileNotFoundError:
            print(f"  ✗ Black stone not found")
            self.black_stone = None

        try:
            self.white_stone = Image.open(self.assets["white_stone"]).convert("RGBA")
            print(f"  ✓ White stone: {self.white_stone.size}")
        except FileNotFoundError:
            print(f"  ✗ White stone not found")
            self.white_stone = None

        print()

    def _calculate_dimensions(self):
        """Calculate board dimensions."""
        image_size = self.resolution
        board_size_px = int(image_size * self.board_to_image_ratio)
        board_border = (image_size - board_size_px) // 2

        grid_margin = int(board_size_px * 0.08)
        grid_size_px = board_size_px - 2 * grid_margin
        cell_size = grid_size_px / (self.board_size - 1)
        grid_border = board_border + grid_margin

        return {
            "image_size": image_size,
            "board_size_px": board_size_px,
            "board_border": board_border,
            "grid_border": grid_border,
            "cell_size": cell_size,
        }

    def generate_board_state(self, density_level: str) -> np.ndarray:
        """Generate random board state for given density."""
        board = np.zeros((self.board_size, self.board_size), dtype=int)

        min_pieces, max_pieces = self.density_levels[density_level]["range"]
        num_pieces = random.randint(min_pieces, max_pieces)

        # Randomly select positions
        all_positions = [
            (i, j) for i in range(self.board_size) for j in range(self.board_size)
        ]
        selected = random.sample(all_positions, num_pieces)

        # Alternate black and white
        for idx, (row, col) in enumerate(selected):
            board[row, col] = 1 if idx % 2 == 0 else 2

        return board

    def _get_wood_background(self, size: int) -> Image.Image:
        """Get tiled wood texture."""
        if self.wood_texture is None:
            return Image.new("RGB", (size, size), (220, 180, 120))

        tiles_x = (size // self.wood_texture.width) + 2
        tiles_y = (size // self.wood_texture.height) + 2

        tiled = Image.new(
            "RGB",
            (self.wood_texture.width * tiles_x, self.wood_texture.height * tiles_y),
        )

        for i in range(tiles_x):
            for j in range(tiles_y):
                tiled.paste(
                    self.wood_texture,
                    (i * self.wood_texture.width, j * self.wood_texture.height),
                )

        return tiled.crop((0, 0, size, size))

    def render_board(self, board: np.ndarray, dimensions: dict) -> Image.Image:
        """Render 3D-style board with PNG assets."""
        img_size = dimensions["image_size"]
        board_size_px = dimensions["board_size_px"]
        board_border = dimensions["board_border"]
        grid_border = dimensions["grid_border"]
        cell_size = dimensions["cell_size"]

        # Background
        img = Image.new("RGB", (img_size, img_size), (245, 242, 238))

        # Wood board with rounded corners
        wood_bg = self._get_wood_background(board_size_px)
        mask = Image.new("L", (board_size_px, board_size_px), 0)
        mask_draw = ImageDraw.Draw(mask)
        corner_radius = int(board_size_px * 0.03)
        mask_draw.rounded_rectangle(
            [(0, 0), (board_size_px, board_size_px)], radius=corner_radius, fill=255
        )
        img.paste(wood_bg, (board_border, board_border), mask)

        img = img.convert("RGBA")
        draw = ImageDraw.Draw(img)

        # Grid lines
        line_width = max(2, int(cell_size * 0.035))
        line_color = (70, 45, 25, 255)

        for i in range(self.board_size):
            pos = grid_border + i * cell_size
            draw.line(
                [
                    (grid_border, pos),
                    (grid_border + (self.board_size - 1) * cell_size, pos),
                ],
                fill=line_color,
                width=line_width,
            )
            draw.line(
                [
                    (pos, grid_border),
                    (pos, grid_border + (self.board_size - 1) * cell_size),
                ],
                fill=line_color,
                width=line_width,
            )

        # Star points
        star_radius = int(cell_size * 0.20)
        for row, col in self.star_points:
            x = grid_border + col * cell_size
            y = grid_border + row * cell_size
            draw.ellipse(
                [x - star_radius, y - star_radius, x + star_radius, y + star_radius],
                fill=(70, 45, 25, 255),
            )

        # Coordinate labels
        font_size = max(16, int(cell_size * 0.5))
        try:
            font = ImageFont.truetype("FiraCode-SemiBold.ttf", font_size)
        except:
            font = ImageFont.load_default()

        label_offset = int(cell_size * 0.8)
        edge = grid_border + (self.board_size - 1) * cell_size

        for i in range(self.board_size):
            row_label = chr(ord("A") + i)
            col_label = str(i)
            y = grid_border + i * cell_size
            x = grid_border + i * cell_size

            # All four sides
            draw.text(
                (grid_border - label_offset, y),
                row_label,
                fill=(80, 50, 30, 255),
                font=font,
                anchor="rm",
            )
            draw.text(
                (edge + label_offset, y),
                row_label,
                fill=(80, 50, 30, 255),
                font=font,
                anchor="lm",
            )
            draw.text(
                (x, grid_border - label_offset),
                col_label,
                fill=(80, 50, 30, 255),
                font=font,
                anchor="mb",
            )
            draw.text(
                (x, edge + label_offset),
                col_label,
                fill=(80, 50, 30, 255),
                font=font,
                anchor="mt",
            )

        # Draw stones using PNG assets
        stone_radius = cell_size * 0.48

        for i in range(self.board_size):
            for j in range(self.board_size):
                if board[i, j] != 0:
                    cx = grid_border + j * cell_size
                    cy = grid_border + i * cell_size

                    stone_asset = (
                        self.black_stone if board[i, j] == 1 else self.white_stone
                    )

                    if stone_asset is not None:
                        stone_size = int(stone_radius * 2)
                        stone_resized = stone_asset.resize(
                            (stone_size, stone_size), Image.Resampling.LANCZOS
                        )
                        paste_x = int(cx - stone_radius)
                        paste_y = int(cy - stone_radius)
                        img.paste(stone_resized, (paste_x, paste_y), stone_resized)

        return img.convert("RGB")

    def generate_density_test_suite(self, n_samples_per_density: int = 30):
        """Generate complete density diagnostic test suite."""
        print("=" * 70)
        print("GOMOKU DENSITY DIAGNOSTIC TEST")
        print("=" * 70)
        print(f"Board size: {self.board_size}×{self.board_size}")
        print(f"Resolution: {self.resolution}×{self.resolution}px")
        print(f"Visual style: 3D rendered (controlled)")
        print(f"Samples per density: {n_samples_per_density}")
        print()

        test_metadata = {
            "board_size": self.board_size,
            "resolution": self.resolution,
            "board_to_image_ratio": self.board_to_image_ratio,
            "visual_style": "3D rendered with PNG assets (controlled)",
            "experimental_variable": "density",
            "density_levels": {},
            "test_cases": [],
        }

        dimensions = self._calculate_dimensions()

        # Generate for each density level
        for density_name, density_config in self.density_levels.items():
            print(f"\n{'='*70}")
            print(f"Generating {density_name.upper()} density boards")
            print(f"  Target: {density_config['description']}")
            print(f"{'='*70}")

            boards = []
            for _ in range(n_samples_per_density):
                board = self.generate_board_state(density_name)
                boards.append(board)

            # Calculate actual density statistics
            piece_counts = [np.sum(b > 0) for b in boards]
            avg_pieces = np.mean(piece_counts)
            avg_density = avg_pieces / (self.board_size * self.board_size)

            print(f"  Generated {len(boards)} boards")
            print(f"  Actual average: {avg_pieces:.1f} pieces ({avg_density:.1%})")
            print(f"  Range: {min(piece_counts)}-{max(piece_counts)} pieces")

            test_metadata["density_levels"][density_name] = {
                "target_description": density_config["description"],
                "actual_mean_pieces": float(avg_pieces),
                "actual_mean_density": float(avg_density),
                "piece_count_range": [int(min(piece_counts)), int(max(piece_counts))],
                "n_samples": len(boards),
            }

            # Render images
            print(f"  Rendering images...")
            for idx, board in enumerate(boards):
                img = self.render_board(board, dimensions)

                filename = f"gomoku_density_{density_name}_{idx:03d}.png"
                filepath = self.output_dir / density_name / filename
                img.save(filepath)

                # Save test case metadata
                test_case = {
                    "test_id": f"density_{density_name}_{idx:03d}",
                    "density_level": density_name,
                    "sample_index": idx,
                    "image_file": str(filepath),
                    "ground_truth": board.tolist(),
                    "statistics": {
                        "total_pieces": int(np.sum(board > 0)),
                        "black_count": int(np.sum(board == 1)),
                        "white_count": int(np.sum(board == 2)),
                        "density": float(np.sum(board > 0) / (self.board_size**2)),
                    },
                }

                # Save individual test JSON
                test_json = filepath.parent / f"test_{idx:03d}.json"
                with open(test_json, "w") as f:
                    json.dump(test_case, f, indent=2)

                test_metadata["test_cases"].append(test_case)

            print(f"  ✓ Completed {density_name}")

        # Save master metadata
        metadata_file = self.output_dir / "test_metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(test_metadata, f, indent=2)

        print(f"\n{'='*70}")
        print(f"✅ Test suite complete!")
        print(f"{'='*70}")
        print(f"\nGenerated test structure:")
        print(f"  gomoku_density_test/")
        for density_name in self.density_levels.keys():
            print(f"    ├─ {density_name}/ ({n_samples_per_density} images)")
        print(f"    └─ test_metadata.json")
        print()
        print(f"Total test cases: {len(test_metadata['test_cases'])}")

        return test_metadata

    def generate_visual_comparison(self):
        """Generate side-by-side comparison of all three densities."""
        print("\nGenerating density comparison samples...")

        dimensions = self._calculate_dimensions()

        for sample_idx in range(3):
            # Generate one board for each density
            boards = {
                name: self.generate_board_state(name)
                for name in self.density_levels.keys()
            }

            # Render each
            images = {
                name: self.render_board(board, dimensions)
                for name, board in boards.items()
            }

            # Create comparison image (3 boards side by side)
            comparison = Image.new(
                "RGB",
                (self.resolution * 3 + 80, self.resolution + 100),
                (255, 255, 255),
            )

            # Paste boards
            x_positions = [20, self.resolution + 40, self.resolution * 2 + 60]
            for (name, img), x_pos in zip(images.items(), x_positions):
                comparison.paste(img, (x_pos, 80))

            # Add labels
            draw = ImageDraw.Draw(comparison)
            try:
                title_font = ImageFont.truetype("FiraCode-SemiBold.ttf", 28)
                label_font = ImageFont.truetype("FiraCode-SemiBold.ttf", 20)
            except:
                title_font = ImageFont.load_default()
                label_font = ImageFont.load_default()

            # Title
            draw.text(
                (comparison.width // 2, 20),
                "Density Diagnostic Test",
                fill=(0, 0, 0),
                font=title_font,
                anchor="mm",
            )

            # Density labels
            labels = {
                "low": f"LOW ({self.density_levels['low']['description']})",
                "medium": f"MEDIUM ({self.density_levels['medium']['description']})",
                "high": f"HIGH ({self.density_levels['high']['description']})",
            }

            for (name, label), x_pos in zip(labels.items(), x_positions):
                pieces = np.sum(boards[name] > 0)
                density_pct = pieces / (self.board_size**2) * 100

                text = f"{label}\n{pieces} pieces ({density_pct:.1f}%)"
                draw.text(
                    (x_pos + self.resolution // 2, 50),
                    text,
                    fill=(0, 0, 0),
                    font=label_font,
                    anchor="mm",
                )

            # Save
            filepath = (
                self.output_dir / "debug" / f"density_comparison_{sample_idx:03d}.png"
            )
            comparison.save(filepath)
            print(f"  ✓ Saved: {filepath.name}")

        print("✅ Comparison samples generated!")


if __name__ == "__main__":
    generator = GomokuDensityDiagnosticTest(
        output_dir="gomoku_density_test", board_size=15
    )

    print("\n" + "=" * 70)
    print("DENSITY DIAGNOSTIC TEST GENERATOR")
    print("=" * 70)
    print("\nExperimental Design:")
    print("  Controlled factors:")
    print("    - Visual style: 3D rendered (PNG assets)")
    print("    - Board size: 15×15")
    print("    - Resolution: 1024×1024px")
    print("    - Rendering: Realistic wood texture, glossy stones")
    print()
    print("  Experimental variable:")
    print("    - Density: Low (20-30%) vs Medium (40-50%) vs High (60-70%)")
    print()
    print("  Research Question:")
    print("    Does board density affect VLM stone detection capability?")
    print("=" * 70)
    print()

    # Generate test suite
    metadata = generator.generate_density_test_suite(n_samples_per_density=30)

    # Generate comparison images
    generator.generate_visual_comparison()

    print("\n" + "=" * 70)
    print("NEXT STEPS:")
    print("=" * 70)
    print("1. Run VLM tests on all three density levels")
    print("2. Compare using Stone-Only Accuracy metric")
    print("3. Plot: Density vs Stone Detection Rate")
    print("4. Expected finding: Small models show paradoxical improvement")
    print("   with density, large models remain stable")
    print("=" * 70)
