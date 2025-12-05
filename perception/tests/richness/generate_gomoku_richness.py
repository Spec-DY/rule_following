import numpy as np
from PIL import Image, ImageDraw, ImageFont
import random
import json
from pathlib import Path


class GomokuVisualRichnessTestGenerator:
    """Generate traditional Gomoku board images with stones on intersections."""

    def __init__(self, output_dir="gomoku_visual_richness_tests", board_size=15):
        self.output_dir = Path(output_dir)
        self.board_size = board_size

        # Fixed resolution
        self.resolution = 1024
        self.board_to_image_ratio = 0.75

        # Asset paths - adjusted for new structure (from tests/richness/ to ../assets/)
        script_dir = Path(__file__).parent
        assets_dir = script_dir.parent.parent / "assets"
        self.assets = {
            "wood_texture": str(assets_dir / "wood_texture.jpg"),
            "black_stone": str(assets_dir / "black_stone.png"),
            "white_stone": str(assets_dir / "white_stone.png"),
        }

        # Style configurations
        self.style_groups = {
            "2d_flat": {
                "description": "Minimalist 2D geometric shapes",
                "renderer": self._render_2d,
            },
            "3d_rendered": {
                "description": "Realistic 3D using PNG assets",
                "renderer": self._render_3d_with_assets,
            },
        }

        # Colors for 2D
        self.colors_2d = {
            "background": (245, 245, 240),
            "board": (227, 169, 118),
            "line": (93, 57, 44),
            "black_stone": (30, 30, 30),
            "white_stone": (240, 240, 240),
            "coordinate": (93, 57, 44),
            "star_point": (93, 57, 44),
        }

        # Star points for 15×15 board (corners and edges like reference image)
        # Reference image shows star points at board edges, not center grid
        self.star_points_15x15 = [
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

        # Load and preprocess assets
        self._load_assets()

        # Create output directories
        for style in ["2d_flat", "3d_rendered"]:
            (self.output_dir / style).mkdir(parents=True, exist_ok=True)
        (self.output_dir / "debug").mkdir(parents=True, exist_ok=True)

    def _load_assets(self):
        """Load and preprocess PNG assets."""
        print("Loading PNG assets...")

        try:
            self.wood_texture_original = Image.open(
                self.assets["wood_texture"]
            ).convert("RGB")
            print(f"  ✓ Loaded wood texture: {self.wood_texture_original.size}")
        except FileNotFoundError:
            print(f"  ✗ Wood texture not found, using fallback")
            self.wood_texture_original = None

        try:
            self.black_stone_original = Image.open(self.assets["black_stone"]).convert(
                "RGBA"
            )
            print(f"  ✓ Loaded black stone: {self.black_stone_original.size}")
        except FileNotFoundError:
            print(f"  ✗ Black stone not found")
            self.black_stone_original = None

        try:
            self.white_stone_original = Image.open(self.assets["white_stone"]).convert(
                "RGBA"
            )
            print(f"  ✓ Loaded white stone: {self.white_stone_original.size}")
        except FileNotFoundError:
            print(f"  ✗ White stone not found")
            self.white_stone_original = None

        print()

    def _calculate_dimensions(self) -> dict:
        """Calculate dimensions for traditional Go/Gomoku board."""
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
            "grid_size_px": grid_size_px,
            "grid_margin": grid_margin,
            "cell_size": cell_size,
        }

    def generate_random_board(self, density="high") -> np.ndarray:
        """Generate random board state."""
        board = np.zeros((self.board_size, self.board_size), dtype=int)
        total_intersections = self.board_size * self.board_size

        density_ranges = {
            "ultra_low": (
                int(total_intersections * 0.1),
                int(total_intersections * 0.15),
            ),
            "low": (int(total_intersections * 0.2), int(total_intersections * 0.3)),
            "medium": (int(total_intersections * 0.4), int(total_intersections * 0.5)),
            "high": (int(total_intersections * 0.6), int(total_intersections * 0.7)),
        }
        min_pieces, max_pieces = density_ranges.get(density, density_ranges["high"])
        num_pieces = random.randint(min_pieces, max_pieces)

        all_positions = [
            (i, j) for i in range(self.board_size) for j in range(self.board_size)
        ]
        selected_positions = random.sample(all_positions, num_pieces)

        for idx, (row, col) in enumerate(selected_positions):
            board[row, col] = 1 if idx % 2 == 0 else 2

        return board

    def _get_wood_background(self, size: int) -> Image.Image:
        """Get wood texture background (tiled if necessary)."""
        if self.wood_texture_original is None:
            wood = Image.new("RGB", (size, size), (220, 180, 120))
            return wood

        texture = self.wood_texture_original
        tiles_x = (size // texture.width) + 2
        tiles_y = (size // texture.height) + 2

        tiled = Image.new("RGB", (texture.width * tiles_x, texture.height * tiles_y))
        for i in range(tiles_x):
            for j in range(tiles_y):
                tiled.paste(texture, (i * texture.width, j * texture.height))

        return tiled.crop((0, 0, size, size))

    def _draw_star_points(
        self, draw, grid_border: float, cell_size: float, radius: float, color
    ):
        """Draw star points at key intersections."""
        if self.board_size == 15:
            points = self.star_points_15x15
        else:
            center = self.board_size // 2
            points = [(center, center)]

        for row, col in points:
            x = grid_border + col * cell_size
            y = grid_border + row * cell_size
            draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=color)

    def _draw_2d_stone(
        self, draw, cx: float, cy: float, radius: float, stone_type: int
    ):
        """Draw simple 2D stone."""
        color = (
            self.colors_2d["black_stone"]
            if stone_type == 1
            else self.colors_2d["white_stone"]
        )
        draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], fill=color)

    def _draw_png_stone(
        self, img: Image.Image, cx: float, cy: float, radius: float, stone_type: int
    ):
        """Draw stone using PNG asset."""
        stone_asset = (
            self.black_stone_original if stone_type == 1 else self.white_stone_original
        )

        if stone_asset is None:
            draw = ImageDraw.Draw(img)
            self._draw_2d_stone(draw, cx, cy, radius, stone_type)
            return

        stone_size = int(radius * 2)
        stone_resized = stone_asset.resize(
            (stone_size, stone_size), Image.Resampling.LANCZOS
        )

        paste_x = int(cx - radius)
        paste_y = int(cy - radius)
        img.paste(stone_resized, (paste_x, paste_y), stone_resized)

    def _render_2d(self, board: np.ndarray, dimensions: dict) -> Image.Image:
        """Render 2D flat version with traditional layout."""
        image_size = dimensions["image_size"]
        board_size_px = dimensions["board_size_px"]
        board_border = dimensions["board_border"]
        grid_border = dimensions["grid_border"]
        cell_size = dimensions["cell_size"]

        # Create canvas
        img = Image.new("RGB", (image_size, image_size), self.colors_2d["background"])
        draw = ImageDraw.Draw(img)

        # Board background with rounded corners
        corner_radius = int(board_size_px * 0.03)
        draw.rounded_rectangle(
            [
                board_border,
                board_border,
                board_border + board_size_px,
                board_border + board_size_px,
            ],
            radius=corner_radius,
            fill=self.colors_2d["board"],
        )

        # Grid lines at intersections
        line_width = 2
        for i in range(self.board_size):
            pos = grid_border + i * cell_size
            draw.line(
                [
                    (grid_border, pos),
                    (grid_border + (self.board_size - 1) * cell_size, pos),
                ],
                fill=self.colors_2d["line"],
                width=line_width,
            )
            draw.line(
                [
                    (pos, grid_border),
                    (pos, grid_border + (self.board_size - 1) * cell_size),
                ],
                fill=self.colors_2d["line"],
                width=line_width,
            )

        # Star points (larger and more visible)
        star_radius = int(cell_size * 0.20)
        self._draw_star_points(
            draw, grid_border, cell_size, star_radius, self.colors_2d["star_point"]
        )

        # Coordinate labels (larger, bolder, around all four sides)
        font_size = max(16, int(cell_size * 0.5))
        try:
            font = ImageFont.truetype("FiraCode-SemiBold.ttf", font_size)
        except:
            try:
                font = ImageFont.truetype("arial.ttf", font_size)
            except:
                font = ImageFont.load_default()

        label_offset = int(cell_size * 0.8)
        edge = (
            grid_border + (self.board_size - 1) * cell_size
        )  # Position of rightmost/bottommost line

        for i in range(self.board_size):
            row_label = chr(ord("A") + i)
            col_label = str(i)
            y = grid_border + i * cell_size
            x = grid_border + i * cell_size

            # Left row label
            draw.text(
                (grid_border - label_offset, y),
                row_label,
                fill=self.colors_2d["coordinate"],
                font=font,
                anchor="rm",  # right-middle
            )
            # Right row label
            draw.text(
                (edge + label_offset, y),
                row_label,
                fill=self.colors_2d["coordinate"],
                font=font,
                anchor="lm",  # left-middle
            )

            # Top column label
            draw.text(
                (x, grid_border - label_offset),
                col_label,
                fill=self.colors_2d["coordinate"],
                font=font,
                anchor="mb",  # middle-bottom
            )
            # Bottom column label
            draw.text(
                (x, edge + label_offset),
                col_label,
                fill=self.colors_2d["coordinate"],
                font=font,
                anchor="mt",  # middle-top
            )

        # Draw stones at intersections
        stone_radius = cell_size * 0.45
        for i in range(self.board_size):
            for j in range(self.board_size):
                if board[i, j] != 0:
                    cx = grid_border + j * cell_size
                    cy = grid_border + i * cell_size
                    self._draw_2d_stone(draw, cx, cy, stone_radius, board[i, j])

        return img

    def _render_3d_with_assets(
        self, board: np.ndarray, dimensions: dict
    ) -> Image.Image:
        """Render 3D version with traditional layout."""
        image_size = dimensions["image_size"]
        board_size_px = dimensions["board_size_px"]
        board_border = dimensions["board_border"]
        grid_border = dimensions["grid_border"]
        cell_size = dimensions["cell_size"]

        # Background
        img = Image.new("RGB", (image_size, image_size), (245, 242, 238))

        # Wood texture board with rounded corners
        wood_bg = self._get_wood_background(board_size_px)

        # Create rounded mask
        mask = Image.new("L", (board_size_px, board_size_px), 0)
        mask_draw = ImageDraw.Draw(mask)
        corner_radius = int(board_size_px * 0.03)
        mask_draw.rounded_rectangle(
            [(0, 0), (board_size_px, board_size_px)], radius=corner_radius, fill=255
        )

        # Paste wood with rounded corners
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

        # Star points (larger)
        star_radius = int(cell_size * 0.20)
        self._draw_star_points(
            draw, grid_border, cell_size, star_radius, (70, 45, 25, 255)
        )

        # Coordinate labels (larger, bolder, around all four sides)
        font_size = max(16, int(cell_size * 0.5))
        try:
            font = ImageFont.truetype("FiraCode-SemiBold.ttf", font_size)
        except:
            try:
                font = ImageFont.truetype("arial.ttf", font_size)
            except:
                font = ImageFont.load_default()

        label_offset = int(cell_size * 0.8)
        edge = (
            grid_border + (self.board_size - 1) * cell_size
        )  # Position of rightmost/bottommost line

        for i in range(self.board_size):
            row_label = chr(ord("A") + i)
            col_label = str(i)
            y = grid_border + i * cell_size
            x = grid_border + i * cell_size

            # Left row label
            draw.text(
                (grid_border - label_offset, y),
                row_label,
                fill=self.colors_2d["coordinate"],
                font=font,
                anchor="rm",  # right-middle
            )
            # Right row label
            draw.text(
                (edge + label_offset, y),
                row_label,
                fill=self.colors_2d["coordinate"],
                font=font,
                anchor="lm",  # left-middle
            )

            # Top column label
            draw.text(
                (x, grid_border - label_offset),
                col_label,
                fill=self.colors_2d["coordinate"],
                font=font,
                anchor="mb",  # middle-bottom
            )
            # Bottom column label
            draw.text(
                (x, edge + label_offset),
                col_label,
                fill=self.colors_2d["coordinate"],
                font=font,
                anchor="mt",  # middle-top
            )

        # Draw stones at intersections
        stone_radius = cell_size * 0.48

        for i in range(self.board_size):
            for j in range(self.board_size):
                if board[i, j] != 0:
                    cx = grid_border + j * cell_size
                    cy = grid_border + i * cell_size
                    self._draw_png_stone(img, cx, cy, stone_radius, board[i, j])

        img = img.convert("RGB")
        return img

    def generate_test_suite(self, n_samples: int = 30):
        """Generate test suite."""
        print(f"Generating Gomoku Visual Richness Test Suite")
        print(f"  Board size: {self.board_size}×{self.board_size} (traditional)")
        print(f"  Resolution: {self.resolution}×{self.resolution}px")
        print(f"  Samples: {n_samples}")
        print()

        test_metadata = {
            "board_size": self.board_size,
            "resolution": self.resolution,
            "board_to_image_ratio": self.board_to_image_ratio,
            "style_groups": {k: v["description"] for k, v in self.style_groups.items()},
            "density_info": "high (60-70% occupancy)",
            "test_cases": [],
        }

        dimensions = self._calculate_dimensions()

        print(f"Generating {n_samples} board states...")
        board_states = [self.generate_random_board("low") for _ in range(n_samples)]

        total_intersections = self.board_size * self.board_size
        piece_counts = [np.sum(b > 0) for b in board_states]
        avg_pieces = np.mean(piece_counts)
        avg_density = avg_pieces / total_intersections

        print(f"  ✓ Average pieces: {avg_pieces:.1f} ({avg_density:.1%} density)")
        print()

        test_metadata["density_statistics"] = {
            "average_pieces": float(avg_pieces),
            "average_density": float(avg_density),
            "empty_baseline": float(1 - avg_density),
        }

        for style_name, style_info in self.style_groups.items():
            print(f"Processing {style_name.upper()}...")

            for sample_idx, board in enumerate(board_states):
                img = style_info["renderer"](board, dimensions)

                filename = f"gomoku_{style_name}_{sample_idx:03d}.png"
                filepath = self.output_dir / style_name / filename
                img.save(filepath)

                test_case = {
                    "test_id": f"gomoku_visual_{style_name}_{sample_idx:03d}",
                    "style": style_name,
                    "sample_index": sample_idx,
                    "image_file": str(filepath),
                    "dimensions": dimensions,
                    "ground_truth": board.tolist(),
                    "statistics": {
                        "total_pieces": int(np.sum(board > 0)),
                        "black_count": int(np.sum(board == 1)),
                        "white_count": int(np.sum(board == 2)),
                        "density": float(np.sum(board > 0) / total_intersections),
                    },
                }

                test_json = filepath.parent / f"test_{sample_idx:03d}.json"
                with open(test_json, "w") as f:
                    json.dump(test_case, f, indent=2)

                test_metadata["test_cases"].append(test_case)

            print(f"  ✓ Generated {n_samples} images\n")

        metadata_file = self.output_dir / "test_metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(test_metadata, f, indent=2)

        print(f"✅ Complete! Check {self.output_dir}")
        return test_metadata

    def generate_comparison_samples(self, n_samples: int = 3):
        """Generate side-by-side comparisons."""
        print("\nGenerating comparison samples...")

        dimensions = self._calculate_dimensions()

        for i in range(n_samples):
            board = self.generate_random_board("medium")

            img_2d = self._render_2d(board, dimensions)
            img_3d = self._render_3d_with_assets(board, dimensions)

            comparison = Image.new(
                "RGB", (self.resolution * 2 + 40, self.resolution + 80), (255, 255, 255)
            )
            comparison.paste(img_2d, (20, 60))
            comparison.paste(img_3d, (self.resolution + 20, 60))

            draw = ImageDraw.Draw(comparison)
            try:
                font = ImageFont.truetype("arial.ttf", 24)
            except:
                font = ImageFont.load_default()

            draw.text((20, 20), "2D FLAT", fill=(0, 0, 0), font=font)
            draw.text(
                (self.resolution + 20, 20),
                "3D RENDERED (PNG ASSETS)",
                fill=(0, 0, 0),
                font=font,
            )

            filepath = self.output_dir / "debug" / f"comparison_{i:03d}.png"
            comparison.save(filepath)
            print(f"  Saved: {filepath.name}")

        print("✅ Comparison samples generated!")


if __name__ == "__main__":
    generator = GomokuVisualRichnessTestGenerator(
        output_dir="gomoku_visual_richness_tests", board_size=15
    )

    print("=" * 70)
    print("GOMOKU VISUAL RICHNESS TEST (TRADITIONAL GRID)")
    print("=" * 70)
    print()

    test_metadata = generator.generate_test_suite(n_samples=30)
    generator.generate_comparison_samples(n_samples=5)
