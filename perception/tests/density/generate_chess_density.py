import numpy as np
from PIL import Image, ImageDraw, ImageFont
import random
import json
from pathlib import Path
import chess


class ChessDensityDiagnosticTest:
    """
    Diagnostic test to isolate density effects on VLM perception for Chess.

    Design:
    - Simulated gameplay approach using python-chess
    - Three density levels: Low (8-12 pieces), Medium (16-20), High (28-32)
    - Board size: 8x8, Resolution: 1024x1024
    - Focus: Does density affect piece detection capability?
    """

    # LVLM-Playground encoding
    PIECE_ENCODING = {
        "P": 1,
        "N": 2,
        "B": 3,
        "R": 4,
        "Q": 5,
        "K": 6,  # White
        "p": -1,
        "n": -2,
        "b": -3,
        "r": -4,
        "q": -5,
        "k": -6,  # Black
        ".": 0,  # Empty
    }

    def __init__(self, output_dir="chess_density_test", assets_dir="assets"):
        self.output_dir = Path(output_dir)
        # Adjust assets path for new structure (from tests/density/ to ../assets/)
        script_dir = Path(__file__).parent
        if assets_dir == "assets":
            self.assets_dir = script_dir.parent.parent / "assets"
        else:
            self.assets_dir = Path(assets_dir)

        # Fixed parameters
        self.resolution = 1024
        self.board_size = 8

        # [MODIFIED] Increased ratio to make board larger
        self.board_to_image_ratio = 0.92

        # Density configurations
        self.density_levels = {
            "low": {"range": (8, 12), "description": "8-12 pieces (endgame)"},
            "medium": {"range": (16, 20), "description": "16-20 pieces (midgame)"},
            "high": {"range": (28, 32), "description": "28-32 pieces (opening)"},
        }

        self._load_assets()

        # Create directory structure
        for density_name in self.density_levels.keys():
            (self.output_dir / density_name).mkdir(parents=True, exist_ok=True)
        (self.output_dir / "debug").mkdir(parents=True, exist_ok=True)

    def _load_assets(self):
        """Load chess piece PNG assets."""
        print("Loading chess piece assets...")

        self.pieces = {}
        piece_files = {
            "bb": "black bishop",
            "bk": "black king",
            "bn": "black knight",
            "bp": "black pawn",
            "bq": "black queen",
            "br": "black rook",
            "wb": "white bishop",
            "wk": "white king",
            "wn": "white knight",
            "wp": "white pawn",
            "wq": "white queen",
            "wr": "white rook",
        }

        for code, name in piece_files.items():
            filepath = self.assets_dir / f"{code}.png"
            try:
                self.pieces[code] = Image.open(filepath).convert("RGBA")
                print(f"  ✓ Loaded {name}: {self.pieces[code].size}")
            except FileNotFoundError:
                print(f"  ✗ Missing {name} at {filepath}")
                self.pieces[code] = None

        print()

    def _calculate_dimensions(self):
        """Calculate board dimensions."""
        image_size = self.resolution

        # Total board area including outer border
        board_size_px = int(image_size * self.board_to_image_ratio)

        # White space around the board
        board_border = (image_size - board_size_px) // 2

        # [MODIFIED] Reduced margin (the wood frame part) to maximize square size
        # 2% of board size instead of 8%
        grid_margin = int(board_size_px * 0.02)

        grid_size_px = board_size_px - 2 * grid_margin
        square_size = grid_size_px / self.board_size
        grid_border = board_border + grid_margin

        return {
            "image_size": image_size,
            "board_size_px": board_size_px,
            "board_border": board_border,
            "grid_border": grid_border,
            "square_size": square_size,
        }

    def _count_pieces(self, board: chess.Board) -> int:
        """Count total pieces on board."""
        return len(board.piece_map())

    def generate_board_state(
        self, density_level: str, max_attempts: int = 200
    ) -> chess.Board:
        """Generate chess position with target density using simulated gameplay."""
        min_p, max_p = self.density_levels[density_level]["range"]

        # [FIX] Randomly select a specific target count within the range first
        # This ensures we get 9, 10, 11 pieces, not just 12.
        target_piece_count = random.randint(min_p, max_p)

        for attempt in range(max_attempts):
            board = chess.Board()

            if density_level == "high":
                # For high density, we verify we don't drop below min_p
                # Usually opening moves keep 32 pieces, but sometimes captures happen
                moves_to_play = random.randint(3, 8)
                for _ in range(moves_to_play):
                    if board.legal_moves:
                        board.push(random.choice(list(board.legal_moves)))

                # If we accidentally captured too many, try again
                if self._count_pieces(board) < min_p:
                    continue
                return board

            else:
                # Medium and Low density: Play until we hit the specific target
                while self._count_pieces(board) > target_piece_count:
                    if not board.legal_moves:
                        break

                    legal_moves = list(board.legal_moves)
                    capturing_moves = [m for m in legal_moves if board.is_capture(m)]

                    # Dynamic Aggression: Increase capture probability as we get closer/desperate
                    current_count = self._count_pieces(board)

                    if current_count > target_piece_count + 5:
                        capture_prob = 0.8  # Aggressively reduce pieces
                    else:
                        capture_prob = 0.5  # Play more naturally

                    if capturing_moves and random.random() < capture_prob:
                        board.push(random.choice(capturing_moves))
                    else:
                        board.push(random.choice(legal_moves))

                    # Safety break to prevent infinite loops
                    if board.fullmove_number > 250:
                        break

                # Check if we successfully hit the target range
                final_count = self._count_pieces(board)
                if min_p <= final_count <= max_p:
                    return board

        print(
            f"  ⚠️  Could not reach target {target_piece_count} (Got {self._count_pieces(board)})"
        )
        return board

    def _draw_chessboard(self, dimensions: dict) -> Image.Image:
        """Draw chessboard with properly aligned labels."""
        img_size = dimensions["image_size"]
        grid_border = dimensions["grid_border"]
        square_size = dimensions["square_size"]

        # Background (light warm gray)
        img = Image.new("RGB", (img_size, img_size), (245, 242, 238))
        draw = ImageDraw.Draw(img)

        # Colors
        light_square = (240, 217, 181)  # Light brown
        dark_square = (181, 136, 99)  # Dark brown

        # Draw squares
        for row in range(8):
            for col in range(8):
                x = grid_border + col * square_size
                y = grid_border + row * square_size

                # Alternate colors
                color = light_square if (row + col) % 2 == 0 else dark_square

                # Use ceil logic by adding size to ensure no gaps
                draw.rectangle([x, y, x + square_size, y + square_size], fill=color)

        # Font Setup
        font_size = int(square_size * 0.2)
        try:
            font = ImageFont.truetype("FiraCode-SemiBold.ttf", font_size)
        except:
            try:
                font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
            except:
                font = ImageFont.load_default()

        text_color = (60, 40, 20)

        # [MODIFIED] Alignment Logic
        # Distance from the edge of the playing grid
        label_padding = square_size * 0.2

        for i in range(8):
            file_label = chr(ord("a") + i)  # a-h
            rank_label = str(8 - i)  # 8-1

            # --- Files (a-h) ---
            # X: Center of the column
            file_x_center = grid_border + i * square_size + square_size / 2

            # Top labels
            draw.text(
                (file_x_center, grid_border - label_padding),
                file_label,
                fill=text_color,
                font=font,
                anchor="mm",  # Middle Middle alignment
            )
            # Bottom labels
            draw.text(
                (file_x_center, grid_border + 8 * square_size + label_padding),
                file_label,
                fill=text_color,
                font=font,
                anchor="mm",
            )

            # --- Ranks (1-8) ---
            # Y: Center of the row
            rank_y_center = grid_border + i * square_size + square_size / 2

            # Left labels
            draw.text(
                (grid_border - label_padding, rank_y_center),
                rank_label,
                fill=text_color,
                font=font,
                anchor="mm",
            )
            # Right labels
            draw.text(
                (grid_border + 8 * square_size + label_padding, rank_y_center),
                rank_label,
                fill=text_color,
                font=font,
                anchor="mm",
            )

        return img

    def render_board(self, board: chess.Board, dimensions: dict) -> Image.Image:
        """Render chess board with pieces."""
        img = self._draw_chessboard(dimensions)

        grid_border = dimensions["grid_border"]
        square_size = dimensions["square_size"]
        piece_size = int(square_size * 0.90)  # Slightly larger pieces

        # Map python-chess piece symbols to asset codes
        piece_to_asset = {
            "P": "wp",
            "N": "wn",
            "B": "wb",
            "R": "wr",
            "Q": "wq",
            "K": "wk",
            "p": "bp",
            "n": "bn",
            "b": "bb",
            "r": "br",
            "q": "bq",
            "k": "bk",
        }

        # Convert to RGBA for piece transparency
        img = img.convert("RGBA")

        # Place pieces
        for square, piece in board.piece_map().items():
            # Convert square index to row, col
            row = 7 - (square // 8)  # chess lib uses 0=a1, 63=h8
            col = square % 8

            piece_symbol = piece.symbol()
            asset_code = piece_to_asset.get(piece_symbol)

            if asset_code and self.pieces.get(asset_code):
                piece_img = self.pieces[asset_code].resize(
                    (piece_size, piece_size), Image.Resampling.LANCZOS
                )

                x = grid_border + col * square_size + (square_size - piece_size) // 2
                y = grid_border + row * square_size + (square_size - piece_size) // 2

                img.paste(piece_img, (int(x), int(y)), piece_img)

        return img.convert("RGB")

    def board_to_matrix(self, board: chess.Board) -> list:
        """Convert python-chess Board to LVLM-Playground matrix format."""
        matrix = []

        for rank in range(7, -1, -1):  # 8 to 1
            row = []
            for file in range(8):  # a to h
                square = chess.square(file, rank)
                piece = board.piece_at(square)

                if piece is None:
                    row.append(0)
                else:
                    row.append(self.PIECE_ENCODING[piece.symbol()])

            matrix.append(row)

        return matrix

    def generate_density_test_suite(self, n_samples_per_density: int = 30):
        """Generate complete density diagnostic test suite."""
        print("=" * 70)
        print("CHESS DENSITY DIAGNOSTIC TEST")
        print("=" * 70)
        print(f"Board size: {self.board_size}x{self.board_size}")
        print(f"Resolution: {self.resolution}x{self.resolution}px")
        print(f"Generation method: Simulated gameplay")
        print(f"Samples per density: {n_samples_per_density}")
        print()

        test_metadata = {
            "board_size": self.board_size,
            "resolution": self.resolution,
            "board_to_image_ratio": self.board_to_image_ratio,
            "generation_method": "simulated_gameplay",
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
            piece_counts = []

            for i in range(n_samples_per_density):
                board = self.generate_board_state(density_name)
                boards.append(board)
                piece_counts.append(self._count_pieces(board))

                if (i + 1) % 10 == 0:
                    print(f"  Generated {i+1}/{n_samples_per_density}...")

            # Calculate statistics
            avg_pieces = np.mean(piece_counts)
            avg_density = avg_pieces / 32

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

            # Render and save
            print(f"  Rendering images...")
            for idx, board in enumerate(boards):
                img = self.render_board(board, dimensions)
                matrix = self.board_to_matrix(board)

                filename = f"chess_density_{density_name}_{idx:03d}.png"
                filepath = self.output_dir / density_name / filename
                img.save(filepath)

                # Count pieces by type
                piece_map = board.piece_map()
                white_pieces = sum(
                    1 for p in piece_map.values() if p.color == chess.WHITE
                )
                black_pieces = sum(
                    1 for p in piece_map.values() if p.color == chess.BLACK
                )

                # Save test case metadata
                test_case = {
                    "test_id": f"density_{density_name}_{idx:03d}",
                    "density_level": density_name,
                    "sample_index": idx,
                    "image_file": str(filepath),
                    "ground_truth": matrix,
                    "fen": board.fen(),
                    "statistics": {
                        "total_pieces": len(piece_map),
                        "white_pieces": white_pieces,
                        "black_pieces": black_pieces,
                        "density": len(piece_map) / 32,
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
        print(f"  chess_density_test/")
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

            # Create comparison image
            comparison = Image.new(
                "RGB",
                (self.resolution * 3 + 80, self.resolution + 120),
                (255, 255, 255),
            )

            # Paste boards
            x_positions = [20, self.resolution + 40, self.resolution * 2 + 60]
            for (name, img), x_pos in zip(images.items(), x_positions):
                comparison.paste(img, (x_pos, 100))

            # Add labels
            draw = ImageDraw.Draw(comparison)
            try:
                title_font = ImageFont.truetype("FiraCode-SemiBold.ttf", 32)
                label_font = ImageFont.truetype("FiraCode-SemiBold.ttf", 24)
            except:
                try:
                    title_font = ImageFont.truetype("FiraCode-SemiBold.ttf", 32)
                    label_font = ImageFont.truetype("FiraCode-SemiBold.ttf", 24)
                except:
                    title_font = ImageFont.load_default()
                    label_font = ImageFont.load_default()

            # Title
            draw.text(
                (comparison.width // 2, 30),
                "Chess Density Diagnostic Test",
                fill=(0, 0, 0),
                font=title_font,
                anchor="mm",
            )

            # Density labels
            for (name, board), x_pos in zip(boards.items(), x_positions):
                pieces = self._count_pieces(board)
                density_pct = pieces / 32 * 100

                label_text = f"{name.upper()}\n{pieces} pieces ({density_pct:.0f}%)"
                draw.text(
                    (x_pos + self.resolution // 2, 70),
                    label_text,
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
    print("\n" + "=" * 70)
    print("CHESS DENSITY DIAGNOSTIC TEST GENERATOR")
    print("=" * 70)
    print("\nExperimental Design:")
    print("  Controlled factors:")
    print("    - Game: Chess (8x8 board)")
    print("    - Resolution: 1024x1024px")
    print("    - Generation: Simulated gameplay")
    print()
    print("  Experimental variable:")
    print("    - Density: Low (8-12) vs Medium (16-20) vs High (28-32) pieces")
    print("=" * 70)
    print()

    try:
        generator = ChessDensityDiagnosticTest(
            output_dir="chess_density_test", assets_dir="assets"  # Will be auto-adjusted
        )

        # Generate test suite
        metadata = generator.generate_density_test_suite(n_samples_per_density=30)

        # Generate comparison images
        generator.generate_visual_comparison()

        print("\n" + "=" * 70)
        print("SUCCESS")
        print("=" * 70)

    except ImportError as e:
        print("\n❌ Missing dependency!")
        print(f"Error: {e}")
        print("\nPlease install python-chess:")
        print("  pip install python-chess")
