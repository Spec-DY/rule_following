import json
import re
import numpy as np
from pathlib import Path
from typing import Dict, List
from openai import OpenAI
import sys
import time
from datetime import datetime
import base64

# Import shared model configurations
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from shared.model_configs import MODEL_CONFIGS


class ChessDensityTestRunner:
    """Run Chess density diagnostic tests with piece-only accuracy metrics."""

    # Use shared model configurations
    MODEL_CONFIGS = MODEL_CONFIGS

    def __init__(
        self,
        test_dir="chess_density_test",
        output_dir="chess_density_test/results",
        model_key="qwen3-vl-8b",
    ):
        self.test_dir = Path(test_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        if model_key not in MODEL_CONFIGS:
            raise ValueError(f"Unknown model: {model_key}")

        config = MODEL_CONFIGS[model_key]
        self.model_key = model_key
        self.model_name = config["model_name"]

        self.client = OpenAI(
            api_key=config["api_key"],
            base_url=config["base_url"],
        )

        self.log_dir = self.output_dir / "logs"
        self.log_dir.mkdir(exist_ok=True)

        # LVLM-Playground format prompt
        self.system_instruction = (
            "Chess is a strategy game played on an 8x8 board with 64 squares, "
            "using six types of pieces: pawns, knights, bishops, rooks, queens, "
            "and kings, for both white and black players. You are provided with "
            "an image of a chessboard, and your task is to represent the current "
            "state of the game as an 8x8 matrix using the specified numerical "
            "format. Each type of chess piece, both black and white, is "
            "represented by a unique number:\n- Empty squares: 0\n"
            "- White pieces: Pawn=1, Knight=2, Bishop=3, Rook=4, Queen=5, King=6\n"
            "- Black pieces: Pawn=-1, Knight=-2, Bishop=-3, Rook=-4, Queen=-5, "
            "King=-6\n\nFrom the provided chessboard image, convert the visible "
            "board into this 8x8 matrix format. For example, the initial chess "
            "position would be represented as:\n"
            "Game State: [[-4, -2, -3, -5, -6, -3, -2, -4],\n"
            "[-1, -1, -1, -1, -1, -1, -1, -1],\n"
            "[0, 0, 0, 0, 0, 0, 0, 0],\n"
            "[0, 0, 0, 0, 0, 0, 0, 0],\n"
            "[0, 0, 0, 0, 0, 0, 0, 0],\n"
            "[0, 0, 0, 0, 0, 0, 0, 0],\n"
            "[1, 1, 1, 1, 1, 1, 1, 1],\n"
            "[4, 2, 3, 5, 6, 3, 2, 4]]\n\n"
            "Ensure that your output strictly follows this matrix format with no "
            "deviations, based on the pieces shown in the image."
        )

    def encode_image(self, image_path: str) -> str:
        """Encode image to base64."""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def run_single_test(self, test_file: Path) -> Dict:
        """Run a single test and return results."""
        with open(test_file) as f:
            test_case = json.load(f)

        image_path = test_case["image_file"]

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": self.system_instruction},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{self.encode_image(image_path)}"
                        },
                    },
                ],
            }
        ]

        start_time = time.time()

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=4096,
                temperature=0.0,
            )

            model_output = response.choices[0].message.content
            api_time = time.time() - start_time
            time.sleep(1.5)

        except Exception as e:
            return {
                "test_id": test_case["test_id"],
                "density_level": test_case["density_level"],
                "error": str(e),
            }

        # Parse and calculate metrics
        predicted = self._parse_output(model_output)
        ground_truth = test_case["ground_truth"]

        # Calculate BOTH metric sets
        standard_metrics = self._calculate_standard_metrics(predicted, ground_truth)
        true_metrics = self._calculate_true_metrics(predicted, ground_truth)

        return {
            "test_id": test_case["test_id"],
            "density_level": test_case["density_level"],
            "sample_index": test_case["sample_index"],
            "statistics": test_case["statistics"],
            # Standard metrics (per-square breakdown)
            "overall_accuracy": standard_metrics["overall_accuracy"],
            "empty_accuracy": standard_metrics["empty_accuracy"],
            "piece_accuracy": standard_metrics["piece_accuracy"],
            # True metrics (excluding empty bias)
            "piece_only_accuracy": true_metrics["piece_only_accuracy"],
            "balanced_accuracy": true_metrics["balanced_accuracy"],
            "piece_precision": true_metrics["piece_precision"],
            "piece_recall": true_metrics["piece_recall"],
            # Detection breakdown
            "exact_detected": true_metrics["exact_detected"],
            "color_errors": true_metrics["color_errors"],
            "type_errors": true_metrics["type_errors"],
            "missed": true_metrics["missed"],
            # Raw data
            "predicted": predicted,
            "ground_truth": ground_truth,
            "raw_output": model_output,
            "api_time": api_time,
        }

    def _parse_output(self, output: str) -> List[List[int]]:
        """Parse 8×8 matrix from model output with robust fallbacks."""
        import ast

        # Strategy 1: Look for "Game State:" pattern (standard)
        pattern1 = r"Game State:\s*(\[\[.*?\]\])"
        match = re.search(pattern1, output, re.DOTALL)

        if match:
            matrix_str = match.group(1)
            # Clean up whitespace but preserve structure
            matrix_str = re.sub(r"\s+", " ", matrix_str)
            try:
                matrix = ast.literal_eval(matrix_str)
                if self._validate_matrix(matrix):
                    return matrix
            except:
                pass

        # Strategy 2: Look for any 8×8 nested list structure
        # Find all [[ ... ]] patterns in the output
        pattern2 = r"\[\s*\[.*?\]\s*(?:,\s*\[.*?\]\s*){7}\]"
        matches = re.findall(pattern2, output, re.DOTALL)

        for match_str in matches:
            try:
                # Clean and parse
                cleaned = re.sub(r"\s+", " ", match_str)
                matrix = ast.literal_eval(cleaned)
                if self._validate_matrix(matrix):
                    return matrix
            except:
                continue

        # Strategy 3: Look for matrix-like structure without strict formatting
        # Extract all numbers and try to reshape
        try:
            # Find all integers (including negatives)
            numbers = re.findall(r"-?\d+", output)
            if len(numbers) == 64:
                # Try to reshape into 8×8
                matrix = []
                for i in range(8):
                    row = [int(numbers[i * 8 + j]) for j in range(8)]
                    matrix.append(row)
                if self._validate_matrix(matrix):
                    return matrix
        except:
            pass

        # All strategies failed
        print(f"\n⚠️  Failed to parse output. Full output:")
        print(output)  # Print full output without truncation
        print(f"\nOutput length: {len(output)} chars")
        return [[-99] * 8 for _ in range(8)]  # Invalid marker

    def _validate_matrix(self, matrix) -> bool:
        """Validate that matrix is a valid 8×8 chess board."""
        if not isinstance(matrix, list):
            return False
        if len(matrix) != 8:
            return False
        for row in matrix:
            if not isinstance(row, list):
                return False
            if len(row) != 8:
                return False
            # Check all values are valid chess piece codes
            for val in row:
                if not isinstance(val, int):
                    return False
                if val not in range(-6, 7):  # -6 to 6, plus 0
                    return False
        return True

    def _calculate_standard_metrics(
        self, predicted: List[List[int]], ground_truth: List[List[int]]
    ) -> Dict:
        """Calculate standard per-square metrics."""
        metrics = {
            "overall_accuracy": 0.0,
            "empty_accuracy": 0.0,
            "piece_accuracy": 0.0,
        }

        if predicted[0][0] == -99:
            return metrics

        pred = np.array(predicted)
        truth = np.array(ground_truth)

        # Overall
        metrics["overall_accuracy"] = float(np.mean(pred == truth))

        # Empty squares
        empty_mask = truth == 0
        if np.sum(empty_mask) > 0:
            metrics["empty_accuracy"] = float(np.mean(pred[empty_mask] == 0))

        # Occupied squares (any piece)
        piece_mask = truth != 0
        if np.sum(piece_mask) > 0:
            metrics["piece_accuracy"] = float(
                np.mean(pred[piece_mask] == truth[piece_mask])
            )

        return metrics

    def _calculate_true_metrics(
        self, predicted: List[List[int]], ground_truth: List[List[int]]
    ) -> Dict:
        """Calculate true capability metrics (piece-focused, with breakdown)."""
        metrics = {
            "piece_only_accuracy": 0.0,
            "balanced_accuracy": 0.0,
            "piece_precision": 0.0,
            "piece_recall": 0.0,
            "exact_detected": 0,
            "color_errors": 0,
            "type_errors": 0,
            "missed": 0,
        }

        if predicted[0][0] == -99:
            return metrics

        pred = np.array(predicted)
        truth = np.array(ground_truth)

        # Piece-Only Accuracy: only look at squares with pieces
        piece_mask = truth != 0
        if np.sum(piece_mask) > 0:
            piece_only_acc = np.mean(pred[piece_mask] == truth[piece_mask])
            metrics["piece_only_accuracy"] = float(piece_only_acc)

            # Detection breakdown
            for i in range(8):
                for j in range(8):
                    if truth[i, j] != 0:  # Ground truth has a piece
                        pred_val = pred[i, j]
                        truth_val = truth[i, j]

                        if pred_val == truth_val:
                            # Exact match (position + piece type + color)
                            metrics["exact_detected"] += 1
                        elif pred_val != 0:
                            # Detected something, but wrong
                            # Check if color is correct (same sign)
                            if (pred_val > 0) == (truth_val > 0):
                                # Correct color, wrong type
                                metrics["type_errors"] += 1
                            else:
                                # Wrong color (implies wrong type too)
                                metrics["color_errors"] += 1
                        else:
                            # Predicted empty (missed the piece)
                            metrics["missed"] += 1

        # Balanced Accuracy: average of empty and piece accuracy
        empty_mask = truth == 0
        class_accs = []

        if np.sum(empty_mask) > 0:
            empty_acc = np.mean(pred[empty_mask] == 0)
            class_accs.append(empty_acc)

        if np.sum(piece_mask) > 0:
            piece_acc = np.mean(pred[piece_mask] == truth[piece_mask])
            class_accs.append(piece_acc)

        if class_accs:
            metrics["balanced_accuracy"] = float(np.mean(class_accs))

        # Piece Detection Precision/Recall (binary: piece vs empty)
        pred_has_piece = pred != 0
        truth_has_piece = truth != 0

        tp = np.sum(pred_has_piece & truth_has_piece)
        fp = np.sum(pred_has_piece & ~truth_has_piece)
        fn = np.sum(~pred_has_piece & truth_has_piece)

        if tp + fp > 0:
            metrics["piece_precision"] = float(tp / (tp + fp))
        if tp + fn > 0:
            metrics["piece_recall"] = float(tp / (tp + fn))

        return metrics

    def run_density_level(self, density_level: str):
        """Run all tests for one density level."""
        test_dir = self.test_dir / density_level
        test_files = sorted(test_dir.glob("test_*.json"))

        print(f"\n{'='*70}")
        print(f"Testing {density_level.upper()} DENSITY")
        print(f"Model: {self.model_name}")
        print(f"Tests: {len(test_files)}")
        print(f"{'='*70}\n")

        results = []

        for i, test_file in enumerate(test_files):
            print(f"[{i+1}/{len(test_files)}] {test_file.name}...", end=" ", flush=True)

            result = self.run_single_test(test_file)
            results.append(result)

            if "error" in result:
                print(f"✗ ERROR: {result['error']}")
            else:
                overall = result["overall_accuracy"]
                piece = result["piece_only_accuracy"]

                # Status based on piece-only accuracy
                status = "✓" if piece > 0.85 else "○" if piece > 0.70 else "✗"
                print(f"{status} Overall: {overall:.1%} | Piece-Only: {piece:.1%}")

        # Save detailed log
        log_file = (
            self.log_dir
            / f"{density_level}_{self.model_key}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(log_file, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\n✓ Log saved: {log_file.name}\n")

        return results

    def run_all_densities(self):
        """Run tests across all density levels."""
        print(f"\n{'#'*70}")
        print(f"# CHESS DENSITY DIAGNOSTIC TEST")
        print(f"# Model: {self.model_name}")
        print(f"{'#'*70}")

        all_results = {}

        for density in ["low", "medium", "high"]:
            results = self.run_density_level(density)
            all_results[density] = results

        # Generate comprehensive report
        report = self._generate_report(all_results)

        # Save report
        report_file = (
            self.output_dir
            / f"density_report_{self.model_key}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)

        # Print summary
        self._print_summary(report)

        return report

    def _generate_report(self, all_results: Dict) -> Dict:
        """Generate comprehensive report with dual metrics."""
        report = {
            "model_key": self.model_key,
            "model_name": self.model_name,
            "timestamp": datetime.now().isoformat(),
            "density_levels": {},
        }

        for density, results in all_results.items():
            valid_results = [r for r in results if "error" not in r]

            if not valid_results:
                continue

            # Standard metrics
            overall_accs = [r["overall_accuracy"] for r in valid_results]
            empty_accs = [r["empty_accuracy"] for r in valid_results]
            piece_accs = [r["piece_accuracy"] for r in valid_results]

            # True metrics
            piece_only_accs = [r["piece_only_accuracy"] for r in valid_results]
            balanced_accs = [r["balanced_accuracy"] for r in valid_results]

            # Detection breakdown
            exact_detected = [r["exact_detected"] for r in valid_results]
            color_errors = [r["color_errors"] for r in valid_results]
            type_errors = [r["type_errors"] for r in valid_results]
            missed = [r["missed"] for r in valid_results]

            # Aggregate statistics
            avg_density = np.mean([r["statistics"]["density"] for r in valid_results])
            avg_pieces = np.mean(
                [r["statistics"]["total_pieces"] for r in valid_results]
            )

            report["density_levels"][density] = {
                "n_tests": len(valid_results),
                "avg_density": float(avg_density),
                "avg_pieces": float(avg_pieces),
                # Standard metrics
                "standard_metrics": {
                    "overall_accuracy": {
                        "mean": float(np.mean(overall_accs)),
                        "std": float(np.std(overall_accs)),
                    },
                    "empty_accuracy": {
                        "mean": float(np.mean(empty_accs)),
                        "std": float(np.std(empty_accs)),
                    },
                    "piece_accuracy": {
                        "mean": float(np.mean(piece_accs)),
                        "std": float(np.std(piece_accs)),
                    },
                },
                # True metrics
                "true_metrics": {
                    "piece_only_accuracy": {
                        "mean": float(np.mean(piece_only_accs)),
                        "std": float(np.std(piece_only_accs)),
                    },
                    "balanced_accuracy": {
                        "mean": float(np.mean(balanced_accs)),
                        "std": float(np.std(balanced_accs)),
                    },
                },
                # Detection breakdown
                "detection_breakdown": {
                    "exact_detected": {
                        "mean": float(np.mean(exact_detected)),
                        "std": float(np.std(exact_detected)),
                    },
                    "color_errors": {
                        "mean": float(np.mean(color_errors)),
                        "std": float(np.std(color_errors)),
                    },
                    "type_errors": {
                        "mean": float(np.mean(type_errors)),
                        "std": float(np.std(type_errors)),
                    },
                    "missed": {
                        "mean": float(np.mean(missed)),
                        "std": float(np.std(missed)),
                    },
                },
            }

        return report

    def _print_summary(self, report: Dict):
        """Print comprehensive summary."""
        print(f"\n{'='*80}")
        print(f"CHESS DENSITY DIAGNOSTIC TEST SUMMARY")
        print(f"Model: {report['model_name']} ({report['model_key']})")
        print(f"{'='*80}\n")

        # Table 1: Standard Metrics
        print("TABLE 1: STANDARD METRICS (Per-Square Accuracy)")
        print(f"{'='*80}")
        print(
            f"{'Density':<12} {'Pieces':<8} {'Overall':<10} "
            f"{'Empty':<10} {'Piece':<10}"
        )
        print(f"{'-'*80}")

        for density, data in report["density_levels"].items():
            std = data["standard_metrics"]
            print(
                f"{density.capitalize():<12} "
                f"{data['avg_pieces']:<8.1f} "
                f"{std['overall_accuracy']['mean']:<10.1%} "
                f"{std['empty_accuracy']['mean']:<10.1%} "
                f"{std['piece_accuracy']['mean']:<10.1%}"
            )

        # Table 2: True Metrics
        print(f"\n{'='*80}")
        print("TABLE 2: TRUE CAPABILITY METRICS (Piece-Only)")
        print(f"{'='*80}")
        print(
            f"{'Density':<12} {'Pieces':<8} {'Overall':<10} "
            f"{'Piece-Only':<12} {'Balanced':<10}"
        )
        print(f"{'-'*80}")

        for density, data in report["density_levels"].items():
            std = data["standard_metrics"]
            true = data["true_metrics"]
            print(
                f"{density.capitalize():<12} "
                f"{data['avg_pieces']:<8.1f} "
                f"{std['overall_accuracy']['mean']:<10.1%} "
                f"{true['piece_only_accuracy']['mean']:<12.1%} "
                f"{true['balanced_accuracy']['mean']:<10.1%}"
            )

        # Table 3: Detection Breakdown
        print(f"\n{'='*80}")
        print("TABLE 3: DETECTION BREAKDOWN (Mean Counts)")
        print(f"{'='*80}")
        print(
            f"{'Density':<12} {'GT Pieces':<10} {'Exact':<10} "
            f"{'Color Err':<10} {'Type Err':<10} {'Missed':<10}"
        )
        print(f"{'-'*80}")

        for density, data in report["density_levels"].items():
            breakdown = data["detection_breakdown"]
            print(
                f"{density.capitalize():<12} "
                f"{data['avg_pieces']:<10.1f} "
                f"{breakdown['exact_detected']['mean']:<10.1f} "
                f"{breakdown['color_errors']['mean']:<10.1f} "
                f"{breakdown['type_errors']['mean']:<10.1f} "
                f"{breakdown['missed']['mean']:<10.1f}"
            )

        # Analysis
        print(f"\n{'='*80}")
        print("DENSITY EFFECT ANALYSIS")
        print(f"{'='*80}")

        densities = list(report["density_levels"].keys())

        if len(densities) >= 2:
            low_data = report["density_levels"][densities[0]]
            high_data = report["density_levels"][densities[-1]]

            low_overall = low_data["standard_metrics"]["overall_accuracy"]["mean"]
            high_overall = high_data["standard_metrics"]["overall_accuracy"]["mean"]

            low_piece = low_data["true_metrics"]["piece_only_accuracy"]["mean"]
            high_piece = high_data["true_metrics"]["piece_only_accuracy"]["mean"]

            print(f"\nComparing {densities[0].upper()} vs {densities[-1].upper()}:")
            print(f"\nOverall Accuracy:")
            print(f"  {densities[0]}: {low_overall:.1%}")
            print(f"  {densities[-1]}: {high_overall:.1%}")
            print(f"  Change: {(high_overall - low_overall):+.1%}")

            print(f"\nPiece-Only Accuracy:")
            print(f"  {densities[0]}: {low_piece:.1%}")
            print(f"  {densities[-1]}: {high_piece:.1%}")
            print(f"  Change: {(high_piece - low_piece):+.1%}")

            print(f"\n{'='*80}")
            print("INTERPRETATION:")
            print(f"{'='*80}")

            if (high_overall - low_overall) < -0.05 and (high_piece - low_piece) > 0.05:
                print("⚠️  DENSITY PARADOX DETECTED!")
                print("   Overall accuracy decreases with density")
                print("   BUT piece detection improves with density")
                print("   → Evidence of empty-square bias / shortcut learning")
            elif abs(high_piece - low_piece) < 0.05:
                print("✓  DENSITY-INVARIANT PERFORMANCE")
                print("   Piece detection stable across densities")
                print("   → Evidence of genuine visual understanding")
            else:
                print("→  Mixed pattern, requires further investigation")

        print(f"{'='*80}\n")


if __name__ == "__main__":
    print("\n♟️  Available models:")
    for key in MODEL_CONFIGS.keys():
        print(f"  - {key}")
    print()

    # Run test
    runner = ChessDensityTestRunner(
        test_dir="chess_density_test",
        output_dir="chess_density_test/results",
        model_key="qwen3-vl-30b",  # Change this to test different models
    )

    report = runner.run_all_densities()

    print("\n🎯 Test complete!")
    print("Check results/ for:")
    print("  - Detailed logs per density level")
    print("  - Comprehensive report with dual metrics")
    print("  - Detection breakdown analysis")
