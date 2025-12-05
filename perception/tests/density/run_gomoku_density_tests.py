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


class GomokuDensityTestRunner:
    """Run density diagnostic tests with dual accuracy metrics."""

    # Use shared model configurations
    MODEL_CONFIGS = MODEL_CONFIGS

    def __init__(
        self,
        test_dir="gomoku_density_test",
        output_dir="gomoku_density_test/results",
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
            "Gomoku is a game played on a 15×15 grid where players take turns "
            "placing black or white stones on the intersections. Given a "
            "screenshot of the Gomoku board, please determine the current game "
            "state using a 15×15 matrix. In this matrix, an empty intersection "
            "should be represented by 0, a black stone by 1, and a white stone by "
            "2. Please strictly follow the format:\n"
            "Game State: <boardmatrix>\n"
            "where <boardmatrix> is a 15×15 matrix. For example,\n"
            "Game State: [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], "
            "[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], "
            "[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], "
            "[0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0], "
            "[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], "
            "[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], "
            "[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], "
            "[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], "
            "[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], "
            "[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], "
            "[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], "
            "[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], "
            "[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], "
            "[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], "
            "[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]\n"
            "represents a partially filled Gomoku board."
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
            # Standard metrics (per-class breakdown)
            "overall_accuracy": standard_metrics["overall_accuracy"],
            "empty_accuracy": standard_metrics["empty_accuracy"],
            "black_accuracy": standard_metrics["black_accuracy"],
            "white_accuracy": standard_metrics["white_accuracy"],
            # True metrics (excluding empty bias)
            "stone_only_accuracy": true_metrics["stone_only_accuracy"],
            "balanced_accuracy": true_metrics["balanced_accuracy"],
            "stone_precision": true_metrics["stone_precision"],
            "stone_recall": true_metrics["stone_recall"],
            # Raw data
            "predicted": predicted,
            "ground_truth": ground_truth,
            "raw_output": model_output,
            "api_time": api_time,
        }

    def _parse_output(self, output: str) -> List[List[int]]:
        """Parse 15×15 matrix from model output."""
        pattern = r"Game State:\s*(\[\[.*?\]\])"
        match = re.search(pattern, output, re.DOTALL)

        if match:
            matrix_str = match.group(1).replace("\n", "").replace(" ", "")
            try:
                import ast

                matrix = ast.literal_eval(matrix_str)
                if (
                    isinstance(matrix, list)
                    and len(matrix) == 15
                    and all(isinstance(row, list) and len(row) == 15 for row in matrix)
                ):
                    return matrix
            except:
                pass

        return [[-1] * 15 for _ in range(15)]

    def _calculate_standard_metrics(
        self, predicted: List[List[int]], ground_truth: List[List[int]]
    ) -> Dict:
        """Calculate standard per-class metrics."""
        metrics = {
            "overall_accuracy": 0.0,
            "empty_accuracy": 0.0,
            "black_accuracy": 0.0,
            "white_accuracy": 0.0,
        }

        if predicted[0][0] == -1:
            return metrics

        pred = np.array(predicted)
        truth = np.array(ground_truth)

        # Overall
        metrics["overall_accuracy"] = float(np.mean(pred == truth))

        # Per-class
        for value, name in [(0, "empty"), (1, "black"), (2, "white")]:
            mask = truth == value
            if np.sum(mask) > 0:
                acc = np.mean(pred[mask] == value)
                metrics[f"{name}_accuracy"] = float(acc)

        return metrics

    def _calculate_true_metrics(
        self, predicted: List[List[int]], ground_truth: List[List[int]]
    ) -> Dict:
        """Calculate true capability metrics (excluding empty bias)."""
        metrics = {
            "stone_only_accuracy": 0.0,
            "balanced_accuracy": 0.0,
            "stone_precision": 0.0,
            "stone_recall": 0.0,
        }

        if predicted[0][0] == -1:
            return metrics

        pred = np.array(predicted)
        truth = np.array(ground_truth)

        # Stone-Only Accuracy: only look at positions with stones
        stone_mask = truth > 0
        if np.sum(stone_mask) > 0:
            stone_only_acc = np.mean(pred[stone_mask] == truth[stone_mask])
            metrics["stone_only_accuracy"] = float(stone_only_acc)

        # Balanced Accuracy: simple average of three classes
        class_accs = []
        for value in [0, 1, 2]:
            mask = truth == value
            if np.sum(mask) > 0:
                acc = np.mean(pred[mask] == value)
                class_accs.append(acc)

        if class_accs:
            metrics["balanced_accuracy"] = float(np.mean(class_accs))

        # Stone Detection Precision/Recall (binary: stone vs empty)
        pred_has_stone = pred > 0
        truth_has_stone = truth > 0

        tp = np.sum(pred_has_stone & truth_has_stone)
        fp = np.sum(pred_has_stone & ~truth_has_stone)
        fn = np.sum(~pred_has_stone & truth_has_stone)

        if tp + fp > 0:
            metrics["stone_precision"] = float(tp / (tp + fp))
        if tp + fn > 0:
            metrics["stone_recall"] = float(tp / (tp + fn))

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
                stone = result["stone_only_accuracy"]

                # Status based on stone-only accuracy
                status = "✓" if stone > 0.85 else "○" if stone > 0.70 else "✗"
                print(f"{status} Overall: {overall:.1%} | Stone-Only: {stone:.1%}")

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
        print(f"# GOMOKU DENSITY DIAGNOSTIC TEST")
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

            # Standard metrics (per-class)
            overall_accs = [r["overall_accuracy"] for r in valid_results]
            empty_accs = [r["empty_accuracy"] for r in valid_results]
            black_accs = [r["black_accuracy"] for r in valid_results]
            white_accs = [r["white_accuracy"] for r in valid_results]

            # True metrics (stone-focused)
            stone_only_accs = [r["stone_only_accuracy"] for r in valid_results]
            balanced_accs = [r["balanced_accuracy"] for r in valid_results]

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
                    "black_accuracy": {
                        "mean": float(np.mean(black_accs)),
                        "std": float(np.std(black_accs)),
                    },
                    "white_accuracy": {
                        "mean": float(np.mean(white_accs)),
                        "std": float(np.std(white_accs)),
                    },
                },
                # True metrics
                "true_metrics": {
                    "stone_only_accuracy": {
                        "mean": float(np.mean(stone_only_accs)),
                        "std": float(np.std(stone_only_accs)),
                    },
                    "balanced_accuracy": {
                        "mean": float(np.mean(balanced_accs)),
                        "std": float(np.std(balanced_accs)),
                    },
                },
            }

        return report

    def _print_summary(self, report: Dict):
        """Print comprehensive summary with both metric sets."""
        print(f"\n{'='*80}")
        print(f"DENSITY DIAGNOSTIC TEST SUMMARY")
        print(f"Model: {report['model_name']} ({report['model_key']})")
        print(f"{'='*80}\n")

        # Table 1: Standard Metrics (Per-Class Breakdown)
        print("TABLE 1: STANDARD METRICS (Per-Class Accuracy)")
        print(f"{'='*80}")
        print(
            f"{'Density':<12} {'Pieces':<8} {'Overall':<10} "
            f"{'Empty':<10} {'Black':<10} {'White':<10}"
        )
        print(f"{'-'*80}")

        for density, data in report["density_levels"].items():
            std = data["standard_metrics"]
            print(
                f"{density.capitalize():<12} "
                f"{data['avg_pieces']:<8.1f} "
                f"{std['overall_accuracy']['mean']:<10.1%} "
                f"{std['empty_accuracy']['mean']:<10.1%} "
                f"{std['black_accuracy']['mean']:<10.1%} "
                f"{std['white_accuracy']['mean']:<10.1%}"
            )

        # Table 2: True Metrics (Stone-Focused)
        print(f"\n{'='*80}")
        print("TABLE 2: TRUE CAPABILITY METRICS (Stone-Only)")
        print(f"{'='*80}")
        print(
            f"{'Density':<12} {'Pieces':<8} {'Overall':<10} "
            f"{'Stone-Only':<12} {'Balanced':<10}"
        )
        print(f"{'-'*80}")

        for density, data in report["density_levels"].items():
            std = data["standard_metrics"]
            true = data["true_metrics"]
            print(
                f"{density.capitalize():<12} "
                f"{data['avg_pieces']:<8.1f} "
                f"{std['overall_accuracy']['mean']:<10.1%} "
                f"{true['stone_only_accuracy']['mean']:<12.1%} "
                f"{true['balanced_accuracy']['mean']:<10.1%}"
            )

        # Analysis
        print(f"\n{'='*80}")
        print("DENSITY EFFECT ANALYSIS")
        print(f"{'='*80}")

        densities = list(report["density_levels"].keys())

        if len(densities) >= 2:
            # Compare low vs high (or available pairs)
            low_data = report["density_levels"][densities[0]]
            high_data = report["density_levels"][densities[-1]]

            low_overall = low_data["standard_metrics"]["overall_accuracy"]["mean"]
            high_overall = high_data["standard_metrics"]["overall_accuracy"]["mean"]

            low_stone = low_data["true_metrics"]["stone_only_accuracy"]["mean"]
            high_stone = high_data["true_metrics"]["stone_only_accuracy"]["mean"]

            print(f"\nComparing {densities[0].upper()} vs {densities[-1].upper()}:")
            print(f"\nOverall Accuracy (Standard):")
            print(f"  {densities[0]}: {low_overall:.1%}")
            print(f"  {densities[-1]}: {high_overall:.1%}")
            print(f"  Change: {(high_overall - low_overall):+.1%}")

            print(f"\nStone-Only Accuracy (True Capability):")
            print(f"  {densities[0]}: {low_stone:.1%}")
            print(f"  {densities[-1]}: {high_stone:.1%}")
            print(f"  Change: {(high_stone - low_stone):+.1%}")

            print(f"\n{'='*80}")
            print("INTERPRETATION:")
            print(f"{'='*80}")

            if (high_overall - low_overall) < -5 and (high_stone - low_stone) > 5:
                print("⚠️  PARADOX DETECTED!")
                print("   Overall accuracy decreases with density")
                print("   BUT stone detection improves with density")
                print("   → Evidence of empty-cell bias / shortcut learning")
            elif abs(high_stone - low_stone) < 5:
                print("✓  DENSITY-INVARIANT PERFORMANCE")
                print("   Stone detection stable across densities")
                print("   → Evidence of genuine visual understanding")
            else:
                print("→  Mixed pattern, requires further investigation")

        print(f"{'='*80}\n")


if __name__ == "__main__":
    print("\n🎮 Available models:")
    for key in MODEL_CONFIGS.keys():
        print(f"  - {key}")
    print()

    # Run test
    runner = GomokuDensityTestRunner(
        test_dir="gomoku_density_test",
        output_dir="gomoku_density_test/results",
        model_key="qwen3-vl-8b",  # Change this to test different models
    )

    report = runner.run_all_densities()

    print("\n🎯 Test complete!")
    print("Check results/ for:")
    print("  - Detailed logs per density level")
    print("  - Comprehensive report with dual metrics")
    print("  - Paradox detection analysis")
