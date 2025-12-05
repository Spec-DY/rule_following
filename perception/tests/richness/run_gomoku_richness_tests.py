import json
import re
import numpy as np
from pathlib import Path
from typing import Dict, List
from openai import OpenAI
from collections import defaultdict
import os
import sys
import time
from datetime import datetime
import base64

# Import shared model configurations
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from shared.model_configs import MODEL_CONFIGS


class GomokuVisualRichnessTestRunner:
    """Run VLM tests on Gomoku Visual Richness test suite and analyze results."""

    # Use shared model configurations
    MODEL_CONFIGS = MODEL_CONFIGS

    def __init__(
        self,
        test_dir="gomoku_visual_richness_tests",
        output_dir="gomoku_visual_richness_tests/results",
        model_key="glm4v-thinking",  # Default model
    ):
        self.test_dir = Path(test_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Get model configuration
        if model_key not in MODEL_CONFIGS:
            raise ValueError(
                f"Unknown model key: {model_key}. "
                f"Available: {list(MODEL_CONFIGS.keys())}"
            )

        config = MODEL_CONFIGS[model_key]
        self.model_key = model_key
        self.model_name = config["model_name"]

        # Initialize API client
        self.client = OpenAI(
            api_key=config["api_key"],
            base_url=config["base_url"],
        )

        # Create log directory
        self.log_dir = self.output_dir / "logs"
        self.log_dir.mkdir(exist_ok=True)

        # System instruction for 15×15 Gomoku (following LVLM-Playground format)
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
        """Encode image to base64 for API."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def run_single_test(
        self,
        test_file: Path,
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> Dict:
        """Run a single test and return detailed results."""
        with open(test_file) as f:
            test_case = json.load(f)

        # Get image path
        image_path = test_case["image_file"]

        # Prepare messages with image
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

        # Record start time
        start_time = time.time()

        try:
            # Call API
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )

            model_output = response.choices[0].message.content
            api_time = time.time() - start_time
            time.sleep(1.5)  # Rate limiting

        except Exception as e:
            return {
                "test_id": test_case["test_id"],
                "style": test_case["style"],
                "error": str(e),
                "accuracy": 0.0,
            }

        # Parse model output
        predicted = self._parse_model_output(model_output)
        ground_truth = test_case["ground_truth"]

        # Calculate metrics
        metrics = self._calculate_metrics(predicted, ground_truth)

        return {
            "test_id": test_case["test_id"],
            "style": test_case["style"],
            "sample_index": test_case["sample_index"],
            "dimensions": test_case["dimensions"],
            "accuracy": metrics["overall_accuracy"],
            "metrics": metrics,
            "predicted": predicted,
            "ground_truth": ground_truth,
            "raw_output": model_output,
            "api_time": api_time,
            "statistics": test_case["statistics"],
        }

    def _parse_model_output(self, output: str) -> List[List[int]]:
        """Parse 15×15 Gomoku board from model output."""
        # Look for "Game State:" followed by matrix
        pattern = r"Game State:\s*(\[\[.*?\]\])"
        match = re.search(pattern, output, re.DOTALL)

        if match:
            matrix_str = match.group(1)
            # Clean up the string
            matrix_str = matrix_str.replace("\n", "").replace(" ", "")
            try:
                import ast

                matrix = ast.literal_eval(matrix_str)

                # Validate dimensions (15×15)
                if isinstance(matrix, list) and len(matrix) == 15:
                    if all(isinstance(row, list) and len(row) == 15 for row in matrix):
                        return matrix
            except:
                pass

        # If parsing fails, return error matrix
        return [[-1] * 15 for _ in range(15)]

    def _calculate_metrics(
        self, predicted: List[List[int]], ground_truth: List[List[int]]
    ) -> Dict:
        """Calculate detailed metrics for 15×15 board."""
        metrics = {
            "overall_accuracy": 0.0,
            "empty_accuracy": 0.0,
            "black_accuracy": 0.0,
            "white_accuracy": 0.0,
            "precision_black": 0.0,
            "precision_white": 0.0,
            "recall_black": 0.0,
            "recall_white": 0.0,
            "position_errors": [],
        }

        # Check if parsing failed
        if predicted[0][0] == -1:
            return metrics

        # Convert to numpy arrays
        pred_array = np.array(predicted)
        truth_array = np.array(ground_truth)

        # Overall accuracy
        correct = np.sum(pred_array == truth_array)
        total = 15 * 15
        metrics["overall_accuracy"] = correct / total

        # Per-class accuracy
        for value, name in [(0, "empty"), (1, "black"), (2, "white")]:
            mask = truth_array == value
            if np.sum(mask) > 0:
                correct_class = np.sum(pred_array[mask] == value)
                metrics[f"{name}_accuracy"] = correct_class / np.sum(mask)

        # Precision and Recall for black and white stones
        for value, name in [(1, "black"), (2, "white")]:
            pred_mask = pred_array == value
            truth_mask = truth_array == value

            tp = np.sum(pred_mask & truth_mask)
            fp = np.sum(pred_mask & ~truth_mask)
            fn = np.sum(~pred_mask & truth_mask)

            if tp + fp > 0:
                metrics[f"precision_{name}"] = tp / (tp + fp)
            if tp + fn > 0:
                metrics[f"recall_{name}"] = tp / (tp + fn)

        # Collect position errors
        for i in range(15):
            for j in range(15):
                if pred_array[i, j] != truth_array[i, j]:
                    metrics["position_errors"].append(
                        {
                            "row": chr(ord("A") + i),
                            "col": j,
                            "predicted": int(pred_array[i, j]),
                            "truth": int(truth_array[i, j]),
                        }
                    )

        return metrics

    def run_style_tests(
        self,
        style: str,
        max_tests: int = None,
    ):
        """Run all tests for a specific style."""
        test_dir = self.test_dir / style
        test_files = sorted(test_dir.glob("test_*.json"))

        if max_tests:
            test_files = test_files[:max_tests]

        print(f"\n{'='*70}")
        print(f"Testing {style.upper().replace('_', ' ')}")
        print(f"Model: {self.model_name}")
        print(f"Number of tests: {len(test_files)}")
        print(f"{'='*70}\n")

        results = []
        log_entries = []

        for i, test_file in enumerate(test_files):
            print(f"[{i+1}/{len(test_files)}] {test_file.name}...", end=" ", flush=True)

            result = self.run_single_test(test_file)
            results.append(result)

            # Create detailed log entry
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "test_file": str(test_file),
                **result,
            }
            log_entries.append(log_entry)

            if "error" in result:
                print(f"✗ ERROR: {result['error']}")
            else:
                acc = result["accuracy"]
                status = "✓" if acc > 0.95 else "○" if acc > 0.85 else "✗"
                print(f"{status} Accuracy: {acc:.1%}")

        # Save detailed logs
        log_file = (
            self.log_dir
            / f"{style}_{self.model_key}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(log_file, "w") as f:
            json.dump(log_entries, f, indent=2)

        print(f"\nDetailed log saved to: {log_file}")

        return results

    def run_all_styles(self, max_tests_per_style: int = None):
        """Run tests for all styles (2D vs 3D)."""
        print(f"\n{'#'*70}")
        print(f"# GOMOKU VISUAL RICHNESS TEST SUITE")
        print(f"# Model: {self.model_name}")
        print(f"# Tests per style: {max_tests_per_style or 'All'}")
        print(f"{'#'*70}")

        # Load test metadata
        with open(self.test_dir / "test_metadata.json") as f:
            metadata = json.load(f)

        all_results = {}

        # Test each style
        for style_name in ["2d_flat", "3d_rendered"]:
            results = self.run_style_tests(style_name, max_tests_per_style)
            all_results[style_name] = results

        # Generate comprehensive report
        report = self._generate_comprehensive_report(all_results, metadata)

        # Save report
        report_file = (
            self.output_dir
            / f"visual_richness_report_{self.model_key}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)

        # Print summary
        self._print_summary(report)

        return report

    def _generate_comprehensive_report(self, all_results: Dict, metadata: Dict) -> Dict:
        """Generate comprehensive analysis report."""
        report = {
            "model_key": self.model_key,
            "model_name": self.model_name,
            "timestamp": datetime.now().isoformat(),
            "board_size": metadata["board_size"],
            "resolution": metadata["resolution"],
            "board_to_image_ratio": metadata["board_to_image_ratio"],
            "density_statistics": metadata["density_statistics"],
            "styles": {},
        }

        for style_name, results in all_results.items():
            # Filter out errors
            valid_results = [r for r in results if "error" not in r]

            if not valid_results:
                continue

            # Calculate aggregate statistics
            accuracies = [r["accuracy"] for r in valid_results]

            style_stats = {
                "style": style_name,
                "n_tests": len(results),
                "n_valid": len(valid_results),
                "n_errors": len(results) - len(valid_results),
                "mean_accuracy": np.mean(accuracies),
                "std_accuracy": np.std(accuracies),
                "min_accuracy": np.min(accuracies),
                "max_accuracy": np.max(accuracies),
                "median_accuracy": np.median(accuracies),
            }

            # Aggregate metrics
            metric_keys = [
                "empty_accuracy",
                "black_accuracy",
                "white_accuracy",
                "precision_black",
                "precision_white",
                "recall_black",
                "recall_white",
            ]

            for key in metric_keys:
                values = [r["metrics"][key] for r in valid_results]
                style_stats[f"mean_{key}"] = np.mean(values)
                style_stats[f"std_{key}"] = np.std(values)

            # Error analysis
            all_errors = []
            for r in valid_results:
                all_errors.extend(r["metrics"]["position_errors"])

            style_stats["total_position_errors"] = len(all_errors)

            # Average API time
            style_stats["mean_api_time"] = np.mean(
                [r["api_time"] for r in valid_results]
            )

            report["styles"][style_name] = style_stats

        # Comparison: 2D vs 3D
        report["comparison"] = self._compare_styles(report["styles"])

        return report

    def _compare_styles(self, styles_data: Dict) -> Dict:
        """Compare 2D flat vs 3D rendered styles."""
        comparison = {
            "accuracy_difference": 0.0,
            "winner": None,
        }

        if "2d_flat" in styles_data and "3d_rendered" in styles_data:
            acc_2d = styles_data["2d_flat"]["mean_accuracy"]
            acc_3d = styles_data["3d_rendered"]["mean_accuracy"]

            diff = acc_3d - acc_2d
            comparison["accuracy_difference"] = diff

            if abs(diff) < 0.02:  # Less than 2% difference
                comparison["winner"] = "No significant difference"
            elif diff > 0:
                comparison["winner"] = "3D rendered (better)"
            else:
                comparison["winner"] = "2D flat (better)"

            # Per-cell-type comparison
            comparison["per_cell_comparison"] = {}
            for cell_type in ["empty", "black", "white"]:
                key = f"mean_{cell_type}_accuracy"
                comparison["per_cell_comparison"][cell_type] = {
                    "2d_flat": styles_data["2d_flat"][key],
                    "3d_rendered": styles_data["3d_rendered"][key],
                    "difference": styles_data["3d_rendered"][key]
                    - styles_data["2d_flat"][key],
                }

        return comparison

    def _print_summary(self, report: Dict):
        """Print formatted summary of results."""
        print(f"\n{'='*80}")
        print(f"GOMOKU VISUAL RICHNESS TEST SUMMARY")
        print(f"Model: {report['model_name']} ({report['model_key']})")
        print(f"Board size: {report['board_size']}×{report['board_size']}")
        print(f"Resolution: {report['resolution']}×{report['resolution']}")
        print(f"Average density: {report['density_statistics']['average_density']:.1%}")
        print(f"Timestamp: {report['timestamp']}")
        print(f"{'='*80}\n")

        # Results table
        print(
            f"{'Style':<20} {'Tests':<8} {'Accuracy':<12} {'Std Dev':<10} "
            f"{'Empty':<10} {'Black':<10} {'White':<10}"
        )
        print(f"{'-'*80}")

        for style_name, data in report["styles"].items():
            style_display = style_name.replace("_", " ").title()
            print(
                f"{style_display:<20} "
                f"{data['n_valid']:<8} "
                f"{data['mean_accuracy']:<12.1%} "
                f"{data['std_accuracy']:<10.3f} "
                f"{data['mean_empty_accuracy']:<10.1%} "
                f"{data['mean_black_accuracy']:<10.1%} "
                f"{data['mean_white_accuracy']:<10.1%}"
            )

        # Comparison
        if "comparison" in report:
            comp = report["comparison"]

            print(f"\n{'='*80}")
            print(f"2D FLAT vs 3D RENDERED COMPARISON")
            print(f"{'-'*80}")
            print(f"Winner: {comp['winner']}")
            print(f"Accuracy Difference: {comp['accuracy_difference']:+.1%} (3D - 2D)")

            if "per_cell_comparison" in comp:
                print(f"\nPer-Cell-Type Comparison:")
                for cell_type, data in comp["per_cell_comparison"].items():
                    print(
                        f"  {cell_type.capitalize():<8}: "
                        f"2D={data['2d_flat']:.1%}  "
                        f"3D={data['3d_rendered']:.1%}  "
                        f"Δ={data['difference']:+.1%}"
                    )

        print(f"\n{'='*80}")
        print(
            "\n💡 Key Question: Does visual richness (3D rendering) help or hinder VLM perception?"
        )
        print("   - If 3D > 2D: Models benefit from redundant visual cues")
        print("   - If 2D > 3D: Visual complexity introduces noise")
        print(
            "   - If no difference: Visual style is orthogonal to perception failures"
        )


if __name__ == "__main__":
    print("\n🎮 Available models:")
    for key in MODEL_CONFIGS.keys():
        print(f"  - {key}")
    print()

    # Initialize runner
    runner = GomokuVisualRichnessTestRunner(
        test_dir="gomoku_visual_richness_tests",
        output_dir="gomoku_visual_richness_tests/results",
        model_key="gemma3",  # Change this to test different models
    )

    # Run all styles
    report = runner.run_all_styles(
        max_tests_per_style=None,  # Use None for full run
    )

    print("\n🎯 Analysis complete!")
    print("Check the results folder for:")
    print("  - Detailed logs for each style")
    print("  - Comprehensive report with 2D vs 3D comparison")
    print("  - Per-cell-type accuracy breakdown")

