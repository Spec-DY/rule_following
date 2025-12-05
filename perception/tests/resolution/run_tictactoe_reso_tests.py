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


class TicTacToeResolutionTestRunner:
    """Run VLM tests on Tic-Tac-Toe resolution test suite and analyze results."""

    # Use shared model configurations
    MODEL_CONFIGS = MODEL_CONFIGS

    def __init__(
        self,
        test_dir="tictactoe_resolution_tests",
        output_dir="tictactoe_resolution_tests/results",
        model_key="qwen3-vl-8b",
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

        # System instruction for Tic-Tac-Toe
        self.system_instruction = (
            "Tic Tac Toe is a game played on a 3x3 grid where players take turns "
            "placing X or O in the cells. Given a screenshot of the game board, "
            "please determine the current game state using a 3x3 matrix. In this "
            "matrix, an empty cell should be represented by 0, X should be "
            "represented by 1, and O should be represented by 2. Please strictly "
            "follow the format:\n"
            "Game State: <boardmatrix>\n"
            "where <boardmatrix> is a 3x3 matrix. For example,\n"
            "Game State: [[0, 0, 0], [0, 0, 0], [0, 0, 0]]\n"
            "represents an empty board, and\n"
            "Game State: [[1, 2, 1], [2, 1, 2], [0, 1, 0]]\n"
            "represents a partially filled board."
        )

    def encode_image(self, image_path: str) -> str:
        """Encode image to base64 for API."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def run_single_test(
        self,
        test_file: Path,
        max_tokens: int = 1024,
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
            time.sleep(1)

        except Exception as e:
            return {
                "test_id": test_case["test_id"],
                "group": test_case["group"],
                "resolution": test_case["resolution"],
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
            "group": test_case["group"],
            "resolution": test_case["resolution"],
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
        """Parse 3x3 Tic-Tac-Toe board from model output."""
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

                # Validate dimensions (3x3)
                if isinstance(matrix, list) and len(matrix) == 3:
                    if all(isinstance(row, list) and len(row) == 3 for row in matrix):
                        return matrix
            except:
                pass

        # If parsing fails, return error matrix
        return [[-1] * 3 for _ in range(3)]

    def _calculate_metrics(
        self, predicted: List[List[int]], ground_truth: List[List[int]]
    ) -> Dict:
        """Calculate detailed metrics for 3x3 board."""
        metrics = {
            "overall_accuracy": 0.0,
            "empty_accuracy": 0.0,
            "X_accuracy": 0.0,
            "O_accuracy": 0.0,
            "precision_X": 0.0,
            "precision_O": 0.0,
            "recall_X": 0.0,
            "recall_O": 0.0,
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
        total = 3 * 3
        metrics["overall_accuracy"] = correct / total

        # Per-class accuracy
        for value, name in [(0, "empty"), (1, "X"), (2, "O")]:
            mask = truth_array == value
            if np.sum(mask) > 0:
                correct_class = np.sum(pred_array[mask] == value)
                metrics[f"{name}_accuracy"] = correct_class / np.sum(mask)

        # Precision and Recall for X and O
        for value, name in [(1, "X"), (2, "O")]:
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
        for i in range(3):
            for j in range(3):
                if pred_array[i, j] != truth_array[i, j]:
                    metrics["position_errors"].append(
                        {
                            "row": i,
                            "col": j,
                            "predicted": int(pred_array[i, j]),
                            "truth": int(truth_array[i, j]),
                        }
                    )

        return metrics

    def run_resolution_tests(
        self,
        group: str,
        resolution: int,
        max_tests: int = None,
    ):
        """Run all tests for a specific resolution."""
        test_dir = self.test_dir / group / f"{resolution}x{resolution}"
        test_files = sorted(test_dir.glob("test_*.json"))

        if max_tests:
            test_files = test_files[:max_tests]

        print(f"\n{'='*70}")
        print(f"Testing {group.upper()} - {resolution}×{resolution}")
        print(f"Model: {self.model_name} ({self.model_key})")
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
            / f"{group}_{resolution}_{self.model_key}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(log_file, "w") as f:
            json.dump(log_entries, f, indent=2)

        print(f"\nDetailed log saved to: {log_file}")

        return results

    def run_all_resolutions(
        self,
        max_tests_per_resolution: int = None,
    ):
        """Run tests for all resolutions."""
        print(f"\n{'#'*70}")
        print(f"# TIC-TAC-TOE RESOLUTION TEST SUITE")
        print(f"# Model: {self.model_name} ({self.model_key})")
        print(f"# Tests per resolution: {max_tests_per_resolution or 'All'}")
        print(f"{'#'*70}")

        # Load test metadata
        with open(self.test_dir / "test_metadata.json") as f:
            metadata = json.load(f)

        all_results = {}

        # Test each resolution group
        for group_name, group_info in metadata["resolution_groups"].items():
            all_results[group_name] = {}

            for resolution in group_info["resolutions"]:
                results = self.run_resolution_tests(
                    group_name, resolution, max_tests_per_resolution
                )
                all_results[group_name][resolution] = results

        # Generate comprehensive report
        report = self._generate_comprehensive_report(all_results, metadata)

        # Save report
        report_file = (
            self.output_dir
            / f"tictactoe_resolution_report_{self.model_key}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)

        # Print summary
        self._print_summary(report)

        return report

    def _generate_comprehensive_report(
        self, all_results: Dict, metadata: Dict
    ) -> Dict:
        """Generate comprehensive analysis report."""
        report = {
            "model_key": self.model_key,
            "model_name": self.model_name,
            "timestamp": datetime.now().isoformat(),
            "board_size": "3x3",
            "patch_size": metadata["patch_size"],
            "board_to_image_ratio": metadata["board_to_image_ratio"],
            "groups": {},
        }

        for group_name, resolutions_results in all_results.items():
            report["groups"][group_name] = {}

            for resolution, results in resolutions_results.items():
                # Filter out errors
                valid_results = [r for r in results if "error" not in r]

                if not valid_results:
                    continue

                # Calculate aggregate statistics
                accuracies = [r["accuracy"] for r in valid_results]

                resolution_stats = {
                    "resolution": resolution,
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
                    "X_accuracy",
                    "O_accuracy",
                    "precision_X",
                    "precision_O",
                    "recall_X",
                    "recall_O",
                ]

                for key in metric_keys:
                    values = [r["metrics"][key] for r in valid_results]
                    resolution_stats[f"mean_{key}"] = np.mean(values)

                # Error analysis
                all_errors = []
                for r in valid_results:
                    all_errors.extend(r["metrics"]["position_errors"])

                resolution_stats["total_position_errors"] = len(all_errors)

                # Get dimension info from first valid result
                resolution_stats["dimensions"] = valid_results[0]["dimensions"]

                report["groups"][group_name][str(resolution)] = resolution_stats

        # Cross-group comparison
        report["comparison"] = self._compare_groups(report["groups"])

        return report

    def _compare_groups(self, groups_data: Dict) -> Dict:
        """Compare divisible vs non-divisible groups."""
        comparison = {
            "divisible_vs_non_divisible": {},
            "resolution_ranking": [],
        }

        # Aggregate by group
        group_averages = {}
        for group_name, resolutions in groups_data.items():
            accuracies = [data["mean_accuracy"] for data in resolutions.values()]
            group_averages[group_name] = {
                "mean_accuracy": np.mean(accuracies),
                "std_accuracy": np.std(accuracies),
            }

        comparison["divisible_vs_non_divisible"] = group_averages

        # Ranking across all resolutions
        all_resolutions = []
        for group_name, resolutions in groups_data.items():
            for res, data in resolutions.items():
                all_resolutions.append(
                    {
                        "group": group_name,
                        "resolution": res,
                        "accuracy": data["mean_accuracy"],
                        "is_divisible": data["dimensions"]["is_divisible"],
                    }
                )

        comparison["resolution_ranking"] = sorted(
            all_resolutions, key=lambda x: x["accuracy"], reverse=True
        )

        return comparison

    def _print_summary(self, report: Dict):
        """Print formatted summary of results."""
        print(f"\n{'='*80}")
        print(f"TIC-TAC-TOE RESOLUTION TEST SUMMARY")
        print(f"Model: {report['model_name']} ({report['model_key']})")
        print(f"Board size: {report['board_size']}")
        print(f"Patch size: {report['patch_size']}×{report['patch_size']}")
        print(f"Timestamp: {report['timestamp']}")
        print(f"{'='*80}\n")

        # Results table
        print(
            f"{'Group':<15} {'Resolution':<12} {'Tests':<8} {'Accuracy':<12} "
            f"{'Std Dev':<10} {'X Acc':<10} {'O Acc':<10}"
        )
        print(f"{'-'*80}")

        for group_name in ["divisible", "non_divisible"]:
            if group_name in report["groups"]:
                for res, data in sorted(report["groups"][group_name].items()):
                    print(
                        f"{group_name:<15} "
                        f"{res:<12} "
                        f"{data['n_valid']:<8} "
                        f"{data['mean_accuracy']:<12.1%} "
                        f"{data['std_accuracy']:<10.3f} "
                        f"{data['mean_X_accuracy']:<10.1%} "
                        f"{data['mean_O_accuracy']:<10.1%}"
                    )

        # Group comparison
        if "comparison" in report:
            comp = report["comparison"]

            print(f"\n{'='*80}")
            print(f"GROUP COMPARISON")
            print(f"{'-'*80}")

            if "divisible_vs_non_divisible" in comp:
                for group, stats in comp["divisible_vs_non_divisible"].items():
                    print(
                        f"{group.upper():<15} "
                        f"Mean: {stats['mean_accuracy']:.1%}  "
                        f"Std: {stats['std_accuracy']:.3f}"
                    )

            if "resolution_ranking" in comp:
                print(f"\nResolution Ranking (by accuracy):")
                for i, item in enumerate(comp["resolution_ranking"], 1):
                    divisible_marker = "✓" if item["is_divisible"] else "✗"
                    print(
                        f"  {i}. {item['group']:<15} {item['resolution']:<8} "
                        f"{item['accuracy']:.1%}  [Divisible: {divisible_marker}]"
                    )

        print(f"\n{'='*80}")
        print("\n💡 Compare with 15×15 Gomoku results to understand scale effects")


if __name__ == "__main__":
    print("\n🎮 Available models:")
    for key in MODEL_CONFIGS.keys():
        print(f"  - {key}")
    print()

    # Initialize runner
    runner = TicTacToeResolutionTestRunner(
        test_dir="tictactoe_resolution_tests",
        output_dir="tictactoe_resolution_tests/results",
        model_key="qwen3-vl-8b",  # Change this to test different models
    )

    # Run all resolutions
    report = runner.run_all_resolutions(
        max_tests_per_resolution=None,  # Use None for full run
    )

    print("\n🎯 Analysis complete!")
    print("Check the results folder for:")
    print("  - Detailed logs for each resolution")
    print("  - Comprehensive report")
    print("  - Group comparison (divisible vs non-divisible)")
    print("\n📊 Next: Compare 3×3 vs 15×15 to identify scale threshold")
