import json
import re
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from openai import OpenAI
from collections import Counter, defaultdict
import os
import sys
import time
from datetime import datetime
import base64

# Import shared model configurations
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from shared.model_configs import MODEL_CONFIGS


class GomokuPatchTestRunner:
    # Use shared model configurations
    MODEL_CONFIGS = MODEL_CONFIGS

    def __init__(
        self,
        test_dir="gomoku_patch_tests",
        output_dir="gomoku_patch_tests/results",
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

    def encode_image(self, image_path: str) -> str:
        """Encode image to base64 for API."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def run_single_test(
        self,
        test_file: Path,
        max_tokens: int = 2048,
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
                    {"type": "text", "text": test_case["prompt"]},
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

        except Exception as e:
            return {
                "test_id": test_case["test_id"],
                "condition": test_case["condition"],
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
            "condition": test_case["condition"],
            "sample_index": test_case["sample_index"],
            "accuracy": metrics["overall_accuracy"],
            "metrics": metrics,
            "predicted": predicted,
            "ground_truth": ground_truth,
            "raw_output": model_output,
            "api_time": api_time,
            "statistics": test_case["statistics"],
        }

    def _parse_model_output(self, output: str) -> List[List[int]]:
        """Parse 15x15 Gomoku board from model output."""
        # Look for "Game State:" followed by matrix
        pattern = r"Game State:\s*(\[\[.*?\]\])"
        match = re.search(pattern, output, re.DOTALL)

        if match:
            matrix_str = match.group(1)
            # Clean up the string
            matrix_str = matrix_str.replace("\n", "").replace(" ", "")
            try:
                # Use ast.literal_eval for safer evaluation
                import ast

                matrix = ast.literal_eval(matrix_str)

                # Validate dimensions
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
        """Calculate detailed metrics for 15x15 board."""
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

        # Precision and Recall for stones
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
                            "row": i,
                            "col": j,
                            "predicted": int(pred_array[i, j]),
                            "truth": int(truth_array[i, j]),
                        }
                    )

        return metrics

    def run_condition_tests(
        self,
        condition: str,
        max_tests: int = None,
    ):
        """Run all tests for a specific alignment condition."""
        test_files = sorted((self.test_dir / condition).glob("test_*.json"))

        if max_tests:
            test_files = test_files[:max_tests]

        print(f"\n{'='*60}")
        print(f"Testing {condition.upper()} condition")
        print(f"Model: {self.model_name} ({self.model_key})")
        print(f"Number of tests: {len(test_files)}")
        print(f"{'='*60}\n")

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
            / f"{condition}_{self.model_key}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(log_file, "w") as f:
            json.dump(log_entries, f, indent=2)

        print(f"\nDetailed log saved to: {log_file}")

        return results

    def run_all_conditions(
        self,
        max_tests_per_condition: int = None,
    ):
        """Run tests for all alignment conditions."""
        conditions = [
            "aligned",
            "offset_quarter",
            "offset_half",
            "offset_three_quarter",
        ]
        all_results = {}

        print(f"\n{'#'*60}")
        print(f"# GOMOKU PATCH ALIGNMENT TEST SUITE")
        print(f"# Model: {self.model_name} ({self.model_key})")
        print(f"# Tests per condition: {max_tests_per_condition or 'All'}")
        print(f"{'#'*60}")

        for condition in conditions:
            results = self.run_condition_tests(
                condition, max_tests_per_condition
            )
            all_results[condition] = results

        # Generate comprehensive report
        report = self._generate_comprehensive_report(all_results)

        # Save report
        report_file = (
            self.output_dir
            / f"patch_alignment_report_{self.model_key}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)

        # Print summary
        self._print_summary(report)

        return report

    def _generate_comprehensive_report(self, all_results: Dict) -> Dict:
        """Generate comprehensive analysis report."""
        report = {
            "model_key": self.model_key,
            "model_name": self.model_name,
            "timestamp": datetime.now().isoformat(),
            "conditions": {},
        }

        for condition, results in all_results.items():
            # Filter out errors
            valid_results = [r for r in results if "error" not in r]

            if not valid_results:
                continue

            # Calculate aggregate statistics
            accuracies = [r["accuracy"] for r in valid_results]

            condition_stats = {
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
                condition_stats[f"mean_{key}"] = np.mean(values)

            # Analyze error patterns
            all_errors = []
            for r in valid_results:
                all_errors.extend(r["metrics"]["position_errors"])

            # Error heatmap (which positions have most errors)
            error_heatmap = np.zeros((15, 15))
            for error in all_errors:
                error_heatmap[error["row"], error["col"]] += 1

            condition_stats["total_position_errors"] = len(all_errors)
            condition_stats["error_heatmap"] = error_heatmap.tolist()

            # Find most error-prone positions
            if all_errors:
                error_positions = defaultdict(int)
                for error in all_errors:
                    error_positions[(error["row"], error["col"])] += 1

                top_errors = sorted(
                    error_positions.items(), key=lambda x: x[1], reverse=True
                )[:5]
                condition_stats["top_error_positions"] = [
                    {"position": list(pos), "count": count} for pos, count in top_errors
                ]

            report["conditions"][condition] = condition_stats

        # Cross-condition comparison
        report["comparison"] = self._compare_conditions(report["conditions"])

        return report

    def _compare_conditions(self, conditions_data: Dict) -> Dict:
        """Compare performance across alignment conditions."""
        comparison = {}

        if len(conditions_data) < 2:
            return comparison

        # Find best and worst conditions
        accuracies = {
            cond: data["mean_accuracy"] for cond, data in conditions_data.items()
        }

        best_condition = max(accuracies, key=accuracies.get)
        worst_condition = min(accuracies, key=accuracies.get)

        comparison["best_condition"] = {
            "name": best_condition,
            "accuracy": accuracies[best_condition],
        }

        comparison["worst_condition"] = {
            "name": worst_condition,
            "accuracy": accuracies[worst_condition],
        }

        comparison["accuracy_drop"] = (
            accuracies[best_condition] - accuracies[worst_condition]
        )

        # Statistical significance (simplified - you might want to use scipy.stats)
        comparison["accuracy_ranking"] = sorted(
            accuracies.items(), key=lambda x: x[1], reverse=True
        )

        return comparison

    def _print_summary(self, report: Dict):
        """Print formatted summary of results."""
        print(f"\n{'='*80}")
        print(f"PATCH ALIGNMENT TEST SUMMARY")
        print(f"Model: {report['model_name']} ({report['model_key']})")
        print(f"Timestamp: {report['timestamp']}")
        print(f"{'='*80}\n")

        # Condition results table
        print(
            f"{'Condition':<20} {'Tests':<8} {'Accuracy':<12} {'Std Dev':<10} {'Black Acc':<12} {'White Acc':<12}"
        )
        print(f"{'-'*80}")

        for condition in [
            "aligned",
            "offset_quarter",
            "offset_half",
            "offset_three_quarter",
        ]:
            if condition in report["conditions"]:
                data = report["conditions"][condition]
                print(
                    f"{condition:<20} "
                    f"{data['n_valid']:<8} "
                    f"{data['mean_accuracy']:<12.1%} "
                    f"{data['std_accuracy']:<10.3f} "
                    f"{data['mean_black_accuracy']:<12.1%} "
                    f"{data['mean_white_accuracy']:<12.1%}"
                )

        # Comparison results
        if "comparison" in report and report["comparison"]:
            comp = report["comparison"]
            print(f"\n{'='*80}")
            print(f"COMPARISON ANALYSIS")
            print(f"{'-'*80}")
            print(
                f"Best condition:  {comp['best_condition']['name']} "
                f"(accuracy: {comp['best_condition']['accuracy']:.1%})"
            )
            print(
                f"Worst condition: {comp['worst_condition']['name']} "
                f"(accuracy: {comp['worst_condition']['accuracy']:.1%})"
            )
            print(f"Performance drop: {comp['accuracy_drop']:.1%}")

            print(f"\nAccuracy Ranking:")
            for i, (cond, acc) in enumerate(comp["accuracy_ranking"], 1):
                print(f"  {i}. {cond:<20} {acc:.1%}")

        print(f"\n{'='*80}")

    def analyze_error_patterns(self, report_file: Path):
        """Analyze error patterns from a saved report."""
        with open(report_file) as f:
            report = json.load(f)

        print(f"\n{'='*60}")
        print(f"ERROR PATTERN ANALYSIS")
        print(f"{'='*60}\n")

        for condition, data in report["conditions"].items():
            print(f"\n[{condition.upper()}]")
            print(f"Total position errors: {data['total_position_errors']}")

            if "top_error_positions" in data and data["top_error_positions"]:
                print(f"Most error-prone positions:")
                for item in data["top_error_positions"]:
                    row, col = item["position"]
                    count = item["count"]
                    print(f"  Position ({row},{col}): {count} errors")

            # Analyze if errors cluster at patch boundaries
            # This would need the error heatmap analysis

        print(f"\n{'='*60}")


# Example usage
if __name__ == "__main__":
    print("\n🎮 Available models:")
    for key in MODEL_CONFIGS.keys():
        print(f"  - {key}")
    print()

    # Initialize runner
    runner = GomokuPatchTestRunner(
        test_dir="gomoku_patch_tests",
        output_dir="gomoku_patch_tests/results",
        model_key="qwen3-vl-8b",  # Change this to test different models
    )

    # Run all conditions with limited tests for quick testing
    # Set to None to run all tests
    report = runner.run_all_conditions(
        max_tests_per_condition=None,  # Start with 5 for testing, then use None for full run
    )

    print("\n🎯 Analysis complete!")
    print("Check the results folder for:")
    print("  - Detailed logs for each condition")
    print("  - Comprehensive report with statistics")
    print("  - Error pattern analysis")
