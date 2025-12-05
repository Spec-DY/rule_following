"""
Unified density plotting for both Gomoku and Chess tests.

Supports:
- Single model analysis (accuracy vs density)
- All models comparison
- Chess-specific detection breakdown analysis
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional
from scipy.stats import pearsonr


class DensityPlotter:
    """
    Unified plotter for density tests (Gomoku and Chess).

    Automatically handles differences between games:
    - Gomoku: uses 'stone_only_accuracy', results_dir='gomoku_density_test/results'
    - Chess: uses 'piece_only_accuracy', results_dir='chess_density_test/results'
    """

    # Game-specific configurations
    GAME_CONFIGS = {
        "gomoku": {
            "accuracy_field": "stone_only_accuracy",
            "accuracy_display": "Stone-Only Accuracy (%)",
            "piece_display": "Number of Stones on Board",
            "default_results_dir": "gomoku_density_test/results",
            "title_prefix": "Density vs Stone Detection Capability",
            "invalid_marker": -1,  # Gomoku uses -1 for invalid
        },
        "chess": {
            "accuracy_field": "piece_only_accuracy",
            "accuracy_display": "Piece-Only Accuracy (%)",
            "piece_display": "Number of Pieces on Board",
            "default_results_dir": "chess_density_test/results",
            "title_prefix": "Density vs Piece Detection Capability",
            "invalid_marker": -99,  # Chess uses -99 for invalid
        },
    }

    # Color scheme for each model
    MODEL_COLORS = {
        "qwen3-vl-8b": "#3498db",  # Blue
        "qwen3-vl-8b-thinking": "#3498db",  # Blue
        "qwen3-vl-30b": "#9b59b6",  # Purple
        "qwen3-vl-30b-a3b-instruct": "#9b59b6",  # Purple
        "qwen3-vl-235b": "#e74c3c",  # Red (highlight)
        "qwen3-vl-plus": "#c0392b",  # Dark red
        "gemma3": "#e67e22",  # Orange
        "gemma-3-27b": "#e67e22",  # Orange
        "gemini-2.5-flash-lite": "#1abc9c",  # Teal
        "gemini-3-pro-preview": "#16a085",  # Dark teal
        "glm4v-thinking": "#f39c12",  # Yellow
        "glm-4.5v": "#8e44ad",  # Dark purple
    }

    # Display names for models
    MODEL_NAMES = {
        "qwen3-vl-8b": "Qwen3-VL 8B",
        "qwen3-vl-8b-thinking": "Qwen3-VL 8B Thinking",
        "qwen3-vl-30b": "Qwen3-VL 30B",
        "qwen3-vl-30b-a3b-instruct": "Qwen3-VL 30B",
        "qwen3-vl-235b": "Qwen3-VL 235B",
        "qwen3-vl-plus": "Qwen3-VL Plus",
        "gemma3": "Gemma-3 27B",
        "gemma-3-27b": "Gemma-3 27B",
        "gemini-2.5-flash-lite": "Gemini 2.5 Flash Lite",
        "gemini-3-pro-preview": "Gemini 3 Pro Preview",
        "glm4v-thinking": "GLM-4.1V Thinking",
        "glm-4.5v": "GLM-4.5V",
    }

    def __init__(self, game_type: str = "gomoku", results_dir: Optional[str] = None):
        """
        Initialize density plotter.

        Args:
            game_type: "gomoku" or "chess"
            results_dir: Path to results directory (defaults to game-specific)
        """
        if game_type not in self.GAME_CONFIGS:
            raise ValueError(
                f"Unknown game_type: {game_type}. Must be one of {list(self.GAME_CONFIGS.keys())}"
            )

        self.game_type = game_type
        self.config = self.GAME_CONFIGS[game_type]

        if results_dir is None:
            results_dir = self.config["default_results_dir"]

        self.results_dir = Path(results_dir)
        self.log_dir = self.results_dir / "logs"

    def load_density_data(self, model_key: str) -> Dict[str, List[Dict]]:
        """Load data for all three densities for one model."""
        data = {"low": [], "medium": [], "high": []}
        accuracy_field = self.config["accuracy_field"]
        invalid_marker = self.config["invalid_marker"]

        for density in ["low", "medium", "high"]:
            pattern = f"{density}_{model_key}_*.json"
            log_files = list(self.log_dir.glob(pattern))

            if not log_files:
                print(f"⚠️  No log file found for {model_key} - {density}")
                continue

            # Use the most recent if multiple
            log_file = sorted(log_files)[-1]
            print(f"📂 Loading: {log_file.name}")

            with open(log_file, "r") as f:
                results = json.load(f)

            for test in results:
                if "error" in test:
                    continue

                # Check for invalid parses
                if test.get("predicted", [[]])[0][0] == invalid_marker:
                    continue

                if accuracy_field in test:
                    data[density].append(
                        {
                            "piece_count": test["statistics"]["total_pieces"],
                            "density_pct": test["statistics"]["density"] * 100,
                            "accuracy": test[accuracy_field] * 100,
                        }
                    )

        return data

    def plot_single_model(
        self,
        model_key: str,
        model_display_name: Optional[str] = None,
        save_path: Optional[str] = None,
    ):
        """
        Generate density plot for a single model.

        Args:
            model_key: Model key (e.g., "qwen3-vl-8b")
            model_display_name: Display name for the model (defaults to MODEL_NAMES)
            save_path: Path to save the plot (if None, shows plot)
        """
        # Load data
        data = self.load_density_data(model_key)

        if not any(data.values()):
            print(f"❌ No data found for {model_key}")
            return None

        if model_display_name is None:
            model_display_name = self.MODEL_NAMES.get(model_key, model_key)

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))

        # Prepare data for plotting
        all_pieces = []
        all_densities = []
        all_accs = []
        density_colors = {
            "low": "#3498db",  # Blue
            "medium": "#e74c3c",  # Red
            "high": "#2ecc71",  # Green
        }

        # Plot scatter points for each density level
        for density_name, density_data in data.items():
            if not density_data:
                continue

            pieces = [d["piece_count"] for d in density_data]
            densities = [d["density_pct"] for d in density_data]
            accs = [d["accuracy"] for d in density_data]

            # Scatter plot
            ax.scatter(
                pieces,
                accs,
                alpha=0.5,
                s=80,
                color=density_colors[density_name],
                label=f"{density_name.capitalize()} Density",
                edgecolors="black",
                linewidth=0.5,
            )

            all_pieces.extend(pieces)
            all_densities.extend(densities)
            all_accs.extend(accs)

        # Calculate means for each density level
        density_means = {}
        for density_name in ["low", "medium", "high"]:
            if data[density_name]:
                density_means[density_name] = {
                    "piece_count": np.mean(
                        [d["piece_count"] for d in data[density_name]]
                    ),
                    "density_pct": np.mean(
                        [d["density_pct"] for d in data[density_name]]
                    ),
                    "accuracy": np.mean([d["accuracy"] for d in data[density_name]]),
                }

        # Plot line connecting means
        if len(density_means) >= 2:
            mean_pieces = [
                density_means[d]["piece_count"]
                for d in ["low", "medium", "high"]
                if d in density_means
            ]
            mean_accs = [
                density_means[d]["accuracy"]
                for d in ["low", "medium", "high"]
                if d in density_means
            ]

            ax.plot(
                mean_pieces,
                mean_accs,
                "k--",
                linewidth=2,
                alpha=0.7,
                label="Mean Trend",
                marker="D",
                markersize=10,
                markerfacecolor="yellow",
                markeredgecolor="black",
                markeredgewidth=1.5,
            )

        # Calculate correlation
        if len(all_pieces) > 0:
            corr, p_value = pearsonr(all_pieces, all_accs)
        else:
            corr, p_value = 0, 1

        # Labels and title
        ax.set_xlabel(self.config["piece_display"], fontsize=13, fontweight="bold")
        ax.set_ylabel(self.config["accuracy_display"], fontsize=13, fontweight="bold")

        ax.set_title(
            f"{self.config['title_prefix']}\nModel: {model_display_name}",
            fontsize=14,
            fontweight="bold",
            pad=20,
        )

        # Add secondary x-axis for density %
        ax2 = ax.twiny()
        ax2.set_xlabel("Board Density (%)", fontsize=13, fontweight="bold")

        # Set secondary axis limits based on primary
        if len(density_means) >= 2:
            ax2.set_xlim(ax.get_xlim())

            # Set tick labels at mean positions
            tick_positions = mean_pieces
            tick_labels = [
                f"{density_means[d]['density_pct']:.0f}%"
                for d in ["low", "medium", "high"]
                if d in density_means
            ]

            ax2.set_xticks(tick_positions)
            ax2.set_xticklabels(tick_labels)

        # Grid
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.set_axisbelow(True)

        # Legend
        ax.legend(loc="upper left", fontsize=10, framealpha=0.9)

        # Add correlation text box
        textstr = (
            f"Correlation: r = {corr:.3f}\n"
            f"p-value = {p_value:.2e}\n"
            f"n = {len(all_pieces)} samples"
        )
        props = dict(boxstyle="round", facecolor="wheat", alpha=0.8)
        ax.text(
            0.98,
            0.02,
            textstr,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="bottom",
            horizontalalignment="right",
            bbox=props,
        )

        # Set y-axis to start from 0
        if all_accs:
            ax.set_ylim(0, max(100, max(all_accs) * 1.1))

        plt.tight_layout()

        # Save or show
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"\n✅ Plot saved to: {save_path}")
        else:
            plt.show()

        # Print summary statistics
        self._print_summary_statistics(
            model_display_name, data, density_means, corr, p_value
        )

        return fig

    def plot_all_models(self, model_keys: List[str], save_path: Optional[str] = None):
        """
        Plot all models on one figure for comparison.

        Args:
            model_keys: List of model keys to plot
            save_path: Path to save the plot (if None, shows plot)
        """
        fig, ax = plt.subplots(figsize=(14, 8))

        # Track overall data range for axis
        all_pieces = []
        all_accs = []

        # Plot each model
        for model_key in model_keys:
            if model_key not in self.MODEL_COLORS:
                print(f"⚠️  Unknown model: {model_key}, skipping")
                continue

            data = self.load_density_data(model_key)

            if not any(data.values()):
                print(f"⚠️  No data for {model_key}, skipping")
                continue

            print(f"📊 Plotting {self.MODEL_NAMES.get(model_key, model_key)}...")

            color = self.MODEL_COLORS[model_key]
            model_name = self.MODEL_NAMES.get(model_key, model_key)

            # Collect all points for this model
            model_pieces = []
            model_accs = []

            for density_data in data.values():
                if density_data:
                    pieces = [d["piece_count"] for d in density_data]
                    accs = [d["accuracy"] for d in density_data]
                    model_pieces.extend(pieces)
                    model_accs.extend(accs)

            # Scatter plot (transparent)
            ax.scatter(
                model_pieces,
                model_accs,
                alpha=0.15,  # Very transparent
                s=60,
                color=color,
                edgecolors="none",
            )

            # Calculate means for each density
            means_x = []
            means_y = []

            for density_name in ["low", "medium", "high"]:
                if data[density_name]:
                    mean_pieces = np.mean(
                        [d["piece_count"] for d in data[density_name]]
                    )
                    mean_acc = np.mean([d["accuracy"] for d in data[density_name]])
                    means_x.append(mean_pieces)
                    means_y.append(mean_acc)

            # Plot trend line (solid, opaque)
            if len(means_x) >= 2:
                # Determine line width and style
                linewidth = 3.5 if model_key == "qwen3-vl-235b" else 2.5
                linestyle = "-"

                ax.plot(
                    means_x,
                    means_y,
                    color=color,
                    linewidth=linewidth,
                    linestyle=linestyle,
                    marker="o",
                    markersize=10,
                    markerfacecolor=color,
                    markeredgecolor="white",
                    markeredgewidth=2,
                    label=model_name,
                    alpha=0.9,
                    zorder=10 if model_key == "qwen3-vl-235b" else 5,
                )

            all_pieces.extend(model_pieces)
            all_accs.extend(model_accs)

        # Styling
        ax.set_xlabel(self.config["piece_display"], fontsize=14, fontweight="bold")
        ax.set_ylabel(self.config["accuracy_display"], fontsize=14, fontweight="bold")
        ax.set_title(
            f"{self.config['title_prefix']}\nAll Models Comparison - {self.game_type.capitalize()}",
            fontsize=16,
            fontweight="bold",
            pad=20,
        )

        # Secondary x-axis for density %
        ax2 = ax.twiny()
        ax2.set_xlabel("Board Density (%)", fontsize=14, fontweight="bold")
        ax2.set_xlim(ax.get_xlim())

        # Set density ticks (game-specific)
        if self.game_type == "gomoku":
            density_ticks = [56, 100, 147]  # Approximate piece counts
            density_labels = ["25%", "45%", "65%"]
        else:  # chess
            density_ticks = [10, 18, 30]  # Approximate piece counts
            density_labels = ["31%", "56%", "94%"]

        ax2.set_xticks(density_ticks)
        ax2.set_xticklabels(density_labels)

        # Grid
        ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
        ax.set_axisbelow(True)

        # Y-axis limits
        ax.set_ylim(0, 100)

        # Legend
        ax.legend(
            loc="upper left",
            fontsize=11,
            framealpha=0.95,
            edgecolor="black",
            fancybox=True,
        )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"\n✅ Plot saved to: {save_path}")
        else:
            plt.show()

        # Print summary
        print(f"\n{'='*70}")
        print(f"ALL MODELS COMPARISON - {self.game_type.upper()}")
        print(f"{'='*70}")
        print(f"Total models plotted: {len(model_keys)}")
        print(f"Total data points: {len(all_pieces)}")
        print(f"{'='*70}\n")

        return fig

    def _print_summary_statistics(
        self,
        model_name: str,
        data: Dict,
        density_means: Dict,
        corr: float,
        p_value: float,
    ):
        """Print summary statistics for single model plot."""
        print(f"\n{'='*60}")
        print(f"SUMMARY STATISTICS - {model_name}")
        print(f"{'='*60}")

        for density_name in ["low", "medium", "high"]:
            if density_name in density_means:
                stats = density_means[density_name]
                density_data = data[density_name]

                accs = [d["accuracy"] for d in density_data]

                print(f"\n{density_name.upper()} Density:")
                print(f"  Average pieces: {stats['piece_count']:.1f}")
                print(f"  Average density: {stats['density_pct']:.1f}%")
                print(f"  Accuracy: {stats['accuracy']:.1f}%")
                print(f"  Std Dev: {np.std(accs):.1f}%")
                print(f"  Range: {min(accs):.1f}% - {max(accs):.1f}%")

        print(f"\n{'='*60}")
        print(f"Overall Correlation: r = {corr:.3f} (p = {p_value:.2e})")
        print(f"{'='*60}\n")


# Chess-specific detection breakdown plotter
class ChessDetectionBreakdownPlotter:
    """
    Chess-specific plotter for detection breakdown analysis.

    Shows exact detection, color errors, type errors, and missed pieces.
    """

    def __init__(self, results_dir: Optional[str] = None):
        if results_dir is None:
            results_dir = "chess_density_test/results"
        self.results_dir = Path(results_dir)
        self.log_dir = self.results_dir / "logs"

    def _calculate_detection_metrics(self, predicted, ground_truth):
        """Calculate complete detection metrics for chess pieces."""
        pred = np.array(predicted)
        truth = np.array(ground_truth)

        piece_mask = truth != 0
        gt_count = np.sum(piece_mask)

        if gt_count == 0:
            return None

        # Exact detection (position + type + color correct)
        exact_detected = np.sum(pred[piece_mask] == truth[piece_mask])

        # Breakdown of errors
        color_errors = 0
        type_errors = 0
        missed = 0

        for i in range(8):
            for j in range(8):
                if truth[i, j] != 0:  # Ground truth has a piece
                    pred_val = pred[i, j]
                    truth_val = truth[i, j]

                    if pred_val == truth_val:
                        # Exact match - already counted
                        pass
                    elif pred_val != 0:
                        # Detected something, but wrong
                        if (pred_val > 0) == (truth_val > 0):
                            # Correct color, wrong type
                            type_errors += 1
                        else:
                            # Wrong color
                            color_errors += 1
                    else:
                        # Missed the piece
                        missed += 1

        return {
            "ground_truth_count": int(gt_count),
            "exact_detected": int(exact_detected),
            "color_errors": int(color_errors),
            "type_errors": int(type_errors),
            "missed": int(missed),
            "exact_recall": float(exact_detected / gt_count),
        }

    def load_density_data_with_breakdown(self, model_key: str):
        """Load data with detection breakdown for all densities."""
        data = {"low": [], "medium": [], "high": []}

        for density in ["low", "medium", "high"]:
            pattern = f"{density}_{model_key}_*.json"
            log_files = list(self.log_dir.glob(pattern))

            if not log_files:
                print(f"⚠️  No log file found for {model_key} - {density}")
                continue

            log_file = sorted(log_files)[-1]
            print(f"📂 Loading: {log_file.name}")

            with open(log_file, "r") as f:
                results = json.load(f)

            for test in results:
                if "error" not in test and "predicted" in test:
                    # Skip invalid parses
                    if test["predicted"][0][0] == -99:
                        continue

                    # Calculate detection breakdown
                    metrics = self._calculate_detection_metrics(
                        test["predicted"], test["ground_truth"]
                    )

                    if metrics:
                        data[density].append(
                            {
                                "piece_count": test["statistics"]["total_pieces"],
                                "density_pct": test["statistics"]["density"] * 100,
                                "piece_only_acc": test["piece_only_accuracy"] * 100,
                                **metrics,
                            }
                        )

        return data

    def plot_detection_analysis(
        self,
        model_key: str,
        model_display_name: Optional[str] = None,
        save_path: Optional[str] = None,
    ):
        """Generate combined plot: scatter + stacked bar breakdown."""
        # Load data
        data = self.load_density_data_with_breakdown(model_key)

        if not any(data.values()):
            print(f"❌ No data found for {model_key}")
            return None

        if model_display_name is None:
            model_display_name = DensityPlotter.MODEL_NAMES.get(model_key, model_key)

        # Create figure with two subplots
        fig = plt.figure(figsize=(14, 11))
        gs = fig.add_gridspec(2, 1, height_ratios=[2, 1], hspace=0.35)

        ax1 = fig.add_subplot(gs[0])  # Top: Scatter plot
        ax2 = fig.add_subplot(gs[1])  # Bottom: Stacked bar

        # =====================================================================
        # TOP PLOT: Exact Match Detection Scatter
        # =====================================================================

        density_colors = {
            "low": "#3498db",  # Blue
            "medium": "#e74c3c",  # Red
            "high": "#2ecc71",  # Green
        }

        all_pieces = []
        all_exact = []
        density_means = {}

        # Plot scatter points
        for density_name, density_data in data.items():
            if not density_data:
                continue

            pieces = [d["piece_count"] for d in density_data]
            exact_detected = [d["exact_detected"] for d in density_data]

            # Scatter
            ax1.scatter(
                pieces,
                exact_detected,
                alpha=0.5,
                s=80,
                color=density_colors[density_name],
                label=f"{density_name.capitalize()} Density",
                edgecolors="black",
                linewidth=0.5,
            )

            all_pieces.extend(pieces)
            all_exact.extend(exact_detected)

            # Calculate means
            density_means[density_name] = {
                "piece_count": np.mean(pieces),
                "exact_detected": np.mean(exact_detected),
                "color_errors": np.mean([d["color_errors"] for d in density_data]),
                "type_errors": np.mean([d["type_errors"] for d in density_data]),
                "missed": np.mean([d["missed"] for d in density_data]),
                "exact_recall": np.mean([d["exact_recall"] for d in density_data]),
            }

        # Perfect detection diagonal line (y = x)
        max_pieces = max(all_pieces) if all_pieces else 32
        ax1.plot(
            [0, max_pieces],
            [0, max_pieces],
            "k--",
            alpha=0.3,
            linewidth=2,
            label="Perfect Detection (y=x)",
        )

        # Trend line connecting means
        if len(density_means) >= 2:
            mean_pieces = [
                density_means[d]["piece_count"]
                for d in ["low", "medium", "high"]
                if d in density_means
            ]
            mean_exact = [
                density_means[d]["exact_detected"]
                for d in ["low", "medium", "high"]
                if d in density_means
            ]

            ax1.plot(
                mean_pieces,
                mean_exact,
                "k-",
                linewidth=3,
                alpha=0.8,
                marker="D",
                markersize=12,
                markerfacecolor="yellow",
                markeredgecolor="black",
                markeredgewidth=1.5,
                label="Mean Trend",
                zorder=10,
            )

        # Styling for top plot
        ax1.set_xlabel("Ground Truth Piece Count", fontsize=13, fontweight="bold")
        ax1.set_ylabel(
            "Correctly Detected Pieces\n(Exact Match)", fontsize=13, fontweight="bold"
        )
        ax1.set_title(
            f"Density vs Piece Detection Capability\nModel: {model_display_name}",
            fontsize=15,
            fontweight="bold",
            pad=20,
        )

        ax1.grid(True, alpha=0.3, linestyle="--")
        ax1.set_axisbelow(True)
        ax1.legend(loc="upper left", fontsize=10, framealpha=0.9)

        # Set limits
        if all_pieces:
            ax1.set_xlim(0, max(all_pieces) * 1.1)
            ax1.set_ylim(0, max(max(all_exact) * 1.15, max(all_pieces) * 0.3))

        # Correlation stats
        if len(all_pieces) > 0:
            corr, p_value = pearsonr(all_pieces, all_exact)
            textstr = (
                f"Correlation: r = {corr:.3f}\np = {p_value:.2e}\nn = {len(all_pieces)}"
            )
            props = dict(boxstyle="round", facecolor="wheat", alpha=0.8)
            ax1.text(
                0.98,
                0.02,
                textstr,
                transform=ax1.transAxes,
                fontsize=10,
                verticalalignment="bottom",
                horizontalalignment="right",
                bbox=props,
            )

        # =====================================================================
        # BOTTOM PLOT: Detection Breakdown (Stacked Bar)
        # =====================================================================

        if len(density_means) >= 2:
            density_order = ["low", "medium", "high"]
            densities_present = [d for d in density_order if d in density_means]

            positions = np.arange(len(densities_present))
            width = 0.6

            # Prepare data for stacking
            exact_counts = [
                density_means[d]["exact_detected"] for d in densities_present
            ]
            color_error_counts = [
                density_means[d]["color_errors"] for d in densities_present
            ]
            type_error_counts = [
                density_means[d]["type_errors"] for d in densities_present
            ]
            missed_counts = [density_means[d]["missed"] for d in densities_present]

            # Create stacked bars
            ax2.bar(
                positions,
                exact_counts,
                width,
                label="Correctly Detected",
                color="#2ecc71",  # Green
                alpha=0.8,
                edgecolor="black",
                linewidth=1,
            )

            ax2.bar(
                positions,
                color_error_counts,
                width,
                bottom=exact_counts,
                label="Color Error",
                color="#e74c3c",  # Red
                alpha=0.8,
                edgecolor="black",
                linewidth=1,
            )

            bottom_for_type = np.array(exact_counts) + np.array(color_error_counts)
            ax2.bar(
                positions,
                type_error_counts,
                width,
                bottom=bottom_for_type,
                label="Type Error (Color Correct)",
                color="#f39c12",  # Orange
                alpha=0.8,
                edgecolor="black",
                linewidth=1,
            )

            bottom_for_missed = bottom_for_type + np.array(type_error_counts)
            ax2.bar(
                positions,
                missed_counts,
                width,
                bottom=bottom_for_missed,
                label="Missed Pieces",
                color="#95a5a6",  # Gray
                alpha=0.8,
                edgecolor="black",
                linewidth=1,
            )

            # Add value labels on bars
            for i, d in enumerate(densities_present):
                total = density_means[d]["piece_count"]
                exact = density_means[d]["exact_detected"]
                color_err = density_means[d]["color_errors"]
                type_err = density_means[d]["type_errors"]
                missed = density_means[d]["missed"]

                # Label for exact detected
                if exact > 2:
                    ax2.text(
                        i,
                        exact / 2,
                        f"{exact:.1f}",
                        ha="center",
                        va="center",
                        fontsize=10,
                        fontweight="bold",
                        color="white",
                    )

                # Label for color errors
                if color_err > 2:
                    ax2.text(
                        i,
                        exact + color_err / 2,
                        f"{color_err:.1f}",
                        ha="center",
                        va="center",
                        fontsize=10,
                        fontweight="bold",
                        color="white",
                    )

                # Label for type errors
                if type_err > 1:
                    ax2.text(
                        i,
                        exact + color_err + type_err / 2,
                        f"{type_err:.1f}",
                        ha="center",
                        va="center",
                        fontsize=9,
                        fontweight="bold",
                        color="white",
                    )

                # Label for missed
                if missed > 2:
                    ax2.text(
                        i,
                        exact + color_err + type_err + missed / 2,
                        f"{missed:.1f}",
                        ha="center",
                        va="center",
                        fontsize=10,
                        fontweight="bold",
                        color="white",
                    )

                # Total count on top
                ax2.text(
                    i,
                    total + 2,
                    f"Total: {total:.0f}",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    fontweight="bold",
                )

            # Styling for bottom plot
            ax2.set_xlabel("Density Level", fontsize=13, fontweight="bold")
            ax2.set_ylabel("Piece Count", fontsize=13, fontweight="bold")
            ax2.set_title(
                "Detection Breakdown by Density (Mean Values)",
                fontsize=13,
                fontweight="bold",
                pad=15,
            )

            ax2.set_xticks(positions)
            ax2.set_xticklabels([d.capitalize() for d in densities_present])
            ax2.legend(loc="upper right", fontsize=10, framealpha=0.9)
            ax2.grid(True, axis="y", alpha=0.3, linestyle="--")
            ax2.set_axisbelow(True)

            # Y limit
            max_total = max(
                [density_means[d]["piece_count"] for d in densities_present]
            )
            ax2.set_ylim(0, max_total * 1.15)

        plt.tight_layout()

        # Save or show
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"\n✅ Plot saved to: {save_path}")
        else:
            plt.show()

        # Print summary statistics
        self._print_summary_statistics(model_display_name, data, density_means)

        return fig

    def _print_summary_statistics(self, model_name, data, density_means):
        """Print detailed breakdown statistics."""
        print(f"\n{'='*70}")
        print(f"DETECTION BREAKDOWN ANALYSIS - {model_name}")
        print(f"{'='*70}")

        for density_name in ["low", "medium", "high"]:
            if density_name not in density_means:
                continue

            stats = density_means[density_name]
            density_data = data[density_name]

            print(f"\n{density_name.upper()} DENSITY:")
            print(f"  Ground Truth Pieces: {stats['piece_count']:.1f} (avg)")
            print(
                f"  ├─ Correctly Detected: {stats['exact_detected']:.1f} "
                f"({stats['exact_recall']*100:.1f}%)"
            )
            print(f"  ├─ Color Errors: {stats['color_errors']:.1f}")
            print(f"  ├─ Type Errors: {stats['type_errors']:.1f}")
            print(f"  └─ Missed: {stats['missed']:.1f}")

            # Calculate standard deviations
            exact_values = [d["exact_detected"] for d in density_data]
            print(f"  Std Dev (Exact): {np.std(exact_values):.1f}")

        print(f"\n{'='*70}")
        print("KEY FINDINGS:")
        print(f"{'='*70}")

        # Compare low vs high
        if "low" in density_means and "high" in density_means:
            low = density_means["low"]
            high = density_means["high"]

            exact_improvement = high["exact_detected"] - low["exact_detected"]
            missed_reduction = low["missed"] - high["missed"]

            print(f"From LOW to HIGH density:")
            print(f"  ✓ Exact detection: +{exact_improvement:.1f} pieces")
            print(f"  ✓ Missed pieces: {missed_reduction:+.1f}")
            print(
                f"  → Detection recall: "
                f"{low['exact_recall']*100:.1f}% → {high['exact_recall']*100:.1f}%"
            )

            # Chess-specific insight
            total_errors_low = low["color_errors"] + low["type_errors"]
            total_errors_high = high["color_errors"] + high["type_errors"]
            print(
                f"  → Total errors: "
                f"{total_errors_low:.1f} → {total_errors_high:.1f}"
            )

        print(f"{'='*70}\n")
