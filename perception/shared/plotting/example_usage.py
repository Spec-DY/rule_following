"""
Example usage of the unified density plotting module.

This demonstrates how to use DensityPlotter for both Gomoku and Chess tests.

Usage:
    # From project root:
    python -m shared.plotting.example_usage

    # Or import in your own scripts:
    from shared.plotting import DensityPlotter, ChessDetectionBreakdownPlotter
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared.plotting.density_plots import DensityPlotter, ChessDetectionBreakdownPlotter


def example_gomoku_single_model():
    """Example: Plot single model for Gomoku."""
    print("=" * 70)
    print("GOMOKU - SINGLE MODEL PLOT")
    print("=" * 70)
    print()

    plotter = DensityPlotter(game_type="gomoku")

    plotter.plot_single_model(
        model_key="qwen3-vl-8b",
        model_display_name="Qwen3-VL-8B",
        save_path="gomoku_density_qwen8b.png",
    )


def example_gomoku_all_models():
    """Example: Plot all models comparison for Gomoku."""
    print("=" * 70)
    print("GOMOKU - ALL MODELS COMPARISON")
    print("=" * 70)
    print()

    plotter = DensityPlotter(game_type="gomoku")

    models_to_plot = [
        "qwen3-vl-8b",
        "qwen3-vl-30b",
        "gemma-3-27b",
        "gemini-2.5-flash-lite",
        "qwen3-vl-235b",
        "glm4v-thinking",
    ]

    plotter.plot_all_models(
        model_keys=models_to_plot,
        save_path="gomoku_density_all_models.png",
    )


def example_chess_single_model():
    """Example: Plot single model for Chess."""
    print("=" * 70)
    print("CHESS - SINGLE MODEL PLOT")
    print("=" * 70)
    print()

    plotter = DensityPlotter(game_type="chess")

    plotter.plot_single_model(
        model_key="gemini-2.5-flash-lite",
        model_display_name="Gemini 2.5 Flash Lite",
        save_path="chess_density_gemini25flashlite.png",
    )


def example_chess_all_models():
    """Example: Plot all models comparison for Chess."""
    print("=" * 70)
    print("CHESS - ALL MODELS COMPARISON")
    print("=" * 70)
    print()

    plotter = DensityPlotter(game_type="chess")

    models_to_plot = [
        "qwen3-vl-30b",
        "gemma-3-27b",
        "gemini-2.5-flash-lite",
    ]

    plotter.plot_all_models(
        model_keys=models_to_plot,
        save_path="chess_density_all_models.png",
    )


def example_chess_detection_breakdown():
    """Example: Chess-specific detection breakdown analysis."""
    print("=" * 70)
    print("CHESS - DETECTION BREAKDOWN ANALYSIS")
    print("=" * 70)
    print()

    plotter = ChessDetectionBreakdownPlotter()

    plotter.plot_detection_analysis(
        model_key="gemini-2.5-flash-lite",
        model_display_name="Gemini 2.5 Flash Lite",
        save_path="chess_detection_breakdown_gemini25flashlite.png",
    )


if __name__ == "__main__":
    # Uncomment the example you want to run:

    # Gomoku examples
    # example_gomoku_single_model()
    example_gomoku_all_models()

    # Chess examples
    example_chess_single_model()
    example_chess_all_models()
    example_chess_detection_breakdown()

    print("\n💡 Uncomment examples in the script to run them!")
    print("   Or import DensityPlotter in your own scripts.")
