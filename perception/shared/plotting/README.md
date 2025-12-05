# Unified Density Plotting Module

This module provides unified plotting utilities for density test results, supporting both **Gomoku** and **Chess** tests.

## Features

- **Unified Interface**: Single `DensityPlotter` class for both games
- **Automatic Game Detection**: Handles differences between Gomoku and Chess automatically
- **Single Model Analysis**: Plot accuracy vs density for one model
- **All Models Comparison**: Compare multiple models on one figure
- **Chess-Specific Breakdown**: Detailed detection error analysis for Chess

## Quick Start

### Basic Usage

```python
from shared.plotting import DensityPlotter

# For Gomoku
plotter = DensityPlotter(game_type="gomoku")
plotter.plot_single_model(
    model_key="qwen3-vl-8b",
    save_path="gomoku_density_qwen8b.png"
)

# For Chess
plotter = DensityPlotter(game_type="chess")
plotter.plot_single_model(
    model_key="gemini-2.5-flash-lite",
    save_path="chess_density_gemini25flashlite.png"
)
```

### All Models Comparison

```python
from shared.plotting import DensityPlotter

# Gomoku
plotter = DensityPlotter(game_type="gomoku")
plotter.plot_all_models(
    model_keys=["qwen3-vl-8b", "qwen3-vl-30b", "qwen3-vl-235b"],
    save_path="gomoku_density_all_models.png"
)

# Chess
plotter = DensityPlotter(game_type="chess")
plotter.plot_all_models(
    model_keys=["qwen3-vl-30b", "gemma-3-27b", "gemini-2.5-flash-lite"],
    save_path="chess_density_all_models.png"
)
```

### Chess Detection Breakdown

For detailed error analysis (exact detection, color errors, type errors, missed pieces):

```python
from shared.plotting import ChessDetectionBreakdownPlotter

plotter = ChessDetectionBreakdownPlotter()
plotter.plot_detection_analysis(
    model_key="gemini-2.5-flash-lite",
    save_path="chess_detection_breakdown.png"
)
```

## API Reference

### `DensityPlotter`

#### `__init__(game_type: str = "gomoku", results_dir: Optional[str] = None)`

Initialize the plotter.

- `game_type`: `"gomoku"` or `"chess"`
- `results_dir`: Path to results directory (defaults to game-specific)

#### `plot_single_model(model_key: str, model_display_name: Optional[str] = None, save_path: Optional[str] = None)`

Plot density vs accuracy for a single model.

- `model_key`: Model identifier (e.g., `"qwen3-vl-8b"`)
- `model_display_name`: Display name (defaults to predefined names)
- `save_path`: Path to save plot (if `None`, shows plot)

#### `plot_all_models(model_keys: List[str], save_path: Optional[str] = None)`

Plot comparison of multiple models.

- `model_keys`: List of model identifiers
- `save_path`: Path to save plot (if `None`, shows plot)

### `ChessDetectionBreakdownPlotter`

#### `plot_detection_analysis(model_key: str, model_display_name: Optional[str] = None, save_path: Optional[str] = None)`

Generate detailed detection breakdown plot for Chess (scatter + stacked bar).

## Game-Specific Differences

The module automatically handles differences between games:

| Feature | Gomoku | Chess |
|---------|--------|-------|
| Accuracy field | `stone_only_accuracy` | `piece_only_accuracy` |
| Invalid marker | `-1` | `-99` |
| Default results dir | `gomoku_density_test/results` | `chess_density_test/results` |
| Piece terminology | "Stones" | "Pieces" |

## Model Colors

Predefined colors for common models:

- `qwen3-vl-8b`: Blue (#3498db)
- `qwen3-vl-30b`: Purple (#9b59b6)
- `qwen3-vl-235b`: Red (#e74c3c) - highlighted
- `gemma-3-27b`: Orange (#e67e22)
- `gemini-2.5-flash-lite`: Teal (#1abc9c)
- `glm4v-thinking`: Yellow (#f39c12)

## Examples

See `example_usage.py` for complete examples.

## Dependencies

- `matplotlib>=3.5.0`
- `scipy>=1.7.0`
- `numpy>=1.20.0`

