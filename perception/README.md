# Perception Test Suite

A unified codebase for VLM (Vision Language Model) perception testing across multiple game scenarios.

## Project Structure

```
perception/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── tests/                       # All test scripts
│   ├── richness/                # Visual richness tests (2D vs 3D)
│   │   ├── generate_gomoku_richness.py
│   │   └── run_gomoku_richness_tests.py
│   ├── density/                 # Density diagnostic tests
│   │   ├── generate_chess_density.py
│   │   ├── run_chess_density_tests.py
│   │   ├── generate_gomoku_density.py
│   │   └── run_gomoku_density_tests.py
│   ├── resolution/              # Resolution tests
│   │   ├── generate_tictactoe_reso.py
│   │   └── run_tictactoe_reso_tests.py
│   └── patch/                   # Patch alignment tests
│       ├── generate_gomoku.py
│       └── run_gomoku_tests.py
├── shared/                      # Shared components
│   ├── __init__.py
│   ├── model_configs.py        # Unified model configurations
│   └── plotting/               # Unified plotting utilities
│       ├── __init__.py
│       ├── density_plots.py    # Density test plotting (Gomoku & Chess)
│       ├── example_usage.py    # Plotting examples
│       └── README.md           # Plotting documentation
└── assets/                      # Shared assets (wood textures, pieces, etc.)
    ├── wood_texture.jpg
    ├── black_stone.png
    ├── white_stone.png
    ├── ...
```

## Test Types

### 1. Patch Alignment Test (Gomoku)

Tests how patch alignment (aligned vs offset) affects perception.

- **Generator**: `tests/patch/generate_gomoku.py`
- **Runner**: `tests/patch/run_gomoku_tests.py`
- **Output**: `gomoku_patch_tests/` (contains `aligned/`, `offset_quarter/`, etc., `results/`)

### 2. Resolution Test (TicTacToe)

Tests how image resolution and patch divisibility affect perception.

- **Generator**: `tests/resolution/generate_tictactoe_reso.py`
- **Runner**: `tests/resolution/run_tictactoe_reso_tests.py`
- **Output**: `tictactoe_resolution_tests/` (contains `divisible/`, `non_divisible/`, `results/`)

### 3. Visual Richness Test (Gomoku)

Tests whether 2D flat vs 3D rendered visual styles affect VLM perception.

- **Generator**: `tests/richness/generate_gomoku_richness.py`
- **Runner**: `tests/richness/run_gomoku_richness_tests.py`
- **Output**: `gomoku_visual_richness_tests/` (contains `2d_flat/`, `3d_rendered/`, `results/`)

### 4. Density Test (Gomoku)

Similar to Chess density test but for Gomoku stones.

- **Generator**: `tests/density/generate_gomoku_density.py`
- **Runner**: `tests/density/run_gomoku_density_tests.py`
- **Output**: `gomoku_density_test/` (contains `low/`, `medium/`, `high/`, `results/`)

### 5. Density Test (Chess)

Tests how board density (low/medium/high piece count) affects piece detection.

- **Generator**: `tests/density/generate_chess_density.py`
- **Runner**: `tests/density/run_chess_density_tests.py`
- **Output**: `chess_density_test/` (contains `low/`, `medium/`, `high/`, `results/`)

## Usage

### Generate Test Data

Each test has a generator script. Run from the project root:

```bash
# Patch Alignment
python tests/patch/generate_gomoku.py

# Resolution
python tests/resolution/generate_tictactoe_reso.py

# Visual Richness
python tests/richness/generate_gomoku_richness.py

# Gomoku Density
python tests/density/generate_gomoku_density.py

# Chess Density
python tests/density/generate_chess_density.py
```

### Run Tests

Each test has a runner script. Run from the project root:

```bash
# Patch Alignment
python tests/patch/run_gomoku_tests.py

# Resolution
python tests/resolution/run_tictactoe_reso_tests.py

# Visual Richness
python tests/richness/run_gomoku_richness_tests.py

# Gomoku Density
python tests/density/run_gomoku_density_tests.py

# Chess Density
python tests/density/run_chess_density_tests.py
```

## Output Format

All tests follow a consistent output structure:

- Test images and JSON metadata in test-specific directories (e.g., `gomoku_visual_richness_tests/`)
- Results and reports in `{test_name}/results/`
- Detailed logs in `{test_name}/results/logs/`

All output directories are created at the project root level (same level as `tests/`).

## Dependencies

See `requirements.txt` for full list. Main dependencies:

- `numpy`
- `Pillow` (PIL)
- `openai` (for API client)
- `python-chess` (for Chess density test)
- `pathlib` (standard library)

## Model Configurations

All model configurations are centralized in `shared/model_configs.py`. This includes:

- **Qwen3-VL models** (Alibaba DashScope): qwen3-vl-8b, qwen3-vl-30b, qwen3-vl-235b, etc.
- **Google models** (Gemini/Gemma): gemma-3-27b, gemini-2.5-flash-lite, gemini-3-pro-preview
- **GLM models** (Zhipu AI): glm4v-thinking, glm-4.5v

All test runners import and use the shared `MODEL_CONFIGS` dictionary. To add a new model, edit `shared/model_configs.py`.

## Plotting Results

The project includes a unified plotting module for density test results in `shared/plotting/`.

### Density Plotting

The `DensityPlotter` class supports both Gomoku and Chess density tests:

```python
from shared.plotting import DensityPlotter

# For Gomoku
plotter = DensityPlotter(game_type="gomoku")
plotter.plot_single_model(
    model_key="qwen3-vl-8b",
    save_path="gomoku_density_qwen8b.png"
)
plotter.plot_all_models(
    model_keys=["qwen3-vl-8b", "qwen3-vl-30b", "qwen3-vl-235b"],
    save_path="gomoku_density_all_models.png"
)

# For Chess
plotter = DensityPlotter(game_type="chess")
plotter.plot_single_model(
    model_key="gemini-2.5-flash-lite",
    save_path="chess_density_gemini25flashlite.png"
)
plotter.plot_all_models(
    model_keys=["qwen3-vl-30b", "gemma-3-27b", "gemini-2.5-flash-lite"],
    save_path="chess_density_all_models.png"
)
```

### Chess Detection Breakdown

For detailed Chess analysis with error breakdown:

```python
from shared.plotting import ChessDetectionBreakdownPlotter

plotter = ChessDetectionBreakdownPlotter()
plotter.plot_detection_analysis(
    model_key="gemini-2.5-flash-lite",
    save_path="chess_detection_breakdown.png"
)
```

See `shared/plotting/example_usage.py` for more examples.

## Notes

- All output paths are relative to the project root
- Run scripts from the project root directory
- Assets should be placed in the `assets/` directory
- Model configurations are centralized in `shared/model_configs.py`
- Each test can be run independently
- Plotting utilities are in `shared/plotting/` and support both Gomoku and Chess
