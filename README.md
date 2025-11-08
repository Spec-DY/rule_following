# VLM Rule Following Test

A comprehensive testing framework for evaluating Vision Language Models (VLMs) on spatial and temporal reasoning abilities using chess scenarios.

## 📋 Overview

This project systematically tests VLMs through generated chess board images to evaluate their capabilities in:

- **Spatial Reasoning**: Board position recognition, relative directions, diagonal relationships, etc.
- **Rule Understanding**: Chess rules application (piece movements, en passant, castling, etc.)
- **Temporal Reasoning**: Tracking piece movement sequences and understanding state changes

Each test includes a **verification mechanism** to ensure the model can correctly recognize the board before testing its reasoning abilities.

## ✨ Key Features

### 🎯 Four Test Suites

**1. Spatial Test 0 - Pure Spatial Reasoning**

- Tests fundamental spatial understanding (no chess knowledge required)
- Includes: same file/rank detection, diagonal recognition, relative directions, path clearance
- ~90 test cases

**2. Spatial Test 1 - Rule Following Baseline**

- Tests movement rules for all 6 piece types (King, Queen, Rook, Bishop, Knight, Pawn)
- Includes: legal moves, blocked paths, castling rules (through check & in check)
- 100+ test cases

**3. Temporal Test 0 - Pure Temporal Reasoning**

- Tests sequence understanding and state tracking
- Includes: movement tracking, state comparison, sequence comprehension
- Multi-frame image tests

**4. Temporal Test 1 - Temporal Rule Following**

- Tests complex time-dependent chess rules
- Includes: En Passant, Castling with temporal constraints
- Event recognition and rule application

### 🔍 Verification Mechanism

Every test case includes two-step verification:

1. **Board Recognition Verification**: Ensures the model correctly sees the board and pieces
2. **Ability Testing**: Only counts toward results after passing verification

This design distinguishes between "cannot see the board" and "insufficient reasoning ability" failure modes.

### 🔌 Multi-Model Support

Supports any OpenAI-compatible API:

- **DashScope** (Alibaba)
- **Novita AI**
- **XAI** (Grok)
- **Custom endpoints** (easily extensible)
- **Dummy Model** (for testing the framework)

## 🚀 Quick Start

### Prerequisites

```bash
# Python 3.8+
pip install -r requirements.txt
```

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd rulefollow_test

# Install dependencies
pip install -r requirements.txt

# Or install in development mode
pip install -e .
```

### Configuration

Create a `.env` file in the project root:

```env
# For DashScope (Alibaba Qwen models)
DASHSCOPE_API_KEY=your_api_key_here
DASHSCOPE_MODEL=qwen-vl-max
DASHSCOPE_BASE_URL=https://dashscope-intl.aliyuncs.com/compatible-mode/v1

# For Novita AI
NOVITA_API_KEY=your_api_key_here
NOVITA_BASE_URL=https://api.novita.ai/openai
NOVITA_MODEL=qwen/qwen3-vl-235b-a22b-thinking

# For XAI (Grok)
XAI_API_KEY=your_api_key_here
XAI_MODEL=grok-4-fast-reasoning
XAI_BASE_URL=https://api.x.ai/v1
```

### Running Tests

```bash
# Run Spatial Test 0 (Pure Spatial Reasoning)
python examples/run_spatial_test_0.py

# Run Spatial Test 1 (Rule Following)
python examples/run_spatial_test_1.py

# Run Temporal Test 0 (Pure Temporal Reasoning)
python examples/run_temporal_test_0.py

# Run Temporal Test 1 (Temporal Rule Following)
python examples/run_temporal_test_1.py
```

### Customizing Test Parameters

Edit the configuration section in each run file:

```python
# In examples/run_spatial_test_0.py
N_CASES_PER_TYPE = 18      # Number of cases per test type
SEED = 57                  # Random seed for reproducibility
MODEL_TYPE = "xai"         # Options: "dummy", "dashscope", "novita", "xai"
RATE_LIMIT_REQUESTS = 0    # Number of requests before pausing
RATE_LIMIT_PAUSE = 0       # Pause duration in seconds
```

## 📂 Project Structure

```
rulefollow_test/
├── assets/
│   └── pieces/          # Chess piece PNG images (wk.png, bq.png, etc.)
├── src/
│   ├── spatial/         # Spatial reasoning tests
│   │   ├── test_0_pure_ability.py
│   │   ├── test_0_generator.py
│   │   ├── test_1_rule_following.py
│   │   ├── test_1_generator.py
│   │   ├── spatial_test_base.py
│   │   └── verification_generator.py
│   ├── temporal/        # Temporal reasoning tests
│   │   ├── test_0_pure_ability.py
│   │   ├── test_0_generator.py
│   │   ├── test_1_rule_following.py
│   │   └── test_1_generator.py
│   ├── board_generator.py      # Chess board image generation
│   ├── model_client.py          # Model API clients
│   └── data_structures.py       # Data models and utilities
├── examples/            # Example run scripts
│   ├── run_spatial_test_0.py
│   ├── run_spatial_test_1.py
│   ├── run_temporal_test_0.py
│   └── run_temporal_test_1.py
├── output/             # Test results (auto-generated)
└── requirements.txt
```

## 🧪 Test Output

Each test run generates:

### Generated Files

- **PNG Images**: Visual test cases with highlighted squares and pieces
- **JSON Results**: Detailed results with verification and accuracy metrics

### Output Structure

```
output/
└── spatial_test_0_MMDD_HHMMSS/
    ├── *.png                    # Test case images
    └── test_0_results.json      # Results with summary
```

### Results JSON Format

```json
{
  "summary": {
    "model_name": "grok-4-fast-reasoning",
    "total_cases": 90,
    "board_recognition": {
      "verified_correctly": 85,
      "failed_to_recognize": 5,
      "verification_rate": 0.944
    },
    "test_accuracy": {
      "correct_among_verified": 78,
      "total_verified": 85,
      "accuracy_given_verified": 0.918,
      "overall_accuracy": 0.867
    },
    "accuracy_by_type_verified_only": {
      "diagonal": {"correct": 15, "total": 18, "accuracy": 0.833},
      "same_line_same_file": {"correct": 9, "total": 10, "accuracy": 0.900}
    }
  },
  "detailed_results": [...]
}
```

## 💡 How It Works

### 1. Test Case Generation

Each test automatically generates diverse cases:

```python
# Spatial Test 0 Generator
generator = SpatialTest0Generator(seed=42)
cases = generator.generate_all(n_per_type=10)
# Generates: same file, same rank, diagonal, direction, path clear tests
```

### 2. Board Image Creation

Uses custom chess piece images to generate visual test cases:

```python
board_gen = ChessBoardGenerator()
img = board_gen.create_board_with_pieces(
    pieces={"e4": "N", "f6": "n"},  # Uppercase=white, lowercase=black
    highlighted_squares=["e4", "f6"]
)
```

### 3. Combined Verification + Testing

Each query includes both verification and test questions:

```
Verification: What square is the piece on?
Main Question: Can this knight move to the highlighted square?
```

### 4. Result Analysis

The framework tracks:

- **Verification Rate**: How many boards were correctly recognized
- **Test Accuracy (Verified)**: Accuracy among correctly recognized boards
- **Overall Accuracy**: Total accuracy including verification failures
- **Type Breakdown**: Performance by test type and subtype

## 🔧 Adding New Models

Extend the `OpenAICompatibleModelClient` class:

```python
class YourModelClient(OpenAICompatibleModelClient):
    DEFAULT_BASE_URL = "https://api.yourservice.com/v1"
    ENV_API_KEY = "YOUR_API_KEY"
    ENV_BASE_URL = "YOUR_BASE_URL"
    ENV_MODEL = "YOUR_MODEL"
    SERVICE_NAME = "YourService"
```

Add to `.env`:

```env
YOUR_API_KEY=your_key
YOUR_MODEL=your_model_name
```

## 🛠️ Advanced Usage

### Custom Test Cases

```python
# Create custom test
test = SpatialTest0(
    base_output_dir="./custom_output",
    n_cases_per_type=20,  # More cases per type
    seed=123,             # Different random seed
    auto_timestamp=True,
    rate_limit_requests=50,  # Pause every 50 requests
    rate_limit_pause=60      # 60 second pause
)

# Generate and run
test.generate_test_cases()
test.create_test_images()
results, stats = test.run_test(model_client)
```

### Rate Limiting

For API rate limits, configure pause intervals:

```python
test = SpatialTest1(
    rate_limit_requests=50,  # Pause after 50 requests
    rate_limit_pause=60      # Wait 60 seconds
)
```

### Batch Testing Multiple Models

```python
models = ["dashscope", "novita", "xai"]
for model_type in models:
    model_client = create_client(model_type)
    results, stats = test.run_test(model_client)
    # Results auto-saved to separate directories
```

## 🔬 Test Types Explained

### Spatial Test 0 Categories

- **Same Line**: File (vertical) and Rank (horizontal) alignment
- **Diagonal**: Diagonal relationship detection
- **Relative Position**: 8 directional relationships (N, NE, E, SE, S, SW, W, NW)
- **Path Clear**: Obstacle detection between squares

### Spatial Test 1 Categories

Each piece type has 3 subtests:

- **Clear Path**: Valid moves without obstacles
- **Blocked Path**: Invalid moves due to blocking pieces
- **Invalid Move**: Moves that violate piece movement rules

Plus special castling tests:

- **Castling Through Check**: Cannot castle through attacked squares
- **Castling In Check**: Cannot castle while in check

### Temporal Test Categories

- **Movement Tracking**: Identify which piece moved
- **State Comparison**: Compare board states across frames
- **Sequence Understanding**: Understand move sequences
- **Rule Application**: Apply time-dependent rules (en passant timing)
