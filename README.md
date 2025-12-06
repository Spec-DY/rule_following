# Why VLMs Fail?

Diagnostic tests for understanding why VLMs fail in game scenarios.

## Overview
<img width="8007" height="5405" alt="Poster" src="https://github.com/user-attachments/assets/0df641f3-4d1c-4b18-88e4-324df244bb78" />


This repository contains two complementary test suites that systematically evaluate VLM capabilities:

| Module | Focus |
|--------|-------|
| [**Perception**](./perception/README.md) | Visual perception accuracy
| [**Rule Following**](./rule_following/README.md) | Spatial & temporal reasoning

## Test Suites

### Perception Tests

Evaluates how visual factors affect VLM perception accuracy:

- **Patch Alignment** - Aligned vs offset image patches
- **Resolution** - Image resolution & patch divisibility  
- **Visual Richness** - 2D flat vs 3D rendered styles
- **Density** - Low/medium/high piece density

👉 [View detailed documentation](./perception/README.md)

### Rule Following Tests

Evaluates spatial and temporal reasoning with chess rules:

- **Spatial Test 0** - Pure spatial reasoning
- **Spatial Test 1** - Chess movement rules for all piece types
- **Temporal Test 0** - Sequence understanding & state tracking
- **Temporal Test 1** - Time-dependent rules
- **Condition Ladder** - Accuracy with increasing conditions

👉 [View detailed documentation](./rule_following/README.md)

## Quick Start

### Installation

```bash
git clone https://github.com/Spec-DY/why-vlms-fail.git
cd vlm_benchmark

# Install rule_following module
cd rule_following
pip install -e .
cd ..

# Install perception dependencies
cd perception
pip install -r requirements.txt
cd ..
```

### Results
Download from this [google drive link](https://drive.google.com/drive/folders/1jPYZdfO9Z5y_qtnBkqfkC3va7FG4dlNv?usp=sharing) to view our results. 

### Configuration

Create `.env` files in each module directory with your API keys. See individual READMEs for details.

### Running Tests

```bash
# Perception tests
cd perception
python tests/density/run_chess_density_tests.py

# Rule following tests
cd rule_following
python run/run_spatial_test_0.py
```
