"""
Run Temporal Levels Tests
Unified script to run any combination of Level 1-6 tests
"""

from src.model_client import DummyModelClient, NovitaModelClient, DashScopeModelClient, XAIModelClient
from src.temporal_levels import TemporalLevel1, TemporalLevel2, TemporalLevel3, TemporalLevel4
import sys
import argparse
from typing import List, Dict, Any
sys.path.append('.')


# ===== Configuration =====
LEVEL_CONFIG = {
    1: {
        "name": "Basic Movement Rules",
        "class": TemporalLevel1,
        "default_cases": 60,
        "description": "Tests basic movement patterns for all 6 piece types"
    },
    2: {
        "name": "Path Blocked Capture",
        "class": TemporalLevel2,
        "default_cases": 90,
        "description": "Tests capture with path blocking (Rook/Bishop/Queen)"
    },
    3: {
        "name": "En Passant Basic",
        "class": TemporalLevel3,
        "default_cases": 100,
        "description": "Tests 3 basic conditions for en passant"
    },
    4: {
        "name": "En Passant Constraints",
        "class": TemporalLevel4,
        "default_cases": 100,
        "description": "Tests en passant timing and check constraints"
    },
    # 5: {
    #     "name": "En Passant Constraints",
    #     "class": TemporalLevel4,
    #     "default_cases": 100,
    #     "description": "Tests en passant timing and check constraints"
    # },
    # 5: {
    #     "name": "Castling + 2 Check Rules",
    #     "class": TemporalLevel5,
    #     "default_cases": 100,
    #     "description": "Tests castling with 2 check rules"
    # },
    # 6: {
    #     "name": "Castling + 3 Check Rules",
    #     "class": TemporalLevel6,
    #     "default_cases": 100,
    #     "description": "Tests all castling violation combinations"
    # },
}


def get_model_client(model_type: str, use_dummy: bool = False, dummy_pass_rate: float = 0.8):
    """
    Get model client based on type

    Args:
        model_type: Type of model client ('dummy', 'novita', 'dashscope', 'xai')
        use_dummy: If True, use dummy model regardless of model_type
        dummy_pass_rate: Pass rate for dummy model

    Returns:
        Model client instance
    """
    if use_dummy or model_type == 'dummy':
        print(f"\n🤖 Using Dummy Model Client (pass_rate={dummy_pass_rate})")
        return DummyModelClient(verification_pass_rate=dummy_pass_rate)

    if model_type == 'novita':
        print("\n🤖 Using Novita Model Client")
        return NovitaModelClient()
    elif model_type == 'dashscope':
        print("\n🤖 Using DashScope Model Client")
        return DashScopeModelClient()
    elif model_type == 'xai':
        print("\n🤖 Using XAI Model Client")
        return XAIModelClient()
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def run_single_level(level: int,
                     n_cases: int = None,
                     seed: int = 42,
                     model_client=None,
                     output_base: str = "./output",
                     rate_limit_requests: int = 0,
                     rate_limit_pause: int = 0) -> Dict[str, Any]:
    """
    Run a single level test

    Args:
        level: Level number (1-6)
        n_cases: Number of test cases (None = use default)
        seed: Random seed
        model_client: Model client instance
        output_base: Base output directory
        rate_limit_requests: Number of requests before pausing
        rate_limit_pause: Pause duration in seconds

    Returns:
        Dictionary with test results and statistics
    """
    if level not in LEVEL_CONFIG:
        raise ValueError(f"Level {level} not implemented yet")

    config = LEVEL_CONFIG[level]
    n_cases = n_cases or config["default_cases"]

    print("\n" + "=" * 70)
    print(f"LEVEL {level}: {config['name']}")
    print(f"Description: {config['description']}")
    print(f"Test cases: {n_cases}")
    print("=" * 70)

    # Initialize test
    test_class = config["class"]
    test = test_class(
        base_output_dir=f"{output_base}/temporal_level_{level}",
        n_cases=n_cases,
        seed=seed,
        auto_timestamp=True,
        rate_limit_requests=rate_limit_requests,
        rate_limit_pause=rate_limit_pause
    )

    # Generate test cases
    test.generate_test_cases()

    # Create images
    test.create_test_images()

    # Set test cases for dummy model
    if isinstance(model_client, DummyModelClient):
        model_client.set_test_cases(test.test_cases)

    # Run test
    results, stats = test.run_test(model_client, save_results_flag=True)

    return {
        "level": level,
        "name": config["name"],
        "results": results,
        "stats": stats,
        "output_dir": test.output_dir
    }


def run_multiple_levels(levels: List[int],
                        n_cases: int = None,
                        seed: int = 42,
                        model_type: str = "dummy",
                        use_dummy: bool = False,
                        dummy_pass_rate: float = 0.8,
                        output_base: str = "./output",
                        rate_limit_requests: int = 0,
                        rate_limit_pause: int = 0) -> List[Dict[str, Any]]:
    """
    Run multiple level tests

    Args:
        levels: List of level numbers to run
        n_cases: Number of test cases per level (None = use defaults)
        seed: Random seed
        model_type: Type of model client
        use_dummy: Force use of dummy model
        dummy_pass_rate: Pass rate for dummy model
        output_base: Base output directory
        rate_limit_requests: Number of requests before pausing
        rate_limit_pause: Pause duration in seconds

    Returns:
        List of result dictionaries
    """
    # Validate levels
    for level in levels:
        if level not in LEVEL_CONFIG:
            print(
                f"⚠️  Warning: Level {level} not implemented yet, skipping...")
            levels.remove(level)

    if not levels:
        print("❌ No valid levels to run!")
        return []

    print("\n" + "=" * 70)
    print("TEMPORAL LEVELS TEST SUITE")
    print("=" * 70)
    print(f"Levels to run: {levels}")
    print(f"Random seed: {seed}")
    print(f"Output directory: {output_base}")
    if rate_limit_requests > 0:
        print(
            f"Rate limiting: {rate_limit_requests} requests, {rate_limit_pause}s pause")
    print("=" * 70)

    # Initialize model client (shared across all levels)
    model_client = get_model_client(model_type, use_dummy, dummy_pass_rate)

    # Run each level
    all_results = []
    for level in levels:
        try:
            result = run_single_level(
                level=level,
                n_cases=n_cases,
                seed=seed,
                model_client=model_client,
                output_base=output_base,
                rate_limit_requests=rate_limit_requests,
                rate_limit_pause=rate_limit_pause
            )
            all_results.append(result)
        except Exception as e:
            print(f"\n❌ Error running Level {level}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Print summary
    print_summary(all_results)

    return all_results


def print_summary(all_results: List[Dict[str, Any]]):
    """Print summary of all test results"""
    print("\n" + "=" * 70)
    print("SUMMARY OF ALL LEVELS")
    print("=" * 70)

    for result in all_results:
        level = result["level"]
        name = result["name"]
        stats = result["stats"]

        # Check if this is the new stats format (with verification)
        if 'verification_passed' in stats:
            verification_rate = stats['verification_passed'] / \
                stats['total'] if stats['total'] > 0 else 0
            accuracy_given_verified = stats['test_correct_given_verified'] / \
                stats['verification_passed'] if stats['verification_passed'] > 0 else 0
            overall_accuracy = stats['test_correct'] / \
                stats['total'] if stats['total'] > 0 else 0

            print(f"\nLevel {level}: {name}")
            print(f"  Total cases: {stats['total']}")
            print(f"  Verification rate: {verification_rate:.1%}")
            print(
                f"  Accuracy (verified cases): {accuracy_given_verified:.1%}")
            print(f"  Overall accuracy: {overall_accuracy:.1%}")
        else:
            # Old stats format (no verification)
            accuracy = stats['correct'] / \
                stats['total'] if stats['total'] > 0 else 0
            print(f"\nLevel {level}: {name}")
            print(f"  Total cases: {stats['total']}")
            print(f"  Correct: {stats['correct']} ({accuracy:.1%})")

        print(f"  Output: {result['output_dir']}")

    print("\n" + "=" * 70)
    print("✅ All tests completed!")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Run Temporal Levels Tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run Level 1 only
  python run/run_temporal_levels.py -l 1
  
  # Run Levels 1, 2, 3
  python run/run_temporal_levels.py -l 1 2 3
  
  # Run all levels
  python run/run_temporal_levels.py --all
  
  # Run with custom number of cases
  python run/run_temporal_levels.py -l 1 2 -n 50
  
  # Run with real model
  python run/run_temporal_levels.py -l 1 --model novita
  
  # Run with rate limiting
  python run/run_temporal_levels.py --all --rate-limit 20 --rate-pause 5
        """
    )

    # Level selection
    level_group = parser.add_mutually_exclusive_group(required=True)
    level_group.add_argument(
        "-l", "--levels",
        type=int,
        nargs="+",
        help="Levels to run (e.g., -l 1 2 3)"
    )
    level_group.add_argument(
        "--all",
        action="store_true",
        help="Run all available levels"
    )

    # Test configuration
    parser.add_argument(
        "-n", "--n-cases",
        type=int,
        default=None,
        help="Number of test cases per level (default: use level defaults)"
    )
    parser.add_argument(
        "-s", "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="./output",
        help="Output directory (default: ./output)"
    )

    # Model selection
    parser.add_argument(
        "-m", "--model",
        type=str,
        choices=["dummy", "novita", "dashscope", "xai"],
        default="dummy",
        help="Model type to use (default: dummy)"
    )
    parser.add_argument(
        "--dummy-pass-rate",
        type=float,
        default=0.8,
        help="Pass rate for dummy model (default: 0.8)"
    )

    # Rate limiting
    parser.add_argument(
        "--rate-limit",
        type=int,
        default=0,
        help="Number of requests before pausing (0 = no limit)"
    )
    parser.add_argument(
        "--rate-pause",
        type=int,
        default=0,
        help="Seconds to pause when rate limit reached"
    )

    args = parser.parse_args()

    # Determine which levels to run
    if args.all:
        levels = sorted(LEVEL_CONFIG.keys())
    else:
        levels = sorted(args.levels)

    # Run tests
    run_multiple_levels(
        levels=levels,
        n_cases=args.n_cases,
        seed=args.seed,
        model_type=args.model,
        use_dummy=(args.model == "dummy"),
        dummy_pass_rate=args.dummy_pass_rate,
        output_base=args.output,
        rate_limit_requests=args.rate_limit,
        rate_limit_pause=args.rate_pause
    )


if __name__ == "__main__":
    main()
