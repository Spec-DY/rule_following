"""
Run Temporal Levels Tests
Unified script to run any combination of Level 1-6 tests
"""

from src.model_client import DummyModelClient, NovitaModelClient, DashScopeModelClient, XAIModelClient
from src.temporal_levels import TemporalLevel1, TemporalLevel2, TemporalLevel3, TemporalLevel4, TemporalLevel5
import sys
import argparse
import os
import json
from datetime import datetime
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
    5: {
        "name": "Castling with Check Constraints",
        "class": TemporalLevel5,
        "default_cases": 100,
        "description": "Tests castling legality regarding check constraints (in, through, into)"},
}


def get_model_client(model_type: str, use_dummy: bool = False, dummy_pass_rate: float = 0.8):
    """
    Get model client based on type
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
        "stats": stats,
        "output_dir": test.output_dir,
        "model_name": model_client.model_name
    }


def save_suite_summary(all_results: List[Dict[str, Any]], output_base: str):
    """
    Save a summary of all levels to a JSON file
    """
    os.makedirs(output_base, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_data = {
        "timestamp": datetime.now().isoformat(),
        "model_name": all_results[0]["model_name"] if all_results else "unknown",
        "levels_run": len(all_results),
        "results": []
    }

    print("\n" + "=" * 70)
    print("SUMMARY OF ALL LEVELS")
    print("=" * 70)

    for res in all_results:
        level = res["level"]
        name = res["name"]
        stats = res["stats"]

        # Calculate accuracy metrics
        verification_rate = stats['verification_passed'] / \
            stats['total'] if stats['total'] > 0 else 0
        accuracy_verified = stats['test_correct_given_verified'] / \
            stats['verification_passed'] if stats['verification_passed'] > 0 else 0
        overall_accuracy = stats['test_correct'] / \
            stats['total'] if stats['total'] > 0 else 0

        # Add to summary data
        level_summary = {
            "level": level,
            "name": name,
            "total_cases": stats['total'],
            "verification_rate": round(verification_rate, 3),
            "accuracy_given_verified": round(accuracy_verified, 3),
            "overall_accuracy": round(overall_accuracy, 3),
            "output_dir": res["output_dir"],
            "details": stats  # Include full raw stats
        }
        summary_data["results"].append(level_summary)

        # Print to console
        print(f"\nLevel {level}: {name}")
        print(f"  Total cases: {stats['total']}")
        print(f"  Verification rate: {verification_rate:.1%}")
        print(f"  Accuracy (verified cases): {accuracy_verified:.1%}")
        print(f"  Overall accuracy: {overall_accuracy:.1%}")
        print(f"  Output: {res['output_dir']}")

    # Save to file
    filename = f"temporal_levels_summary_{timestamp}.json"
    filepath = os.path.join(output_base, filename)

    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)
        print("\n" + "=" * 70)
        print(f"✅ All tests completed!")
        print(f"📄 Suite summary saved to: {filepath}")
        print("=" * 70)
    except Exception as e:
        print(f"\n⚠️ Failed to save summary file: {e}")


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

    # Save summary to file and print
    save_suite_summary(all_results, output_base)

    return all_results


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
