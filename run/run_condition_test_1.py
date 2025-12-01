"""
Run Condition Test 1
Tests threat counting - count how many pieces can attack a target
"""

from src.model_client import DummyModelClient, DashScopeModelClient, NovitaModelClient, XAIModelClient
from src.condition.condition_test_1 import ConditionTest1
import sys
import os

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))


def main():
    """
    Run Condition Test 1

    Tests ability to count multiple threats:
    - Level 1: Count threats from 1 potential attacker
    - Level 2: Count threats from 2 potential attackers
    - ...
    - Level 6: Count threats from 6 potential attackers
    """

    print("\n" + "="*60)
    print("CONDITION TEST 1: THREAT COUNT")
    print("="*60)

    # ===== Configuration =====

    N_CASES_PER_LEVEL = 3      # Number of cases per level (1-6)
    SEED = 57                    # Random seed for reproducibility
    MODEL_TYPE = "dashscope"        # Options: "dummy", "dashscope", "novita", "xai"
    RATE_LIMIT_REQUESTS = 0      # Number of requests before pausing
    RATE_LIMIT_PAUSE = 0         # Pause duration in seconds

    # ===== Setup Test =====

    test1 = ConditionTest1(
        base_output_dir="./output/condition_test_1",
        n_cases_per_level=N_CASES_PER_LEVEL,
        seed=SEED,
        auto_timestamp=True,
        rate_limit_requests=RATE_LIMIT_REQUESTS,
        rate_limit_pause=RATE_LIMIT_PAUSE
    )

    print(f"\nOutput directory: {test1.output_dir}")
    print(f"Configuration:")
    print(f"  - Cases per level: {N_CASES_PER_LEVEL}")
    print(f"  - Total cases: {N_CASES_PER_LEVEL * 6}")
    print(f"  - Random seed: {SEED}")
    print(f"  - Model: {MODEL_TYPE}")

    # ===== Generate Test Cases =====

    cases = test1.generate_test_cases()
    print(f"\n✓ Generated {len(cases)} test cases")
    print("  (Each with verification question + test question)")

    # Show distribution
    level_counts = {}
    answer_dist = {}

    for case in cases:
        level = case.get('level', 0)
        level_counts[level] = level_counts.get(level, 0) + 1

        # Track answer distribution
        answer = case.get('expected', '0')
        key = f"L{level}_A{answer}"
        answer_dist[key] = answer_dist.get(key, 0) + 1

    print("\nTest case distribution:")
    for level in sorted(level_counts.keys()):
        count = level_counts[level]
        print(
            f"  Level {level} ({level} potential attackers): {count:3d} cases")

        # Show answer distribution for this level
        level_answers = {k: v for k, v in answer_dist.items()
                         if k.startswith(f"L{level}_")}
        if level_answers:
            answer_summary = ", ".join(
                [f"{k.split('_')[1]}: {v}" for k, v in sorted(level_answers.items())])
            print(f"    Answer distribution: {answer_summary}")

    # ===== Create Test Images =====

    test1.create_test_images()
    # ===== Setup Model =====

    print(f"{'='*60}")
    print("MODEL SETUP")
    print("="*60)

    if MODEL_TYPE == "dummy":
        model_client = DummyModelClient()
        print("✓ Using Dummy Model (random answers)\n")

    elif MODEL_TYPE == "dashscope":
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise ValueError("DASHSCOPE_API_KEY not found")
        model_client = DashScopeModelClient(api_key=api_key)
        print(f"✓ Using DashScope: {model_client.model_name}\n")

    elif MODEL_TYPE == "novita":
        api_key = os.getenv("NOVITA_API_KEY")
        if not api_key:
            raise ValueError("NOVITA_API_KEY not found")
        model_client = NovitaModelClient(api_key=api_key, stream=False)
        print(f"✓ Using Novita: {model_client.model_name}\n")

    elif MODEL_TYPE == "xai":
        api_key = os.getenv("XAI_API_KEY")
        if not api_key:
            raise ValueError("XAI_API_KEY not found")
        model_client = XAIModelClient(api_key=api_key, stream=False)
        print(f"✓ Using XAI: {model_client.model_name}\n")

    else:
        raise ValueError(f"Unknown model type: {MODEL_TYPE}")

    # ===== Provide test cases to Dummy Model (if using dummy) =====

    if MODEL_TYPE == "dummy":
        print(f"\n{'='*60}")
        print("DUMMY MODEL SETUP")
        print("="*60)
        model_client.set_test_cases(test1.test_cases)
        print("✓ Test cases provided to Dummy Model\n")

    # ===== Run Test =====

    results, stats = test1.run_test(model_client)

    # ===== Summary =====

    print(f"✓ Test completed!")
    print(f"\nKey Insights:")
    print(f"  - Total test cases: {stats['total']}")
    print(
        f"  - Board recognition rate: {stats['verification_passed']}/{stats['total']} ({stats['verification_passed']/stats['total']:.1%})")

    if stats['verification_passed'] > 0:
        acc = stats['test_correct_given_verified'] / \
            stats['verification_passed']
        print(f"  - Test accuracy (when recognized): {acc:.1%}")
        print(
            f"  - Overall accuracy: {stats['test_correct']}/{stats['total']} ({stats['test_correct']/stats['total']:.1%})")

    print(f"\nAll results saved in:")
    print(f"  {test1.output_dir}/")
    print(f"  - test_1_results.json (with verification info)")

    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    main()
