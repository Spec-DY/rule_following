"""
Run Spatial Test 1 with per-case verification
Each case is verified for board recognition before testing
"""

from src.spatial.test_1_rule_following import SpatialTest1
from src.model_client import DummyModelClient, DashScopeModelClient, NovitaModelClient, ClaudeModelClient
import sys
import os

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))


def main():
    """
    Run Spatial Test 1 with per-case verification

    For each test case:
    1. Ask verification question (e.g., "What pieces are on the board?")
    2. Only if verified, ask the actual test question
    3. Track both verification rate and test accuracy
    """

    print("\n" + "="*60)
    print("SPATIAL TEST 1: RULE FOLLOWING BASELINE")
    print("="*60)

    # ===== Configuration =====

    # Number of cases per piece type (total = 6 * N_CASES_PER_TYPE)
    N_CASES_PER_TYPE = 10
    SEED = 42                  # Random seed for reproducibility
    MODEL_TYPE = "dummy"      # Options: "dummy", "dashscope", "novita"
    DUMMY_VERIFICATION_PASS_RATE = 0.8  # For dummy model

    # ===== Setup Test =====

    test1 = SpatialTest1(
        base_output_dir="./output/spatial_test_1",
        n_cases_per_type=N_CASES_PER_TYPE,
        seed=SEED,
        auto_timestamp=True
    )

    print(f"\nOutput directory: {test1.output_dir}")
    print(f"Configuration:")
    print(f"  - Cases per piece type: {N_CASES_PER_TYPE}")
    print(f"  - Total cases: {N_CASES_PER_TYPE * 6} (6 piece types)")
    print(f"  - Random seed: {SEED}")
    print(f"  - Model: {MODEL_TYPE}")

    # ===== Generate Test Cases =====

    cases = test1.generate_test_cases()
    print(f"\n✓ Generated {len(cases)} test cases")
    print("  (Each with verification question + test question)")

    # Show distribution
    piece_counts = {}
    for case in cases:
        piece_type = case.get('type', 'unknown')
        piece_counts[piece_type] = piece_counts.get(piece_type, 0) + 1

    print("\nTest case distribution by piece type:")
    for piece_type in ['king', 'queen', 'rook', 'bishop', 'knight', 'pawn']:
        if piece_type in piece_counts:
            count = piece_counts[piece_type]
            print(f"  {piece_type.capitalize():10s}: {count:3d} cases")

    # ===== Create Test Images =====

    test1.create_test_images()

    # ===== Setup Model =====

    print(f"{'='*60}")
    print("MODEL SETUP")
    print("="*60)

    if MODEL_TYPE == "dummy":
        model_client = DummyModelClient(
            verification_pass_rate=DUMMY_VERIFICATION_PASS_RATE
        )
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
    else:
        raise ValueError(f"Unknown model type: {MODEL_TYPE}")

    # ===== Provide test cases to Dummy Model (if using dummy) =====

    if MODEL_TYPE == "dummy":
        print(f"\n{'='*60}")
        print("DUMMY MODEL SETUP")
        print("="*60)
        model_client.set_test_cases(test1.test_cases)
        print()

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
            f"  - Overall accuracy (including failures): {stats['test_correct']}/{stats['total']} ({stats['test_correct']/stats['total']:.1%})")
    print(f"\nAll results saved in:")
    print(f"  {test1.output_dir}/")
    print(f"  - test_1_results.json (with verification info)")
    print(f"  - {len(cases)} test images (.png)")

    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    main()
