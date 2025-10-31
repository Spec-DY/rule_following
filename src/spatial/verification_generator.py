"""
Generate verification questions for each test case
To ensure the model can correctly recognize the board before testing
"""

from typing import Dict, List


class VerificationQuestionGenerator:
    """Generate verification questions for test cases"""

    @staticmethod
    def generate_verification(case: Dict) -> Dict:
        """
        Generate a verification question for a test case

        Args:
            case: Test case dictionary

        Returns:
            Dictionary with verification_question and verification_expected
        """
        case_type = case.get('type', 'unknown')
        squares = case.get('squares', [])
        pieces = case.get('pieces', {})

        # Choose verification strategy based on case type
        if case_type in ['same_line', 'diagonal', 'relative_position']:
            # For cases with highlighted squares, ask what's highlighted
            return VerificationQuestionGenerator._verify_highlighted_squares(squares)

        elif case_type == 'distance':
            # For distance cases (3 squares), verify all three
            return VerificationQuestionGenerator._verify_highlighted_squares(squares)

        elif case_type == 'path_clear':
            # For path cases, verify both squares and pieces
            return VerificationQuestionGenerator._verify_pieces_and_squares(squares, pieces)

        else:
            # Default: verify highlighted squares
            return VerificationQuestionGenerator._verify_highlighted_squares(squares)

    @staticmethod
    def _verify_highlighted_squares(squares: List[str]) -> Dict:
        """Ask model to identify highlighted squares"""
        if len(squares) == 0:
            return {
                'verification_question': "Can you see a chess board in this image?",
                'verification_expected': "yes",
                'verification_keywords': ['yes', 'chess', 'board']
            }

        elif len(squares) == 1:
            return {
                'verification_question': f"What square is highlighted on this board? (Answer with the square name only, e.g., 'a1')",
                'verification_expected': squares[0],
                'verification_keywords': [squares[0]]
            }

        elif len(squares) == 2:
            return {
                'verification_question': f"What two squares are highlighted on this board? (List them separated by space or comma, e.g., 'a1 a2')",
                'verification_expected': f"{squares[0]} and {squares[1]}",
                'verification_keywords': squares  # Both must appear
            }

        else:  # 3 or more
            squares_str = ", ".join(squares[:-1]) + " and " + squares[-1]
            return {
                'verification_question': f"What squares are highlighted on this board? (List all of them)",
                'verification_expected': squares_str,
                'verification_keywords': squares  # All must appear
            }

    @staticmethod
    def _verify_pieces_and_squares(squares: List[str], pieces: Dict[str, str]) -> Dict:
        """Ask model to identify both pieces and squares"""
        if len(pieces) == 0:
            # No pieces, just verify squares
            return VerificationQuestionGenerator._verify_highlighted_squares(squares)

        elif len(pieces) == 1:
            piece_sq = list(pieces.keys())[0]
            return {
                'verification_question': f"I see a piece on this board. What square is it on? (Answer with square name only)",
                'verification_expected': piece_sq,
                'verification_keywords': [piece_sq]
            }

        else:
            # Multiple pieces
            piece_squares = list(pieces.keys())
            return {
                'verification_question': f"How many pieces are on this board?",
                'verification_expected': str(len(pieces)),
                'verification_keywords': [str(len(pieces)), f"{len(pieces)}"]
            }

    @staticmethod
    def check_verification_answer(response: str, verification_info: Dict) -> bool:
        """
        Check if the model's verification response is correct

        Args:
            response: Model's response to verification question
            verification_info: Dictionary with expected answer and keywords

        Returns:
            True if verification passed, False otherwise
        """
        response_lower = response.lower().strip()

        # Remove common punctuation
        response_clean = response_lower.replace(".", "").replace(
            ",", "").replace("!", "").replace("?", "")

        keywords = verification_info.get('verification_keywords', [])

        # All keywords must appear in response
        for keyword in keywords:
            keyword_lower = keyword.lower()
            if keyword_lower not in response_clean:
                return False

        return True
