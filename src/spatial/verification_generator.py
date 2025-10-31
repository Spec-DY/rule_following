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

        # Test 0 case types: spatial reasoning without chess rules
        if case_type in ['same_line', 'diagonal', 'relative_position']:
            return VerificationQuestionGenerator._verify_highlighted_squares(squares)

        elif case_type == 'distance':
            return VerificationQuestionGenerator._verify_highlighted_squares(squares)

        elif case_type == 'path_clear':
            return VerificationQuestionGenerator._verify_pieces_and_squares(squares, pieces)

        # Test 1 case types: chess piece movement rules
        elif case_type in ['king', 'queen', 'rook', 'bishop', 'knight', 'pawn']:
            # For chess pieces, verify piece location and target square
            return VerificationQuestionGenerator._verify_piece_location(squares, pieces)

        # Default: verify highlighted squares
        else:
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
            return {
                'verification_question': f"How many pieces are on this board?",
                'verification_expected': str(len(pieces)),
                'verification_keywords': [str(len(pieces)), f"{len(pieces)}"]
            }

    @staticmethod
    def _verify_piece_location(squares: List[str], pieces: Dict[str, str]) -> Dict:
        """
        For chess piece movement tests: verify the piece location

        Args:
            squares: [from_square, to_square]
            pieces: {square: piece_type}
        """
        if len(pieces) == 0 or len(squares) == 0:
            # Fallback to squares only
            return VerificationQuestionGenerator._verify_highlighted_squares(squares)

        # Get the piece square (usually the first square in the list)
        piece_square = list(pieces.keys())[0]

        # Use different verification strategies based on number of pieces
        if len(pieces) == 1:
            # Single piece: ask where it is
            return {
                'verification_question': f"I see one chess piece on this board. What square is it on? (Answer with square name only, e.g., 'e4')",
                'verification_expected': piece_square,
                'verification_keywords': [piece_square]
            }
        elif len(pieces) == 2:
            # Two pieces (e.g., blocking piece): ask about both
            piece_squares = list(pieces.keys())
            return {
                'verification_question': f"I see pieces on this board. What squares are they on? (List them)",
                'verification_expected': f"{piece_squares[0]} and {piece_squares[1]}",
                'verification_keywords': piece_squares
            }
        else:
            # Multiple pieces: ask count
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
            ",", "").replace("!", "").replace("?", "").replace("'", "")

        keywords = verification_info.get('verification_keywords', [])

        # All keywords must appear in response
        for keyword in keywords:
            keyword_lower = keyword.lower()
            if keyword_lower not in response_clean:
                return False

        return True
