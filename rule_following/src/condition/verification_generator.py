"""
Generate verification questions for condition test cases
To ensure the model can correctly recognize the board before testing
"""

from typing import Dict, List


class ConditionVerificationGenerator:
    """Generate verification questions for condition test cases"""

    @staticmethod
    def generate_verification(case: Dict) -> Dict:
        """
        Generate a verification question for a condition test case

        Args:
            case: Test case dictionary

        Returns:
            Dictionary with verification_question, verification_expected, and verification_keywords
        """
        pieces = case.get('pieces', {})
        target_square = case.get('target_square', '')

        # For Type 1, we just need to verify they can see the board correctly
        # Since all 3 images are identical, we ask about the board content once

        # Generate piece descriptions for all pieces
        pieces_desc = []
        keywords = []

        for sq, piece in sorted(pieces.items()):
            piece_name = ConditionVerificationGenerator._piece_name(piece)
            pieces_desc.append(f"{piece_name} at {sq}")

            # Add keywords: square + piece type + color
            keywords.append(sq.lower())
            keywords.append(
                ConditionVerificationGenerator._get_piece_type(piece))
            keywords.append(
                ConditionVerificationGenerator._get_piece_color(piece))

        expected = ", ".join(pieces_desc)

        # Generate question - simple and clear
        question = "What pieces do you see on the board and where are they located?"

        return {
            'verification_question': question,
            'verification_expected': expected,
            'verification_keywords': keywords
        }

    @staticmethod
    def _piece_name(piece_symbol: str) -> str:
        """Convert piece symbol to full name (e.g., 'P' -> 'White Pawn')"""
        piece_map = {
            'K': 'White King',
            'Q': 'White Queen',
            'R': 'White Rook',
            'B': 'White Bishop',
            'N': 'White Knight',
            'P': 'White Pawn',
            'k': 'Black King',
            'q': 'Black Queen',
            'r': 'Black Rook',
            'b': 'Black Bishop',
            'n': 'Black Knight',
            'p': 'Black Pawn',
        }
        return piece_map.get(piece_symbol, 'Unknown')

    @staticmethod
    def _get_piece_type(piece_symbol: str) -> str:
        """Get piece type only (e.g., 'P' -> 'pawn', 'N' -> 'knight')"""
        piece_type_map = {
            'K': 'king', 'Q': 'queen', 'R': 'rook',
            'B': 'bishop', 'N': 'knight', 'P': 'pawn',
            'k': 'king', 'q': 'queen', 'r': 'rook',
            'b': 'bishop', 'n': 'knight', 'p': 'pawn',
        }
        return piece_type_map.get(piece_symbol, 'unknown')

    @staticmethod
    def _get_piece_color(piece_symbol: str) -> str:
        """Get piece color (e.g., 'P' -> 'white', 'p' -> 'black')"""
        if piece_symbol.isupper():
            return 'white'
        else:
            return 'black'

    @staticmethod
    def check_verification_answer(response: str, case: Dict) -> bool:
        """
        Check if the model's verification response is correct

        Args:
            response: Model's response to verification question
            case: Test case dictionary with verification info

        Returns:
            True if verification passed, False otherwise
        """
        response_lower = response.lower().strip()

        # Remove common punctuation
        response_clean = response_lower.replace(".", "").replace(
            ",", "").replace("!", "").replace("?", "").replace("'", "")

        keywords = case.get('verification_keywords', [])

        # All keywords must appear in response
        for keyword in keywords:
            keyword_lower = str(keyword).lower()
            if keyword_lower not in response_clean:
                return False

        return True
