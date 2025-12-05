"""
Generate verification questions for temporal level test cases
To ensure the model can correctly recognize each board state before testing
"""

from typing import Dict, List


class TemporalLevelVerificationGenerator:
    """Generate verification questions for temporal level test cases"""

    @staticmethod
    def generate_verification(case: Dict) -> Dict:
        """
        Generate a verification question for a temporal level test case

        Args:
            case: Test case dictionary with multiple states

        Returns:
            Dictionary with verification_question, verification_expected, verification_keywords
        """
        states = case.get('states', [])

        # Ask about all states
        all_states_desc = []
        all_keywords = []

        for i, state in enumerate(states):
            pieces = state.get('pieces', {})
            pieces_desc = []
            for sq, piece in pieces.items():
                piece_name = TemporalLevelVerificationGenerator._piece_name(
                    piece)
                pieces_desc.append(f"{piece_name} at {sq}")
                # Add position and piece type to keywords
                all_keywords.append(sq)
                all_keywords.append(
                    TemporalLevelVerificationGenerator._get_piece_type(piece))

            state_text = ", ".join(
                pieces_desc) if pieces_desc else "Empty board"
            all_states_desc.append(f"State {i+1}: {state_text}")

        expected = "; ".join(all_states_desc)

        # Generate question based on number of states
        if len(states) == 2:
            question = "What are the pieces and their positions in State 1? What about State 2?"
        elif len(states) == 3:
            question = "What are the pieces and their positions in State 1? State 2? State 3?"
        elif len(states) == 4:
            question = "What are the pieces and their positions in State 1? State 2? State 3? State 4?"
        else:
            question = f"What are the pieces and their positions in each of the {len(states)} states?"

        return {
            'verification_question': question,
            'verification_expected': expected,
            'verification_keywords': all_keywords
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
        """
        Get piece type only (e.g., 'P' -> 'pawn', 'N' -> 'knight')
        Used for verification keywords
        """
        piece_type_map = {
            'K': 'king',
            'Q': 'queen',
            'R': 'rook',
            'B': 'bishop',
            'N': 'knight',
            'P': 'pawn',
            'k': 'king',
            'q': 'queen',
            'r': 'rook',
            'b': 'bishop',
            'n': 'knight',
            'p': 'pawn',
        }
        return piece_type_map.get(piece_symbol, 'unknown')

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
