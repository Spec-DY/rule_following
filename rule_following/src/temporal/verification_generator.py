"""
Generate verification questions for temporal test cases
To ensure the model can correctly recognize each board state before testing
"""

from typing import Dict, List


class TemporalVerificationGenerator:
    """Generate verification questions for temporal test cases"""

    @staticmethod
    def generate_verification(case: Dict) -> Dict:
        """
        Generate a verification question for a temporal test case

        Args:
            case: Test case dictionary with multiple states

        Returns:
            Dictionary with verification_question and verification_expected
        """
        case_type = case.get('type', 'unknown')
        states = case.get('states', [])

        # For event recognition tests (Type 2) with multiple choice
        if 'options' in case:
            return TemporalVerificationGenerator._verify_event_recognition(case)

        # For rule judgment tests (Type 1) and direct movement (Type 3)
        elif case_type in ['en_passant_rule', 'castling_rule', 'direct_movement']:
            return TemporalVerificationGenerator._verify_rule_judgment(case)

        # Default fallback
        else:
            return TemporalVerificationGenerator._verify_all_states(states)

    @staticmethod
    def _verify_event_recognition(case: Dict) -> Dict:
        """
        Verify for event recognition (multiple choice) cases
        Ask about specific content in ALL states
        """
        states = case.get('states', [])

        # Ask about all states
        all_states_desc = []
        all_keywords = []

        for i, state in enumerate(states):
            pieces = state.get('pieces', {})
            pieces_desc = []
            for sq, piece in pieces.items():
                piece_name = TemporalVerificationGenerator._piece_name(piece)
                pieces_desc.append(f"{piece_name} at {sq}")
                # ✅ 添加位置和棋子类型到关键词
                all_keywords.append(sq)
                all_keywords.append(
                    TemporalVerificationGenerator._get_piece_type(piece))

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
    def _verify_rule_judgment(case: Dict) -> Dict:
        """
        Verify for rule judgment cases
        Ask about pieces and their positions in ALL states
        """
        states = case.get('states', [])

        if len(states) == 2:
            return TemporalVerificationGenerator._verify_two_states(states)
        elif len(states) == 3:
            return TemporalVerificationGenerator._verify_three_states(states)
        else:
            return TemporalVerificationGenerator._verify_all_states(states)

    @staticmethod
    def _verify_two_states(states: List[Dict]) -> Dict:
        """Verify 2 states - ask about pieces and positions in BOTH states"""
        all_states_desc = []
        all_keywords = []

        for i, state in enumerate(states):
            pieces = state.get('pieces', {})
            pieces_desc = []
            for sq, piece in pieces.items():
                piece_name = TemporalVerificationGenerator._piece_name(piece)
                pieces_desc.append(f"{piece_name} at {sq}")
                # ✅ 添加位置和棋子类型
                all_keywords.append(sq)
                all_keywords.append(
                    TemporalVerificationGenerator._get_piece_type(piece))

            state_text = ", ".join(
                pieces_desc) if pieces_desc else "Empty board"
            all_states_desc.append(f"State {i+1}: {state_text}")

        expected = "; ".join(all_states_desc)

        return {
            'verification_question': "What are the pieces and their positions in State 1? What about State 2?",
            'verification_expected': expected,
            'verification_keywords': all_keywords
        }

    @staticmethod
    def _verify_three_states(states: List[Dict]) -> Dict:
        """Verify 3 states - ask about pieces and positions in ALL 3 states"""
        all_states_desc = []
        all_keywords = []

        for i, state in enumerate(states):
            pieces = state.get('pieces', {})
            pieces_desc = []
            for sq, piece in pieces.items():
                piece_name = TemporalVerificationGenerator._piece_name(piece)
                pieces_desc.append(f"{piece_name} at {sq}")
                # ✅ 添加位置和棋子类型
                all_keywords.append(sq)
                all_keywords.append(
                    TemporalVerificationGenerator._get_piece_type(piece))

            state_text = ", ".join(
                pieces_desc) if pieces_desc else "Empty board"
            all_states_desc.append(f"State {i+1}: {state_text}")

        expected = "; ".join(all_states_desc)

        return {
            'verification_question': "What are the pieces and their positions in State 1? State 2? State 3?",
            'verification_expected': expected,
            'verification_keywords': all_keywords
        }

    @staticmethod
    def _verify_all_states(states: List[Dict]) -> Dict:
        """
        Verify by asking about ALL states' content
        This is the universal fallback - asks about pieces in ALL states
        """
        if not states:
            return {
                'verification_question': "Can you see chess board states in these images?",
                'verification_expected': "yes",
                'verification_keywords': ['yes', 'board']
            }

        all_states_desc = []
        all_keywords = []

        for i, state in enumerate(states):
            pieces = state.get('pieces', {})
            pieces_desc = []
            for sq, piece in pieces.items():
                piece_name = TemporalVerificationGenerator._piece_name(piece)
                pieces_desc.append(f"{piece_name} at {sq}")
                # ✅ 添加位置和棋子类型到关键词
                all_keywords.append(sq)
                all_keywords.append(
                    TemporalVerificationGenerator._get_piece_type(piece))

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
