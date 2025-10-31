"""
Model client interfaces for querying VLMs
"""

from abc import ABC, abstractmethod
from typing import Optional
import base64
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class ModelClient(ABC):
    """Base class for model clients"""

    def __init__(self, model_name: str = "test_model"):
        self.model_name = model_name

    @abstractmethod
    def query(self, prompt: str, image_path: str) -> str:
        """
        Query the model with text and image

        Args:
            prompt: Text prompt
            image_path: Path to image file

        Returns:
            Model response as string
        """
        pass


class OpenAICompatibleModelClient(ModelClient):
    """
    Base class for OpenAI-compatible API clients
    Supports any service that uses OpenAI's API format
    """

    # Subclasses should override these
    DEFAULT_BASE_URL = None
    ENV_API_KEY = None
    ENV_BASE_URL = None
    ENV_MODEL = None
    SERVICE_NAME = "OpenAI-compatible"

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize OpenAI-compatible client

        Args:
            api_key: API key (if None, reads from environment variable)
            base_url: API base URL (if None, reads from environment variable)
            model_name: Model name (if None, reads from environment variable)
            **kwargs: Additional parameters to pass to the API
        """
        # Get configuration from env vars if not provided
        self.api_key = api_key or os.getenv(self.ENV_API_KEY)
        self.base_url = base_url or os.getenv(
            self.ENV_BASE_URL, self.DEFAULT_BASE_URL)
        model = model_name or os.getenv(self.ENV_MODEL)

        # Validate required parameters
        if not self.api_key:
            raise ValueError(
                f"{self.SERVICE_NAME} API key not found. "
                f"Please set {self.ENV_API_KEY} environment variable or pass api_key parameter."
            )

        if not model:
            raise ValueError(
                f"{self.SERVICE_NAME} model name not found. "
                f"Please set {self.ENV_MODEL} environment variable or pass model_name parameter."
            )

        if not self.base_url:
            raise ValueError(
                f"{self.SERVICE_NAME} base URL not found. "
                f"Please set {self.ENV_BASE_URL} environment variable or pass base_url parameter."
            )

        super().__init__(model_name=model)

        # Store any additional parameters
        self.extra_params = kwargs

        # Initialize OpenAI client
        try:
            from openai import OpenAI
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
            print(f"✓ {self.SERVICE_NAME} client initialized")
            print(f"  Model: {self.model_name}")
            if self.extra_params:
                print(f"  Extra params: {self.extra_params}")
        except ImportError:
            raise ImportError(
                "openai package is required. Install it with: pip install 'openai>=1.0.0'"
            )

    def query(self, prompt: str, image_path: str) -> str:
        """
        Call OpenAI-compatible API with image

        Args:
            prompt: Text prompt
            image_path: Path to image file

        Returns:
            Model response
        """
        # Read and encode image as base64
        with open(image_path, "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode("utf-8")

        # Determine image media type
        if image_path.lower().endswith('.png'):
            media_type = "image/png"
        elif image_path.lower().endswith(('.jpg', '.jpeg')):
            media_type = "image/jpeg"
        else:
            media_type = "image/png"  # default

        # Construct data URL
        image_url = f"data:{media_type};base64,{image_data}"

        try:
            # Prepare base parameters
            params = {
                "model": self.model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": image_url
                                }
                            },
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }
                ]
            }

            # Add any extra parameters if provided
            params.update(self.extra_params)

            # Call API
            chat_completion_res = self.client.chat.completions.create(**params)

            # Handle streaming vs non-streaming responses
            if self.extra_params.get('stream', False):
                # Collect streamed response
                response_text = ""
                for chunk in chat_completion_res:
                    if chunk.choices[0].delta.content:
                        response_text += chunk.choices[0].delta.content
                return response_text
            else:
                # Return non-streaming response
                return chat_completion_res.choices[0].message.content

        except Exception as e:
            raise Exception(f"{self.SERVICE_NAME} API call failed: {str(e)}")


class NovitaModelClient(OpenAICompatibleModelClient):
    """Novita AI model client"""

    DEFAULT_BASE_URL = "https://api.novita.ai/openai"
    ENV_API_KEY = "NOVITA_API_KEY"
    ENV_BASE_URL = "NOVITA_BASE_URL"
    ENV_MODEL = "NOVITA_MODEL"
    SERVICE_NAME = "Novita"


class DashScopeModelClient(OpenAICompatibleModelClient):
    """Alibaba DashScope model client"""

    DEFAULT_BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
    ENV_API_KEY = "DASHSCOPE_API_KEY"
    ENV_BASE_URL = "DASHSCOPE_BASE_URL"
    ENV_MODEL = "DASHSCOPE_MODEL"
    SERVICE_NAME = "DashScope"


class DummyModelClient(ModelClient):
    """Dummy model client for testing the framework"""

    def __init__(self, verification_pass_rate: float = 0.7):
        """
        Initialize dummy model

        Args:
            verification_pass_rate: Probability of passing verification (0.0 to 1.0)
        """
        super().__init__(model_name="dummy_model")
        self.verification_pass_rate = verification_pass_rate
        self.test_cases_lookup = {}  # image_path -> case info

    def set_test_cases(self, test_cases: list):
        """
        Provide test cases to dummy model so it can "know" correct answers

        Args:
            test_cases: List of test case dictionaries
        """
        self.test_cases_lookup = {}
        for case in test_cases:
            if 'image_path' in case:
                self.test_cases_lookup[case['image_path']] = case

        print(f"  Dummy model loaded {len(self.test_cases_lookup)} test cases")

    def query(self, prompt: str, image_path: str) -> str:
        """
        Return simulated answer for testing

        For combined prompts (verification + test question):
        - Verification: Pass with specified probability (using correct answer from case)
        - Main question: Random yes/no/unknown
        """
        import random

        # Look up the case info
        case = self.test_cases_lookup.get(image_path)

        # Check if this is a combined prompt (verification + test)
        if "Verification:" in prompt and "Main answer:" in prompt:
            return self._generate_combined_response(prompt, case)
        else:
            # Single question - random answer
            return random.choice(["yes", "no", "unknown"])

    def _generate_combined_response(self, prompt: str, case: dict = None) -> str:
        """Generate response for combined verification + test prompt"""
        import random

        # Generate verification response (using case info if available)
        if case:
            verification_response = self._generate_verification_response_with_case(
                case)
        else:
            verification_response = self._generate_random_verification()

        # Random answer for main question
        main_answer = random.choice(["yes", "no", "unknown"])

        # Format response
        response = f"""Verification: {verification_response}
Main answer: {main_answer}"""

        return response

    def _generate_verification_response_with_case(self, case: dict) -> str:
        """
        Generate verification response using case information

        Pass with probability = verification_pass_rate
        """
        import random

        # Decide if this attempt passes
        if random.random() > self.verification_pass_rate:
            # Fail: return wrong answer
            return self._generate_wrong_verification()

        # Pass: return correct answer from case
        squares = case.get('squares', [])
        pieces = case.get('pieces', {})

        # Generate correct response based on verification question type
        verification_q = case.get('verification_question', '').lower()

        if "how many squares" in verification_q:
            return "64"

        elif "what square" in verification_q and len(squares) == 1:
            # Single square question
            return squares[0]

        elif "what two squares" in verification_q or ("what squares" in verification_q and len(squares) == 2):
            # Two squares question
            if len(squares) >= 2:
                return f"{squares[0]} and {squares[1]}"
            else:
                return self._generate_wrong_verification()

        elif "what squares" in verification_q and len(squares) >= 3:
            # Three or more squares
            return ", ".join(squares[:-1]) + " and " + squares[-1]

        elif "how many pieces" in verification_q:
            # Piece count question
            return str(len(pieces))

        elif "what square" in verification_q and "piece" in verification_q:
            # Piece position question
            if pieces:
                piece_sq = list(pieces.keys())[0]
                return piece_sq
            else:
                return self._generate_wrong_verification()

        elif "adjacent" in verification_q:
            return "yes"

        else:
            # Default: return squares
            if len(squares) == 1:
                return squares[0]
            elif len(squares) >= 2:
                return f"{squares[0]} {squares[1]}"
            else:
                return "I see the board"

    def _generate_random_verification(self) -> str:
        """Generate random verification response (when case info not available)"""
        import random

        responses = [
            "I see a chess board",
            f"{self._generate_square_name()}",
            f"{self._generate_square_name()} and {self._generate_square_name()}",
        ]
        return random.choice(responses)

    def _generate_square_name(self) -> str:
        """Generate a random valid square name"""
        import random
        files = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
        ranks = ['1', '2', '3', '4', '5', '6', '7', '8']
        return random.choice(files) + random.choice(ranks)

    def _generate_wrong_verification(self) -> str:
        """Generate intentionally wrong verification response"""
        import random

        wrong_responses = [
            "I see a chess board",
            "The board looks normal",
            f"{self._generate_square_name()}",  # Wrong square
            # Wrong squares
            f"{self._generate_square_name()} and {self._generate_square_name()}",
            "I'm not sure",
            "c3 d4",  # Random squares
            "There are pieces on the board",
        ]

        return random.choice(wrong_responses)


class ClaudeModelClient(ModelClient):
    """Claude model client using Anthropic API"""

    def __init__(self, api_key: Optional[str] = None, model_name: str = "claude-sonnet-4-20250514"):
        """
        Initialize Claude client

        Args:
            api_key: Anthropic API key (if None, reads from ANTHROPIC_API_KEY env var)
            model_name: Claude model name
        """
        api_key = api_key or os.getenv("ANTHROPIC_API_KEY")

        if not api_key:
            raise ValueError(
                "Anthropic API key not found. Please set ANTHROPIC_API_KEY environment variable "
                "or pass api_key parameter."
            )

        super().__init__(model_name=model_name)
        self.api_key = api_key

        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=api_key)
        except ImportError:
            raise ImportError(
                "anthropic package is required. Install it with: pip install anthropic"
            )

    def query(self, prompt: str, image_path: str) -> str:
        """Call Claude API with image"""
        # Read and encode image
        with open(image_path, "rb") as image_file:
            image_data = base64.standard_b64encode(
                image_file.read()).decode("utf-8")

        # Determine image type
        if image_path.lower().endswith('.png'):
            media_type = "image/png"
        elif image_path.lower().endswith(('.jpg', '.jpeg')):
            media_type = "image/jpeg"
        else:
            media_type = "image/png"

        # Call API
        message = self.client.messages.create(
            model=self.model_name,
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": image_data,
                            },
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ],
                }
            ],
        )

        return message.content[0].text
