"""Spatial tests module"""

from .spatial_test_base import SpatialTestBase
from .test_0_pure_ability import SpatialTest0
from .test_0_generator import SpatialTest0Generator
from .test_1_rule_following import SpatialTest1
from .test_1_generator import SpatialTest1Generator
from .verification_generator import VerificationQuestionGenerator

__all__ = [
    'SpatialTestBase',
    'SpatialTest0',
    'SpatialTest0Generator',
    'SpatialTest1',
    'SpatialTest1Generator',
    'VerificationQuestionGenerator',
]
