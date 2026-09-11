"""
Data Contract Schema
Defines the common schema between Pandas and Spark processing pipelines.
Both pipelines MUST produce output conforming to this schema.
"""

from dataclasses import dataclass, field
from typing import Optional, List
import json


@dataclass
class ConversationRecord:
    """
    Common data contract for processed conversations.
    Both Pipeline A (Pandas) and Pipeline B (Spark) must produce records
    matching this schema.
    """
    conversation_id: str
    brand: str
    customer_message: str
    historical_support_response: str
    timestamp: str
    conversation_context: str = ""
    customer_author: str = ""
    support_author: str = ""
    parent_tweet_id: Optional[float] = None
    response_tweet_id: Optional[float] = None
    conversation_length: int = 1
    is_escalation: bool = False
    intent: str = ""
    metadata: dict = field(default_factory=dict)

    REQUIRED_COLUMNS = [
        'conversation_id', 'brand', 'customer_message',
        'historical_support_response', 'timestamp'
    ]

    OPTIONAL_COLUMNS = [
        'conversation_context', 'customer_author', 'support_author',
        'parent_tweet_id', 'response_tweet_id', 'conversation_length',
        'is_escalation', 'intent', 'metadata'
    ]

    @classmethod
    def all_columns(cls) -> list:
        return cls.REQUIRED_COLUMNS + cls.OPTIONAL_COLUMNS

    @classmethod
    def validate_dataframe(cls, df) -> dict:
        """
        Validate that a DataFrame conforms to this data contract.
        Returns dict with 'valid' bool and 'errors' list.
        """
        errors = []
        warnings = []

        # Check required columns
        for col in cls.REQUIRED_COLUMNS:
            if col not in df.columns:
                errors.append(f"Missing required column: {col}")

        # Check for nulls in required columns
        for col in cls.REQUIRED_COLUMNS:
            if col in df.columns:
                null_count = df[col].isna().sum()
                if null_count > 0:
                    pct = null_count / len(df) * 100
                    if pct > 50:
                        errors.append(f"Column '{col}' has {pct:.1f}% null values")
                    elif pct > 5:
                        warnings.append(f"Column '{col}' has {pct:.1f}% null values")

        # Check data types
        if 'conversation_length' in df.columns:
            if not df['conversation_length'].dtype.kind in ('i', 'f'):
                warnings.append("conversation_length should be numeric")

        if 'is_escalation' in df.columns:
            if not df['is_escalation'].dtype == bool:
                warnings.append("is_escalation should be boolean")

        # Check row count
        if len(df) == 0:
            errors.append("DataFrame is empty")

        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'n_rows': len(df),
            'n_columns': len(df.columns),
            'required_present': [c for c in cls.REQUIRED_COLUMNS if c in df.columns],
            'optional_present': [c for c in cls.OPTIONAL_COLUMNS if c in df.columns],
        }

    def to_dict(self) -> dict:
        return {
            'conversation_id': self.conversation_id,
            'brand': self.brand,
            'customer_message': self.customer_message,
            'historical_support_response': self.historical_support_response,
            'timestamp': self.timestamp,
            'conversation_context': self.conversation_context,
            'customer_author': self.customer_author,
            'support_author': self.support_author,
            'parent_tweet_id': self.parent_tweet_id,
            'response_tweet_id': self.response_tweet_id,
            'conversation_length': self.conversation_length,
            'is_escalation': self.is_escalation,
            'intent': self.intent,
            'metadata': self.metadata,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=str)
