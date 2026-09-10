"""
Data Validation Module
Enforces the common data contract between processing pipelines.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any
import json
import pandas as pd


@dataclass
class ConversationRecord:
    """
    Common data contract for a customer support conversation.
    Both Pandas and Spark pipelines must produce records matching this schema.
    """
    conversation_id: str
    brand: str
    customer_message: str
    historical_support_response: str
    timestamp: str
    conversation_context: str = ""
    customer_author: str = ""
    support_author: str = ""
    parent_tweet_id: Optional[str] = None
    response_tweet_id: Optional[str] = None
    conversation_length: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> 'ConversationRecord':
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# Required columns in the output DataFrame
REQUIRED_COLUMNS = [
    'conversation_id',
    'brand',
    'customer_message',
    'historical_support_response',
    'timestamp',
    'conversation_context',
    'customer_author',
    'support_author',
    'conversation_length',
]

# Column types for validation
COLUMN_TYPES = {
    'conversation_id': 'object',
    'brand': 'object',
    'customer_message': 'object',
    'historical_support_response': 'object',
    'timestamp': 'object',
    'conversation_context': 'object',
    'customer_author': 'object',
    'support_author': 'object',
    'conversation_length': 'int64',
}


def validate_schema(df: pd.DataFrame) -> dict:
    """
    Validate that a DataFrame conforms to the common data contract.
    Returns a dict with validation results.
    """
    results = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'stats': {
            'total_rows': len(df),
            'columns_found': df.columns.tolist(),
        }
    }

    # Check required columns
    missing_cols = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing_cols:
        results['valid'] = False
        results['errors'].append(f"Missing required columns: {missing_cols}")

    # Check for empty dataframe
    if len(df) == 0:
        results['valid'] = False
        results['errors'].append("DataFrame is empty")
        return results

    # Check for null values in critical fields
    critical_fields = ['conversation_id', 'brand', 'customer_message', 'historical_support_response']
    for col in critical_fields:
        if col in df.columns:
            null_count = df[col].isnull().sum()
            if null_count > 0:
                pct = null_count / len(df) * 100
                if pct > 5:
                    results['errors'].append(f"{col}: {null_count} nulls ({pct:.1f}%)")
                    results['valid'] = False
                else:
                    results['warnings'].append(f"{col}: {null_count} nulls ({pct:.1f}%)")

    # Check for empty strings in critical fields
    for col in critical_fields:
        if col in df.columns:
            empty_count = (df[col] == '').sum()
            if empty_count > 0:
                results['warnings'].append(f"{col}: {empty_count} empty strings")

    # Check conversation_id uniqueness
    if 'conversation_id' in df.columns:
        dup_count = df['conversation_id'].duplicated().sum()
        if dup_count > 0:
            results['warnings'].append(f"conversation_id: {dup_count} duplicates")

    # Check brand consistency
    if 'brand' in df.columns:
        unique_brands = df['brand'].nunique()
        results['stats']['unique_brands'] = unique_brands

    # Stats
    if 'conversation_length' in df.columns:
        results['stats']['mean_conversation_length'] = float(df['conversation_length'].mean())
        results['stats']['max_conversation_length'] = int(df['conversation_length'].max())

    if 'customer_message' in df.columns:
        results['stats']['mean_message_length'] = float(df['customer_message'].str.len().mean())

    return results


def validate_no_leakage(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    """
    Check for data leakage between train and test sets.
    Uses conversation-level checking to prevent near-identical messages
    from the same conversation appearing in both sets.
    """
    results = {
        'leakage_detected': False,
        'issues': [],
    }

    # Check conversation ID overlap
    if 'conversation_id' in train_df.columns and 'conversation_id' in test_df.columns:
        train_convos = set(train_df['conversation_id'].unique())
        test_convos = set(test_df['conversation_id'].unique())
        overlap = train_convos & test_convos
        if overlap:
            results['leakage_detected'] = True
            results['issues'].append(
                f"Conversation ID overlap: {len(overlap)} conversations appear in both train and test"
            )

    # Check exact message overlap
    if 'customer_message' in train_df.columns and 'customer_message' in test_df.columns:
        train_msgs = set(train_df['customer_message'].dropna().unique())
        test_msgs = set(test_df['customer_message'].dropna().unique())
        msg_overlap = train_msgs & test_msgs
        if msg_overlap:
            pct = len(msg_overlap) / len(test_msgs) * 100
            if pct > 1:
                results['leakage_detected'] = True
                results['issues'].append(
                    f"Message overlap: {len(msg_overlap)} messages ({pct:.1f}%) appear in both sets"
                )

    # Check customer author overlap
    if 'customer_author' in train_df.columns and 'customer_author' in test_df.columns:
        train_authors = set(train_df['customer_author'].dropna().unique())
        test_authors = set(test_df['customer_author'].dropna().unique())
        author_overlap = train_authors & test_authors
        if author_overlap:
            pct = len(author_overlap) / len(test_authors) * 100
            results['issues'].append(
                f"Author overlap: {len(author_overlap)} authors ({pct:.1f}%) appear in both sets "
                f"(this may be acceptable but should be monitored)"
            )

    return results
