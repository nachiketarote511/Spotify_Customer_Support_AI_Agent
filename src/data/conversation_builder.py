"""
Conversation Builder Module
Reconstructs customer-support conversations from individual tweets using
tweet_id, response_tweet_id, and in_response_to_tweet_id linkages.
"""

import pandas as pd
import numpy as np
import re
import hashlib
from typing import Optional, Tuple
from collections import defaultdict


def extract_mention(text: str) -> Optional[str]:
    """Extract the first @mention from a tweet."""
    if pd.isna(text):
        return None
    match = re.match(r'@(\w+)', str(text).strip())
    return match.group(1).lower() if match else None


def clean_text(text: str) -> str:
    """Remove @mentions from tweet text for cleaner messages."""
    if pd.isna(text):
        return ""
    # Remove leading @mention
    cleaned = re.sub(r'^@\w+\s*', '', str(text).strip())
    # Remove URLs
    cleaned = re.sub(r'https?://\S+', '[URL]', cleaned)
    # Remove extra whitespace
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned


def build_conversations(df: pd.DataFrame, brand: str) -> pd.DataFrame:
    """
    Reconstruct conversations for a specific brand.
    
    Strategy:
    1. Filter tweets related to the brand (inbound @brand or outbound from brand)
    2. Build conversation chains using response links
    3. Create paired records (customer message + support response)
    4. Preserve multi-turn context where possible
    
    Args:
        df: Full dataset DataFrame
        brand: Brand identifier (lowercase)
    
    Returns:
        DataFrame with reconstructed conversations matching the common schema
    """
    print(f"Building conversations for brand: {brand}")
    
    # Normalize author_id for comparison
    df = df.copy()
    df['author_lower'] = df['author_id'].str.lower()
    df['target_mention'] = df['text'].str.extract(r'^@(\w+)', expand=False).str.lower()
    
    # Filter tweets related to this brand
    brand_outbound = df[(df['author_lower'] == brand) & (df['inbound'] == False)]
    brand_inbound = df[(df['target_mention'] == brand) & (df['inbound'] == True)]
    
    print(f"  Brand outbound tweets: {len(brand_outbound):,}")
    print(f"  Brand inbound tweets: {len(brand_inbound):,}")
    
    # Combine all brand-related tweets
    brand_tweet_ids = set(brand_outbound['tweet_id'].values) | set(brand_inbound['tweet_id'].values)
    brand_tweets = df[df['tweet_id'].isin(brand_tweet_ids)].copy()
    
    # Build lookup dictionaries for fast traversal
    tweet_lookup = {}
    for _, row in brand_tweets.iterrows():
        tweet_lookup[row['tweet_id']] = row
    
    # Build parent -> children mapping
    children_map = defaultdict(list)
    for _, row in brand_tweets.iterrows():
        parent = row.get('in_response_to_tweet_id')
        if pd.notna(parent):
            children_map[int(parent)].append(row['tweet_id'])
    
    # Parse response_tweet_id (can be comma-separated)
    response_map = {}
    for _, row in brand_tweets.iterrows():
        resp = row.get('response_tweet_id')
        if pd.notna(resp):
            for tid in str(resp).split(','):
                tid = tid.strip()
                if tid:
                    try:
                        response_map[row['tweet_id']] = int(float(tid))
                    except ValueError:
                        pass
    
    # Strategy: Find inbound customer tweets that received a brand response
    # This gives us clean customer_message -> support_response pairs
    conversations = []
    seen_pairs = set()
    
    for _, inbound_row in brand_inbound.iterrows():
        customer_tweet_id = inbound_row['tweet_id']
        customer_text = inbound_row['text']
        customer_author = inbound_row['author_id']
        
        # Find the brand's response to this inbound tweet
        # Method 1: Check if inbound tweet has a response_tweet_id pointing to a brand tweet
        response_id = response_map.get(customer_tweet_id)
        if response_id and response_id in tweet_lookup:
            response_row = tweet_lookup[response_id]
            if response_row['author_lower'] == brand:
                pair_key = (customer_tweet_id, response_id)
                if pair_key not in seen_pairs:
                    seen_pairs.add(pair_key)
                    
                    # Build context from earlier messages in the thread
                    context = _build_context(customer_tweet_id, tweet_lookup, max_depth=3)
                    
                    conv_id = _generate_conv_id(customer_tweet_id, response_id)
                    conversations.append({
                        'conversation_id': conv_id,
                        'brand': brand,
                        'customer_message': clean_text(customer_text),
                        'historical_support_response': clean_text(response_row['text']),
                        'timestamp': str(inbound_row['created_at']),
                        'conversation_context': context,
                        'customer_author': str(customer_author),
                        'support_author': str(response_row['author_id']),
                        'parent_tweet_id': str(inbound_row.get('in_response_to_tweet_id', '')),
                        'response_tweet_id': str(response_id),
                        'conversation_length': 1,  # Will be updated
                    })
                    continue
        
        # Method 2: Check children_map — did any brand tweet respond to this inbound?
        child_ids = children_map.get(customer_tweet_id, [])
        for child_id in child_ids:
            if child_id in tweet_lookup:
                child_row = tweet_lookup[child_id]
                if child_row['author_lower'] == brand:
                    pair_key = (customer_tweet_id, child_id)
                    if pair_key not in seen_pairs:
                        seen_pairs.add(pair_key)
                        
                        context = _build_context(customer_tweet_id, tweet_lookup, max_depth=3)
                        
                        conv_id = _generate_conv_id(customer_tweet_id, child_id)
                        conversations.append({
                            'conversation_id': conv_id,
                            'brand': brand,
                            'customer_message': clean_text(customer_text),
                            'historical_support_response': clean_text(child_row['text']),
                            'timestamp': str(inbound_row['created_at']),
                            'conversation_context': context,
                            'customer_author': str(customer_author),
                            'support_author': str(child_row['author_id']),
                            'parent_tweet_id': str(inbound_row.get('in_response_to_tweet_id', '')),
                            'response_tweet_id': str(child_id),
                            'conversation_length': 1,
                        })
                        break  # Take the first brand response
    
    # Create DataFrame
    if not conversations:
        print("  WARNING: No conversations reconstructed!")
        return pd.DataFrame()
    
    conv_df = pd.DataFrame(conversations)
    
    # Calculate conversation lengths by grouping related conversations
    conv_df = _compute_conversation_lengths(conv_df, tweet_lookup)
    
    # Filter out unusable conversations
    conv_df, filter_stats = _filter_conversations(conv_df)
    
    print(f"\n  --- Conversation Reconstruction Summary ---")
    print(f"  Total pairs found: {len(conversations):,}")
    print(f"  After filtering: {len(conv_df):,}")
    print(f"  Filter stats: {filter_stats}")
    print(f"  Avg customer message length: {conv_df['customer_message'].str.len().mean():.0f} chars")
    print(f"  Avg support response length: {conv_df['historical_support_response'].str.len().mean():.0f} chars")
    
    return conv_df


def _build_context(tweet_id: int, tweet_lookup: dict, max_depth: int = 3) -> str:
    """
    Build conversation context by traversing parent links.
    Returns a string with prior messages in chronological order.
    """
    context_tweets = []
    current_id = tweet_id
    depth = 0
    
    while depth < max_depth:
        if current_id not in tweet_lookup:
            break
        row = tweet_lookup[current_id]
        parent_id = row.get('in_response_to_tweet_id')
        if pd.isna(parent_id):
            break
        parent_id = int(parent_id)
        if parent_id not in tweet_lookup:
            break
        
        parent_row = tweet_lookup[parent_id]
        role = "customer" if parent_row['inbound'] else "support"
        context_tweets.append(f"[{role}]: {clean_text(parent_row['text'])}")
        
        current_id = parent_id
        depth += 1
    
    # Reverse to get chronological order
    context_tweets.reverse()
    return " | ".join(context_tweets) if context_tweets else ""


def _generate_conv_id(customer_tweet_id: int, response_tweet_id: int) -> str:
    """Generate a deterministic conversation ID from tweet IDs."""
    raw = f"{customer_tweet_id}_{response_tweet_id}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def _compute_conversation_lengths(conv_df: pd.DataFrame, tweet_lookup: dict) -> pd.DataFrame:
    """Estimate conversation length by counting context segments."""
    def count_length(row):
        ctx = row.get('conversation_context', '')
        if not ctx:
            return 2  # Just the pair (customer + support)
        return len(ctx.split(' | ')) + 2
    
    conv_df['conversation_length'] = conv_df.apply(count_length, axis=1)
    return conv_df


def _filter_conversations(conv_df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """
    Filter out unusable conversations.
    Documents all filtering decisions.
    """
    stats = {'initial': len(conv_df)}
    
    # Filter 1: Remove conversations with empty customer messages
    mask = conv_df['customer_message'].str.strip().str.len() > 5
    conv_df = conv_df[mask]
    stats['after_empty_customer'] = len(conv_df)
    
    # Filter 2: Remove conversations with empty support responses
    mask = conv_df['historical_support_response'].str.strip().str.len() > 5
    conv_df = conv_df[mask]
    stats['after_empty_response'] = len(conv_df)
    
    # Filter 3: Remove support responses that are just DM redirects
    dm_patterns = [
        r'please?\s*(send|dm|direct\s*message)',
        r'(send|dm)\s*(us|me)\s*(a|your)',
        r'private\s*message',
        r'follow\s*(us|me)\s*(so|and)',
    ]
    dm_regex = '|'.join(dm_patterns)
    # Keep DM redirects but flag them - don't remove entirely as they show brand behavior
    conv_df['is_dm_redirect'] = conv_df['historical_support_response'].str.contains(
        dm_regex, case=False, regex=True, na=False
    )
    stats['dm_redirects'] = int(conv_df['is_dm_redirect'].sum())
    
    # Filter 4: Remove exact duplicate customer messages
    before_dedup = len(conv_df)
    conv_df = conv_df.drop_duplicates(subset=['customer_message'], keep='first')
    stats['duplicates_removed'] = before_dedup - len(conv_df)
    
    # Filter 5: Remove very short/meaningless messages
    mask = conv_df['customer_message'].str.split().str.len() >= 3
    conv_df = conv_df[mask]
    stats['after_short_filter'] = len(conv_df)
    
    stats['final'] = len(conv_df)
    
    return conv_df.reset_index(drop=True), stats


if __name__ == '__main__':
    import os
    import yaml
    
    # Load config
    config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'configs', 'config.yaml')
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    brand = config['brand']
    dataset_path = os.path.join(os.path.dirname(__file__), '..', '..', config['dataset']['raw_path'])
    
    print(f"Loading dataset from: {dataset_path}")
    df = pd.read_csv(dataset_path)
    
    conversations = build_conversations(df, brand)
    
    output_path = os.path.join(os.path.dirname(__file__), '..', '..', config['dataset']['processed_path'])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    conversations.to_csv(output_path, index=False)
    print(f"\nSaved {len(conversations):,} conversations to: {output_path}")
