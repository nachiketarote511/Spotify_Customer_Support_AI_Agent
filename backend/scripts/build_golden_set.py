"""
Golden Evaluation Set Builder
Creates a stratified sample for human-labelled evaluation.
Clearly distinguishes machine-generated suggestions from human labels.
"""

import pandas as pd
import numpy as np
import os
import json
from typing import Optional


def build_golden_set(conversations_df: pd.DataFrame,
                     intent_labels: pd.Series = None,
                     target_size: int = 200,
                     output_path: str = None,
                     random_state: int = 42) -> pd.DataFrame:
    """
    Build a golden evaluation dataset with stratified sampling.
    
    Sampling strategy:
    - Stratified across intents (if available)
    - Include common cases (60%)
    - Include rare/edge cases (20%)  
    - Include ambiguous cases (20%)
    
    Args:
        conversations_df: Processed conversations
        intent_labels: Intent labels for each conversation (optional)
        target_size: Target number of examples
        output_path: Path to save the golden set
        random_state: Random seed
    
    Returns:
        Golden evaluation DataFrame
    """
    np.random.seed(random_state)
    
    df = conversations_df.copy()
    if intent_labels is not None:
        df['intent'] = intent_labels.values if hasattr(intent_labels, 'values') else intent_labels
    
    golden_records = []
    
    if 'intent' in df.columns and df['intent'].nunique() > 1:
        # Stratified sampling across intents
        intent_counts = df['intent'].value_counts()
        n_intents = len(intent_counts)
        
        # Allocate samples per intent proportionally (with minimum)
        samples_per_intent = {}
        min_per_intent = max(5, target_size // (n_intents * 3))
        remaining = target_size
        
        for intent, count in intent_counts.items():
            n_samples = max(min_per_intent, int(target_size * count / len(df)))
            n_samples = min(n_samples, count)
            samples_per_intent[intent] = n_samples
            remaining -= n_samples
        
        # Distribute remaining samples
        if remaining > 0:
            for intent in intent_counts.index:
                if remaining <= 0:
                    break
                extra = min(remaining, 5)
                samples_per_intent[intent] += extra
                remaining -= extra
        
        for intent, n_samples in samples_per_intent.items():
            intent_df = df[df['intent'] == intent]
            
            if len(intent_df) <= n_samples:
                sampled = intent_df
            else:
                # Sample strategy within intent
                n_common = int(n_samples * 0.6)  # Common cases
                n_rare = int(n_samples * 0.2)     # Short/unusual messages
                n_ambiguous = n_samples - n_common - n_rare  # Medium-length
                
                # Common: medium-length messages (most typical)
                msg_lengths = intent_df['customer_message'].str.len()
                median_len = msg_lengths.median()
                common_mask = (msg_lengths > median_len * 0.5) & (msg_lengths < median_len * 1.5)
                common_pool = intent_df[common_mask]
                common_sample = common_pool.sample(min(n_common, len(common_pool)), random_state=random_state)
                
                # Rare: very short or very long messages
                rare_mask = ~common_mask
                rare_pool = intent_df[rare_mask]
                rare_sample = rare_pool.sample(min(n_rare, len(rare_pool)), random_state=random_state)
                
                # Ambiguous: from remaining
                used_ids = set(common_sample.index) | set(rare_sample.index)
                remaining_pool = intent_df[~intent_df.index.isin(used_ids)]
                ambiguous_sample = remaining_pool.sample(
                    min(n_ambiguous, len(remaining_pool)), random_state=random_state
                )
                
                sampled = pd.concat([common_sample, rare_sample, ambiguous_sample])
            
            for _, row in sampled.iterrows():
                golden_records.append({
                    'customer_message': row['customer_message'],
                    'true_intent': row.get('intent', ''),
                    'expected_escalation': '',  # To be filled by human
                    'expected_reasoning': '',   # To be filled by human
                    'historical_response': row.get('historical_support_response', ''),
                    'conversation_context': row.get('conversation_context', ''),
                    'conversation_id': row.get('conversation_id', ''),
                    'difficulty': _classify_difficulty(row),
                    'sampling_category': 'stratified',
                    # Machine-generated suggestions (NOT human labels)
                    'machine_suggested_intent': row.get('intent', ''),
                    'machine_suggestion_note': 'AUTO-GENERATED - requires human verification',
                    # Human label fields (empty - to be filled)
                    'human_verified_intent': '',
                    'human_verified_escalation': '',
                    'human_notes': '',
                    'is_human_verified': False,
                })
    else:
        # Random sampling without intent stratification
        if len(df) > target_size:
            sampled = df.sample(target_size, random_state=random_state)
        else:
            sampled = df
        
        for _, row in sampled.iterrows():
            golden_records.append({
                'customer_message': row['customer_message'],
                'true_intent': '',
                'expected_escalation': '',
                'expected_reasoning': '',
                'historical_response': row.get('historical_support_response', ''),
                'conversation_context': row.get('conversation_context', ''),
                'conversation_id': row.get('conversation_id', ''),
                'difficulty': _classify_difficulty(row),
                'sampling_category': 'random',
                'machine_suggested_intent': '',
                'machine_suggestion_note': 'No machine suggestion - human labeling required',
                'human_verified_intent': '',
                'human_verified_escalation': '',
                'human_notes': '',
                'is_human_verified': False,
            })
    
    golden_df = pd.DataFrame(golden_records)
    
    # Shuffle
    golden_df = golden_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    # Trim to target size
    if len(golden_df) > target_size:
        golden_df = golden_df.head(target_size)
    
    # Save
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        golden_df.to_csv(output_path, index=False)
        print(f"Golden set saved: {output_path} ({len(golden_df)} examples)")
        
        # Save metadata
        meta = {
            'total_examples': len(golden_df),
            'sampling_method': 'stratified' if 'intent' in df.columns else 'random',
            'intent_distribution': golden_df['true_intent'].value_counts().to_dict() if 'true_intent' in golden_df.columns else {},
            'difficulty_distribution': golden_df['difficulty'].value_counts().to_dict(),
            'human_verified_count': int(golden_df['is_human_verified'].sum()),
            'note': 'machine_suggested fields are AUTO-GENERATED. '
                    'human_verified fields must be filled by human annotators.',
        }
        meta_path = output_path.replace('.csv', '_metadata.json')
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)
    
    return golden_df


def _classify_difficulty(row) -> str:
    """Classify the difficulty level of a conversation."""
    msg = str(row.get('customer_message', ''))
    
    # Simple heuristics
    if len(msg) < 50:
        return 'easy'
    elif len(msg) > 200:
        return 'hard'
    elif '?' in msg and any(w in msg.lower() for w in ['why', 'how', 'when', 'what']):
        return 'medium'
    elif any(w in msg.lower() for w in ['cancel', 'refund', 'legal', 'fraud']):
        return 'hard'
    else:
        return 'medium'
