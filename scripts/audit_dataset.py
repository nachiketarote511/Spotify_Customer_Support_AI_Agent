"""
Dataset Audit Script for Customer Support on Twitter Dataset
Phase 1 & 2: Complete dataset analysis + brand selection
Optimized: extracts @mentions once instead of per-brand scanning.
"""

import pandas as pd
import numpy as np
import os
import sys
import json
import re
from collections import Counter

DATASET_PATH = r'K:\Hiver_assignment\Customer_Support_AI_Agent\dataset\twcs\twcs.csv'
REPORT_DIR = r'K:\Hiver_assignment\Customer_Support_AI_Agent\reports'
DATA_DIR = r'K:\Hiver_assignment\Customer_Support_AI_Agent\data'

os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, 'raw'), exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, 'processed'), exist_ok=True)


def extract_first_mention(text):
    """Extract the first @mention from a tweet (the target of the tweet)."""
    if pd.isna(text):
        return None
    match = re.match(r'@(\w+)', str(text).strip())
    return match.group(1).lower() if match else None


def run_audit():
    print("=" * 80)
    print("DATASET AUDIT: Customer Support on Twitter")
    print("=" * 80)

    # File info
    file_size_bytes = os.path.getsize(DATASET_PATH)
    file_size_mb = file_size_bytes / (1024 * 1024)
    print(f"\nFile: {DATASET_PATH}")
    print(f"File Size: {file_size_mb:.2f} MB ({file_size_bytes:,} bytes)")

    # Load dataset
    print("\nLoading dataset...")
    df = pd.read_csv(DATASET_PATH)
    print(f"Dataset loaded successfully.")

    mem_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)

    audit = {}

    # Basic stats
    audit['rows'] = len(df)
    audit['columns'] = len(df.columns)
    audit['column_names'] = df.columns.tolist()
    audit['file_size_mb'] = round(file_size_mb, 2)
    audit['memory_mb'] = round(mem_mb, 2)

    print(f"\n--- BASIC STATS ---")
    print(f"Rows: {audit['rows']:,}")
    print(f"Columns: {audit['columns']}")
    print(f"Column Names: {audit['column_names']}")
    print(f"File Size: {audit['file_size_mb']} MB")
    print(f"Memory Usage: {audit['memory_mb']} MB")

    # Data types
    print(f"\n--- DATA TYPES ---")
    dtypes = df.dtypes.astype(str).to_dict()
    audit['dtypes'] = dtypes
    for col, dtype in dtypes.items():
        print(f"  {col}: {dtype}")

    # Missing values
    print(f"\n--- MISSING VALUES ---")
    missing = df.isnull().sum().to_dict()
    missing_pct = (df.isnull().sum() / len(df) * 100).round(2).to_dict()
    audit['missing_values'] = {k: int(v) for k, v in missing.items()}
    audit['missing_pct'] = {k: float(v) for k, v in missing_pct.items()}
    for col in df.columns:
        print(f"  {col}: {missing[col]:,} ({missing_pct[col]}%)")

    # Duplicates
    print(f"\n--- DUPLICATES ---")
    dup_rows = int(df.duplicated().sum())
    dup_tweet_ids = int(df.duplicated(subset=['tweet_id']).sum())
    audit['duplicate_rows'] = dup_rows
    audit['duplicate_tweet_ids'] = dup_tweet_ids
    print(f"  Duplicate rows: {dup_rows:,}")
    print(f"  Duplicate tweet_ids: {dup_tweet_ids:,}")

    # Inbound/Outbound distribution
    print(f"\n--- INBOUND / OUTBOUND ---")
    inbound_counts = df['inbound'].value_counts().to_dict()
    inbound_pct = df['inbound'].value_counts(normalize=True).mul(100).round(2).to_dict()
    audit['inbound_true'] = int(inbound_counts.get(True, 0))
    audit['inbound_false'] = int(inbound_counts.get(False, 0))
    audit['inbound_pct'] = float(inbound_pct.get(True, 0))
    audit['outbound_pct'] = float(inbound_pct.get(False, 0))
    print(f"  Inbound (customer): {audit['inbound_true']:,} ({audit['inbound_pct']}%)")
    print(f"  Outbound (support): {audit['inbound_false']:,} ({audit['outbound_pct']}%)")

    # Unique authors
    print(f"\n--- AUTHORS ---")
    unique_authors = df['author_id'].nunique()
    audit['unique_authors'] = unique_authors
    print(f"  Unique authors: {unique_authors:,}")

    # Date range
    print(f"\n--- DATE RANGE ---")
    dates = pd.to_datetime(df['created_at'], format='%a %b %d %H:%M:%S %z %Y', errors='coerce')
    valid_dates = dates.dropna()
    audit['date_min'] = str(valid_dates.min())
    audit['date_max'] = str(valid_dates.max())
    audit['date_span_days'] = (valid_dates.max() - valid_dates.min()).days
    audit['invalid_dates'] = int(dates.isnull().sum() - df['created_at'].isnull().sum())
    print(f"  Min date: {audit['date_min']}")
    print(f"  Max date: {audit['date_max']}")
    print(f"  Span: {audit['date_span_days']} days")
    print(f"  Invalid dates: {audit['invalid_dates']:,}")

    # Text statistics
    print(f"\n--- TEXT STATISTICS ---")
    text_lengths = df['text'].dropna().str.len()
    word_counts = df['text'].dropna().str.split().str.len()
    audit['text_min_chars'] = int(text_lengths.min())
    audit['text_max_chars'] = int(text_lengths.max())
    audit['text_mean_chars'] = round(float(text_lengths.mean()), 1)
    audit['text_median_chars'] = round(float(text_lengths.median()), 1)
    audit['text_min_words'] = int(word_counts.min())
    audit['text_max_words'] = int(word_counts.max())
    audit['text_mean_words'] = round(float(word_counts.mean()), 1)
    audit['text_median_words'] = round(float(word_counts.median()), 1)
    audit['empty_texts'] = int(df['text'].isnull().sum())
    print(f"  Characters: min={audit['text_min_chars']}, max={audit['text_max_chars']}, "
          f"mean={audit['text_mean_chars']}, median={audit['text_median_chars']}")
    print(f"  Words: min={audit['text_min_words']}, max={audit['text_max_words']}, "
          f"mean={audit['text_mean_words']}, median={audit['text_median_words']}")
    print(f"  Empty/null texts: {audit['empty_texts']:,}")

    # Conversation relationships
    print(f"\n--- CONVERSATION RELATIONSHIPS ---")
    has_response = int(df['response_tweet_id'].notna().sum())
    has_parent = int(df['in_response_to_tweet_id'].notna().sum())
    audit['tweets_with_responses'] = has_response
    audit['tweets_with_parent'] = has_parent
    audit['tweets_with_responses_pct'] = round(has_response / len(df) * 100, 2)
    audit['tweets_with_parent_pct'] = round(has_parent / len(df) * 100, 2)
    print(f"  Tweets with response_tweet_id: {has_response:,} ({audit['tweets_with_responses_pct']}%)")
    print(f"  Tweets with in_response_to_tweet_id: {has_parent:,} ({audit['tweets_with_parent_pct']}%)")

    # Link validity (using set lookups for speed)
    print(f"\n--- LINK VALIDITY ---")
    all_tweet_ids = set(df['tweet_id'].values)

    # Parse response_tweet_id (can be comma-separated)
    valid_resp = 0
    total_resp = 0
    for val in df['response_tweet_id'].dropna():
        for tid in str(val).split(','):
            tid = tid.strip()
            if tid:
                total_resp += 1
                try:
                    if int(float(tid)) in all_tweet_ids:
                        valid_resp += 1
                except ValueError:
                    pass
    audit['total_response_links'] = total_resp
    audit['valid_response_links'] = valid_resp
    audit['broken_response_links'] = total_resp - valid_resp
    print(f"  Response links total: {total_resp:,}")
    print(f"  Response links valid: {valid_resp:,}")
    print(f"  Response links broken: {total_resp - valid_resp:,}")

    parent_ids = df['in_response_to_tweet_id'].dropna().astype(int)
    valid_parents = int(parent_ids.isin(all_tweet_ids).sum())
    audit['total_parent_links'] = len(parent_ids)
    audit['valid_parent_links'] = valid_parents
    audit['broken_parent_links'] = len(parent_ids) - valid_parents
    print(f"  Parent links total: {len(parent_ids):,}")
    print(f"  Parent links valid: {valid_parents:,}")
    print(f"  Parent links broken: {len(parent_ids) - valid_parents:,}")

    # ---- BRAND ANALYSIS (optimized) ----
    print(f"\n--- BRAND ANALYSIS (optimized) ---")

    # Extract first @mention from each tweet (vectorized)
    print("  Extracting @mentions from all tweets...")
    df['target_mention'] = df['text'].str.extract(r'^@(\w+)', expand=False).str.lower()

    # Identify support accounts = outbound authors
    outbound_mask = df['inbound'] == False
    inbound_mask = df['inbound'] == True

    # Outbound counts per brand
    brand_outbound = df.loc[outbound_mask, 'author_id'].str.lower().value_counts()

    # Inbound counts: use extracted @mention to find which brand they're directed at
    brand_inbound = df.loc[inbound_mask, 'target_mention'].value_counts()

    # Merge to get brand stats
    all_brands = set(brand_outbound.index.tolist())
    brand_data = []
    for brand in all_brands:
        n_out = int(brand_outbound.get(brand, 0))
        n_in = int(brand_inbound.get(brand, 0))
        brand_data.append({
            'brand': brand,
            'outbound_tweets': n_out,
            'inbound_tweets': n_in,
            'total': n_out + n_in
        })

    brand_data.sort(key=lambda x: x['total'], reverse=True)
    audit['total_brands'] = len(brand_data)
    audit['top_brands'] = brand_data[:30]

    print(f"  Total unique support accounts: {audit['total_brands']}")
    print(f"\n  Top 30 Brands:")
    print(f"  {'Brand':<25} {'Outbound':>10} {'Inbound':>10} {'Total':>10}")
    print(f"  {'-'*55}")
    for bd in brand_data[:30]:
        print(f"  {bd['brand']:<25} {bd['outbound_tweets']:>10,} {bd['inbound_tweets']:>10,} {bd['total']:>10,}")

    # Save audit results
    audit_path = os.path.join(REPORT_DIR, 'dataset_audit.json')
    with open(audit_path, 'w') as f:
        json.dump(audit, f, indent=2, default=str)
    print(f"\nAudit saved to: {audit_path}")

    print("\n" + "=" * 80)
    print("DATASET AUDIT COMPLETE")
    print("=" * 80)

    return audit, df


def brand_selection_analysis(df):
    """Phase 2: Analyze and rank brands for selection."""
    print("\n" + "=" * 80)
    print("BRAND SELECTION ANALYSIS")
    print("=" * 80)

    outbound_mask = df['inbound'] == False
    inbound_mask = df['inbound'] == True

    # Ensure target_mention exists
    if 'target_mention' not in df.columns:
        df['target_mention'] = df['text'].str.extract(r'^@(\w+)', expand=False).str.lower()

    # Get brand outbound and inbound counts efficiently
    brand_outbound = df.loc[outbound_mask, 'author_id'].str.lower().value_counts()
    brand_inbound = df.loc[inbound_mask, 'target_mention'].value_counts()

    # Only consider brands with meaningful volume
    MIN_OUTBOUND = 500
    candidate_brands = brand_outbound[brand_outbound >= MIN_OUTBOUND].index.tolist()
    print(f"\nCandidate brands (>= {MIN_OUTBOUND} outbound tweets): {len(candidate_brands)}")

    # Pre-compute per-brand data slices
    df['author_lower'] = df['author_id'].str.lower()

    brand_scores = []
    for brand in candidate_brands:
        n_outbound = int(brand_outbound.get(brand, 0))
        n_inbound = int(brand_inbound.get(brand, 0))

        if n_inbound < 100:
            continue  # Skip brands with too few inbound messages

        # Conversation completeness: how many inbound tweets directed at this brand
        # have a response_tweet_id (i.e., got a reply)?
        brand_inbound_tweets = df[(inbound_mask) & (df['target_mention'] == brand)]
        inbound_with_response = brand_inbound_tweets['response_tweet_id'].notna().sum()
        completeness = inbound_with_response / max(len(brand_inbound_tweets), 1)

        # Link density: fraction of brand's tweets that have parent links
        brand_all = df[(df['author_lower'] == brand) | (df['target_mention'] == brand)]
        has_parent = brand_all['in_response_to_tweet_id'].notna().sum()
        parent_ratio = has_parent / max(len(brand_all), 1)

        # Text diversity: unique word patterns in inbound (sample for speed)
        sample_in = brand_inbound_tweets['text'].dropna()
        if len(sample_in) > 2000:
            sample_in = sample_in.sample(2000, random_state=42)
        # Remove @mention then take first 3 words
        cleaned = sample_in.str.replace(r'^@\w+\s*', '', regex=True).str.strip()
        first_words = cleaned.str.split().str[:3].str.join(' ')
        unique_patterns = first_words.nunique()
        diversity = unique_patterns / max(len(sample_in), 1)

        # Composite score
        vol_score = (min(n_outbound / 10000, 1.0) * 0.15 +
                     min(n_inbound / 10000, 1.0) * 0.15)
        comp_score = completeness * 0.25
        div_score = min(diversity, 1.0) * 0.20
        balance_score = min(n_inbound / max(n_outbound, 1), 1.5) / 1.5 * 0.15
        link_score = parent_ratio * 0.10

        total_score = vol_score + comp_score + div_score + balance_score + link_score

        brand_scores.append({
            'brand': brand,
            'n_outbound': n_outbound,
            'n_inbound': n_inbound,
            'conversation_completeness': round(completeness, 4),
            'parent_link_ratio': round(parent_ratio, 4),
            'text_diversity': round(diversity, 4),
            'inbound_outbound_ratio': round(n_inbound / max(n_outbound, 1), 4),
            'composite_score': round(total_score, 4),
        })

    # Sort by composite score
    brand_scores.sort(key=lambda x: x['composite_score'], reverse=True)

    print(f"\n{'Rank':<5} {'Brand':<25} {'Outbound':>10} {'Inbound':>10} "
          f"{'Complete':>10} {'Diversity':>10} {'Score':>8}")
    print("-" * 82)
    for i, bs in enumerate(brand_scores[:20], 1):
        print(f"{i:<5} {bs['brand']:<25} {bs['n_outbound']:>10,} {bs['n_inbound']:>10,} "
              f"{bs['conversation_completeness']:>10.2%} {bs['text_diversity']:>10.4f} "
              f"{bs['composite_score']:>8.4f}")

    # Save brand analysis
    report_path = os.path.join(REPORT_DIR, 'brand_selection.json')
    with open(report_path, 'w') as f:
        json.dump(brand_scores, f, indent=2)
    print(f"\nBrand selection analysis saved to: {report_path}")

    selected = brand_scores[0]
    print(f"\n{'=' * 80}")
    print(f"RECOMMENDED BRAND: {selected['brand']}")
    print(f"Score: {selected['composite_score']:.4f}")
    print(f"Outbound: {selected['n_outbound']:,} | Inbound: {selected['n_inbound']:,}")
    print(f"Completeness: {selected['conversation_completeness']:.2%}")
    print(f"{'=' * 80}")

    return brand_scores


if __name__ == '__main__':
    audit, df = run_audit()
    brand_scores = brand_selection_analysis(df)
