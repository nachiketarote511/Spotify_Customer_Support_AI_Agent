"""
Pandas Pipeline — Lightweight Processing
Pipeline A: For small datasets or quick evaluation.
Uses Python, Pandas, NumPy, scikit-learn.
Produces output matching the common data contract schema.
"""

import pandas as pd
import numpy as np
import os
import sys
import yaml
from typing import Optional

# Add backend root to path (for app.* imports)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from app.data.conversation_builder import build_conversations
from app.data.validation import validate_schema


class PandasPipeline:
    """Lightweight data processing pipeline using Pandas."""
    
    def __init__(self, config_path: str = None):
        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(__file__), '..', '..', '..', 'configs', 'config.yaml'
            )
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        
        self.brand = self.config['brand']
        self.project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    
    def run(self, dataset_path: str = None, output_path: str = None) -> pd.DataFrame:
        """
        Run the complete pandas processing pipeline.
        
        Steps:
        1. Load raw dataset
        2. Filter to brand
        3. Reconstruct conversations
        4. Validate output schema
        5. Save processed data
        
        Returns:
            DataFrame matching the common data contract
        """
        if dataset_path is None:
            dataset_path = os.path.join(self.project_root, self.config['dataset']['raw_path'])
        
        if output_path is None:
            output_path = os.path.join(self.project_root, self.config['dataset']['processed_path'])
        
        print("=" * 60)
        print("PANDAS PIPELINE: Starting")
        print(f"Brand: {self.brand}")
        print(f"Dataset: {dataset_path}")
        print("=" * 60)
        
        # Step 1: Load data
        print("\n[1/4] Loading dataset...")
        df = pd.read_csv(dataset_path)
        print(f"  Loaded {len(df):,} rows")
        
        # Step 2 & 3: Build conversations
        print("\n[2/4] Reconstructing conversations...")
        conversations = build_conversations(df, self.brand)
        print(f"  Reconstructed {len(conversations):,} conversations")
        
        # Step 3: Validate schema
        print("\n[3/4] Validating output schema...")
        validation = validate_schema(conversations)
        if validation['valid']:
            print("  Schema validation: PASSED")
        else:
            print(f"  Schema validation: FAILED")
            for error in validation['errors']:
                print(f"    ERROR: {error}")
        for warning in validation.get('warnings', []):
            print(f"    WARNING: {warning}")
        
        # Step 4: Save
        print("\n[4/4] Saving processed data...")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        conversations.to_csv(output_path, index=False)
        print(f"  Saved to: {output_path}")
        
        # Also save a small sample for quick evaluation
        sample_path = os.path.join(self.project_root, self.config['dataset']['sample_path'])
        os.makedirs(os.path.dirname(sample_path), exist_ok=True)
        sample_size = min(2000, len(conversations))
        sample = conversations.sample(sample_size, random_state=42)
        sample.to_csv(sample_path, index=False)
        print(f"  Sample ({sample_size} rows) saved to: {sample_path}")
        
        print("\n" + "=" * 60)
        print("PANDAS PIPELINE: Complete")
        print(f"  Total conversations: {len(conversations):,}")
        print(f"  Brands: {conversations['brand'].nunique()}")
        print(f"  Schema valid: {validation['valid']}")
        print("=" * 60)
        
        return conversations


if __name__ == '__main__':
    pipeline = PandasPipeline()
    result = pipeline.run()
