"""
Pipeline Selector Module
Analyzes dataset size and recommends the appropriate processing pipeline.
The user always retains manual control over pipeline selection.
"""

import os
import pandas as pd
from typing import Tuple, Optional


class DatasetAnalyzer:
    """Analyzes a dataset and recommends a processing pipeline."""
    
    # Default thresholds
    DEFAULT_SIZE_THRESHOLD_MB = 500
    DEFAULT_ROW_THRESHOLD = 1_000_000
    
    def __init__(self, size_threshold_mb: float = None, row_threshold: int = None):
        self.size_threshold_mb = size_threshold_mb or self.DEFAULT_SIZE_THRESHOLD_MB
        self.row_threshold = row_threshold or self.DEFAULT_ROW_THRESHOLD
    
    def analyze(self, file_path: str) -> dict:
        """
        Analyze the dataset and return classification.
        
        Returns:
            dict with keys: file_size_mb, estimated_rows, estimated_columns,
            estimated_memory_mb, classification, recommendation
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset not found: {file_path}")
        
        file_size_bytes = os.path.getsize(file_path)
        file_size_mb = file_size_bytes / (1024 * 1024)
        
        # Read a small sample to estimate structure
        sample = pd.read_csv(file_path, nrows=1000)
        n_columns = len(sample.columns)
        
        # Estimate rows from file size
        # Use sample to estimate bytes per row
        sample_mem = sample.memory_usage(deep=True).sum()
        bytes_per_row = sample_mem / len(sample)
        estimated_rows = int(file_size_bytes / (bytes_per_row * 0.5))  # CSV is ~2x memory
        
        # Estimate memory requirement
        estimated_memory_mb = (estimated_rows * bytes_per_row) / (1024 * 1024)
        
        # Classify
        is_large = (file_size_mb > self.size_threshold_mb or 
                    estimated_rows > self.row_threshold)
        
        classification = "LARGE" if is_large else "SMALL"
        recommendation = "PySpark + Databricks" if is_large else "Pandas + NumPy"
        
        return {
            'file_path': file_path,
            'file_size_mb': round(file_size_mb, 2),
            'estimated_rows': estimated_rows,
            'estimated_columns': n_columns,
            'column_names': sample.columns.tolist(),
            'estimated_memory_mb': round(estimated_memory_mb, 2),
            'classification': classification,
            'recommendation': recommendation,
            'thresholds': {
                'size_mb': self.size_threshold_mb,
                'rows': self.row_threshold,
            }
        }
    
    @staticmethod
    def format_analysis(analysis: dict) -> str:
        """Format analysis results for display."""
        cls = analysis['classification']
        
        output = []
        output.append("=" * 60)
        output.append(f"  DATASET ANALYSIS")
        output.append("=" * 60)
        output.append(f"  File: {analysis['file_path']}")
        output.append(f"  Size: {analysis['file_size_mb']:.2f} MB")
        output.append(f"  Estimated Rows: {analysis['estimated_rows']:,}")
        output.append(f"  Estimated Columns: {analysis['estimated_columns']}")
        output.append(f"  Estimated Memory: {analysis['estimated_memory_mb']:.2f} MB")
        output.append("")
        output.append(f"  Classification: {cls} DATASET")
        output.append(f"  Recommended: {analysis['recommendation']}")
        output.append("")
        output.append(f"  Available Pipelines:")
        output.append(f"    1. Pandas + NumPy {'(Recommended)' if cls == 'SMALL' else '(Warning: may be slow)'}")
        output.append(f"    2. PySpark + Databricks {'(Recommended)' if cls == 'LARGE' else ''}")
        output.append("=" * 60)
        
        return "\n".join(output)


def select_pipeline(analysis: dict, user_choice: Optional[str] = None) -> str:
    """
    Select the pipeline based on analysis and user choice.
    
    Args:
        analysis: Output from DatasetAnalyzer.analyze()
        user_choice: "pandas" or "spark". If None, uses recommendation.
    
    Returns:
        "pandas" or "spark"
    """
    recommended = "pandas" if analysis['classification'] == "SMALL" else "spark"
    
    if user_choice is None:
        return recommended
    
    choice = user_choice.lower().strip()
    if choice in ("pandas", "1", "pandas + numpy"):
        selected = "pandas"
    elif choice in ("spark", "2", "pyspark", "pyspark + databricks"):
        selected = "spark"
    else:
        print(f"Unknown choice: {user_choice}. Using recommended: {recommended}")
        return recommended
    
    # Warn if non-recommended
    if selected != recommended:
        if selected == "pandas" and analysis['classification'] == "LARGE":
            print("WARNING: You selected Pandas for a LARGE dataset.")
            print("This may cause memory issues or be very slow.")
            print(f"Dataset size: {analysis['file_size_mb']:.2f} MB, "
                  f"~{analysis['estimated_rows']:,} rows")
        elif selected == "spark" and analysis['classification'] == "SMALL":
            print("NOTE: You selected PySpark for a SMALL dataset.")
            print("Pandas would be faster for this dataset size.")
    
    return selected


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python pipeline_selector.py <dataset_path>")
        sys.exit(1)
    
    analyzer = DatasetAnalyzer()
    analysis = analyzer.analyze(sys.argv[1])
    print(analyzer.format_analysis(analysis))
