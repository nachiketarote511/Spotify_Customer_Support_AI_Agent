"""
Human Agreement Analysis Module
Compares LLM judge scores with human labels to measure agreement.
"""

import numpy as np
import json
import os
from typing import List, Dict, Optional


class HumanAgreementAnalyzer:
    """Analyzes agreement between LLM judge and human evaluations."""
    
    DIMENSIONS = [
        'relevance', 'groundedness', 'correctness', 'helpfulness',
        'brand_consistency', 'safety', 'escalation_appropriateness'
    ]
    
    def compute_agreement(self, human_scores: List[dict], 
                          llm_scores: List[dict]) -> dict:
        """
        Compute agreement metrics between human and LLM judge scores.
        
        Args:
            human_scores: List of dicts with dimension scores (1-5)
            llm_scores: List of dicts with dimension scores (1-5)
        
        Returns:
            Agreement metrics dict
        """
        results = {}
        
        for dim in self.DIMENSIONS:
            h_scores = []
            l_scores = []
            
            for h, l in zip(human_scores, llm_scores):
                h_val = self._extract_score(h, dim)
                l_val = self._extract_score(l, dim)
                if h_val is not None and l_val is not None:
                    h_scores.append(h_val)
                    l_scores.append(l_val)
            
            if len(h_scores) >= 3:
                results[dim] = self._compute_dim_agreement(h_scores, l_scores)
        
        # Overall agreement
        all_human = []
        all_llm = []
        for h, l in zip(human_scores, llm_scores):
            h_overall = self._extract_score(h, 'overall_score')
            l_overall = self._extract_score(l, 'overall_score')
            if h_overall is not None and l_overall is not None:
                all_human.append(h_overall)
                all_llm.append(l_overall)
        
        if len(all_human) >= 3:
            results['overall'] = self._compute_dim_agreement(all_human, all_llm)
        
        return results
    
    def _compute_dim_agreement(self, human: list, llm: list) -> dict:
        """Compute agreement metrics for a single dimension."""
        human = np.array(human)
        llm = np.array(llm)
        
        metrics = {}
        
        # Exact agreement
        metrics['exact_agreement'] = float(np.mean(human == llm))
        
        # Near agreement (within 1 point)
        metrics['near_agreement'] = float(np.mean(np.abs(human - llm) <= 1))
        
        # Mean absolute error
        metrics['mae'] = float(np.mean(np.abs(human - llm)))
        
        # Pearson correlation
        if np.std(human) > 0 and np.std(llm) > 0:
            metrics['pearson_r'] = float(np.corrcoef(human, llm)[0, 1])
        else:
            metrics['pearson_r'] = 0.0
        
        # Spearman rank correlation
        try:
            from scipy.stats import spearmanr
            corr, pval = spearmanr(human, llm)
            metrics['spearman_r'] = float(corr)
            metrics['spearman_pval'] = float(pval)
        except ImportError:
            metrics['spearman_r'] = None
        
        metrics['n_samples'] = len(human)
        metrics['human_mean'] = float(np.mean(human))
        metrics['llm_mean'] = float(np.mean(llm))
        
        return metrics
    
    def _extract_score(self, scores_dict: dict, dimension: str) -> Optional[int]:
        """Extract a numeric score from various formats."""
        if dimension in scores_dict:
            val = scores_dict[dimension]
            if isinstance(val, dict):
                return val.get('score')
            if isinstance(val, (int, float)):
                return int(val)
        return None
    
    def generate_report(self, agreement: dict) -> str:
        """Generate a human-readable agreement report."""
        lines = []
        lines.append("=" * 70)
        lines.append("  HUMAN-LLM JUDGE AGREEMENT ANALYSIS")
        lines.append("=" * 70)
        
        for dim, metrics in agreement.items():
            lines.append(f"\n  {dim.upper()}")
            lines.append(f"    Exact Agreement:     {metrics.get('exact_agreement', 0):.1%}")
            lines.append(f"    Near Agreement:      {metrics.get('near_agreement', 0):.1%}")
            lines.append(f"    MAE:                 {metrics.get('mae', 0):.2f}")
            lines.append(f"    Pearson r:           {metrics.get('pearson_r', 0):.3f}")
            if metrics.get('spearman_r') is not None:
                lines.append(f"    Spearman r:          {metrics.get('spearman_r', 0):.3f}")
            lines.append(f"    Samples:             {metrics.get('n_samples', 0)}")
            lines.append(f"    Human mean:          {metrics.get('human_mean', 0):.2f}")
            lines.append(f"    LLM mean:            {metrics.get('llm_mean', 0):.2f}")
        
        lines.append("\n" + "=" * 70)
        lines.append("  LIMITATIONS:")
        lines.append("    - Small sample size may affect correlation reliability")
        lines.append("    - LLM judge may have systematic biases")
        lines.append("    - Human labels reflect individual annotator judgment")
        lines.append("    - 1-5 scale has limited granularity")
        lines.append("=" * 70)
        
        return "\n".join(lines)


def create_annotation_template(messages: list, responses: list, 
                                output_path: str):
    """
    Create a CSV template for human annotation.
    Human labels are clearly distinguished from machine suggestions.
    """
    import pandas as pd
    
    records = []
    for i, (msg, resp) in enumerate(zip(messages, responses)):
        record = {
            'id': i + 1,
            'customer_message': msg,
            'generated_response': resp,
            # Human labels (to be filled)
            'human_relevance': '',
            'human_groundedness': '',
            'human_correctness': '',
            'human_helpfulness': '',
            'human_brand_consistency': '',
            'human_safety': '',
            'human_escalation_appropriateness': '',
            'human_overall': '',
            'human_notes': '',
            # Status
            'annotated': False,
            'annotator': '',
        }
        records.append(record)
    
    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Annotation template saved: {output_path} ({len(records)} items)")
