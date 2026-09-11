"""
Evaluation Metrics Module
Comprehensive metrics for intent classification, retrieval, and generation quality.
"""

import numpy as np
import json
import os
from typing import Dict, List, Any, Optional
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)


def compute_classification_metrics(y_true: list, y_pred: list, 
                                    labels: list = None) -> dict:
    """Compute comprehensive classification metrics."""
    metrics = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'macro_precision': float(precision_score(y_true, y_pred, average='macro', zero_division=0)),
        'macro_recall': float(recall_score(y_true, y_pred, average='macro', zero_division=0)),
        'macro_f1': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
        'weighted_f1': float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
    }
    
    # Per-class metrics
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    metrics['per_class'] = {}
    all_labels = labels or sorted(set(y_true) | set(y_pred))
    for label in all_labels:
        label_str = str(label)
        if label_str in report:
            metrics['per_class'][label_str] = {
                'precision': report[label_str]['precision'],
                'recall': report[label_str]['recall'],
                'f1': report[label_str]['f1-score'],
                'support': report[label_str]['support'],
            }
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=all_labels)
    metrics['confusion_matrix'] = cm.tolist()
    metrics['class_names'] = [str(l) for l in all_labels]
    
    return metrics


def compute_escalation_metrics(y_true: list, y_pred: list) -> dict:
    """Compute escalation decision metrics."""
    # Binary: True = escalate, False = auto-handle
    true_binary = [1 if y else 0 for y in y_true]
    pred_binary = [1 if y else 0 for y in y_pred]
    
    metrics = {
        'accuracy': float(accuracy_score(true_binary, pred_binary)),
        'escalation_precision': float(precision_score(true_binary, pred_binary, zero_division=0)),
        'escalation_recall': float(recall_score(true_binary, pred_binary, zero_division=0)),
        'escalation_f1': float(f1_score(true_binary, pred_binary, zero_division=0)),
        'escalation_rate_true': sum(true_binary) / max(len(true_binary), 1),
        'escalation_rate_predicted': sum(pred_binary) / max(len(pred_binary), 1),
    }
    
    return metrics


def compute_retrieval_metrics(queries: list, retrieved: list, 
                               relevant: list, k_values: list = [1, 3, 5]) -> dict:
    """
    Compute retrieval metrics.
    
    Args:
        queries: Query strings
        retrieved: List of lists of retrieved doc IDs
        relevant: List of sets of relevant doc IDs
        k_values: K values for Recall@K, Precision@K
    """
    metrics = {}
    
    for k in k_values:
        recalls = []
        precisions = []
        
        for ret, rel in zip(retrieved, relevant):
            ret_at_k = set(ret[:k])
            rel_set = set(rel)
            
            if len(rel_set) > 0:
                recall = len(ret_at_k & rel_set) / len(rel_set)
                recalls.append(recall)
            
            precision = len(ret_at_k & rel_set) / max(len(ret_at_k), 1)
            precisions.append(precision)
        
        metrics[f'recall@{k}'] = float(np.mean(recalls)) if recalls else 0.0
        metrics[f'precision@{k}'] = float(np.mean(precisions))
    
    # MRR (Mean Reciprocal Rank)
    mrrs = []
    for ret, rel in zip(retrieved, relevant):
        rel_set = set(rel)
        for rank, doc_id in enumerate(ret, 1):
            if doc_id in rel_set:
                mrrs.append(1.0 / rank)
                break
        else:
            mrrs.append(0.0)
    
    metrics['mrr'] = float(np.mean(mrrs))
    
    return metrics


def format_metrics_report(metrics: dict, title: str = "Evaluation Results") -> str:
    """Format metrics as a readable report."""
    lines = []
    lines.append("=" * 60)
    lines.append(f"  {title}")
    lines.append("=" * 60)
    
    for key, value in metrics.items():
        if key in ('per_class', 'confusion_matrix', 'class_names'):
            continue
        if isinstance(value, float):
            lines.append(f"  {key:<30} {value:.4f}")
        else:
            lines.append(f"  {key:<30} {value}")
    
    if 'per_class' in metrics:
        lines.append("\n  Per-class metrics:")
        for cls, cls_metrics in metrics['per_class'].items():
            f1 = cls_metrics.get('f1', 0)
            support = cls_metrics.get('support', 0)
            lines.append(f"    {cls:<25} F1={f1:.3f}  support={support}")
    
    lines.append("=" * 60)
    return "\n".join(lines)
