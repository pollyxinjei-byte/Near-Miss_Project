# src/decision_logic.py
import numpy as np

def classify_risk(cpa, tcpa, cpa_threshold=0.5):
    """
    Module 3: Decision Logic Layer (Chapter 4.5.2)
    
    Classifies a vessel encounter based on predicted physics metrics.
    A Near-Miss is flagged when:
      - CPA < threshold (safety perimeter violated)
      - TCPA > 0 (encounter is in the future, not diverging)
    
    Args:
        cpa: Closest Point of Approach (nautical miles)
        tcpa: Time to CPA (minutes)
        cpa_threshold: Safety distance limit (default 0.5 nm per IMO standards)
        
    Returns:
        label: "Near-Miss" or "Safe"
    """
    if tcpa > 0 and cpa < cpa_threshold:
        return "Near-Miss"
    
    return "Safe"


def classify_risk_batch(cpas, tcpas, cpa_threshold=0.5):
    """
    Vectorized batch classification for multiple encounters.
    
    Args:
        cpas: Array of CPA values (nautical miles)
        tcpas: Array of TCPA values (minutes)
        cpa_threshold: Safety distance limit (default 0.5 nm)
        
    Returns:
        labels: Array of classifications ("Near-Miss" or "Safe")
        near_miss_rate: Percentage of encounters classified as Near-Miss
    """
    labels = []
    for cpa, tcpa in zip(cpas, tcpas):
        labels.append(classify_risk(cpa, tcpa, cpa_threshold))
    
    labels = np.array(labels)
    near_miss_count = np.sum(labels == "Near-Miss")
    near_miss_rate = near_miss_count / len(labels) * 100
    
    return labels, near_miss_rate


def get_risk_summary(cpas, tcpas, cpa_threshold=0.5):
    """
    Generates a summary report of risk classification results.
    Corresponds to Chapter 5.3 analysis.
    
    Args:
        cpas: Array of CPA values
        tcpas: Array of TCPA values
        cpa_threshold: Safety threshold
        
    Returns:
        summary: Dictionary with classification statistics
    """
    labels, near_miss_rate = classify_risk_batch(cpas, tcpas, cpa_threshold)
    
    summary = {
        "total_encounters": len(labels),
        "near_miss_count": int(np.sum(labels == "Near-Miss")),
        "safe_count": int(np.sum(labels == "Safe")),
        "near_miss_rate": f"{near_miss_rate:.1f}%",
        "cpa_threshold": f"{cpa_threshold} nm"
    }
    
    return summary