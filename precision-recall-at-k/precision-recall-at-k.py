def precision_recall_at_k(recommended: list, relevant: list, k: int) -> list[float]:
    top_k_recommended = recommended[:k]
    relevant_set = set(relevant)
    
    precision = len(set(top_k_recommended) & relevant_set) / k
    recall = len(set(top_k_recommended) & relevant_set) / len(relevant_set)
    
    return [precision, recall]