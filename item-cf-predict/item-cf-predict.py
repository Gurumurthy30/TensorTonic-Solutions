def item_cf_predict(user_ratings: list, item_similarities: list, target: int) -> float:
    numerator = 0.0
    denominator = 0.0
    for i in range(len(user_ratings)):
        if i == target or user_ratings[i] == 0:
            continue
        sim = item_similarities[i]
        if sim > 0:
            numerator += sim * user_ratings[i]
            denominator += sim
    if denominator == 0:
        return 0.0
    return numerator / denominator