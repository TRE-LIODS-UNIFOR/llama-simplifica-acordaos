from semantic_similarity import get_similarity_score


def by_similarity(responses, original):
    scores = []
    for response in responses:
        similarity = get_similarity_score(response, original, method='bertscore')
        scores.append(similarity)
    ranked = sorted(zip(responses, scores), key=lambda x: x[1], reverse=True)
    return ranked
