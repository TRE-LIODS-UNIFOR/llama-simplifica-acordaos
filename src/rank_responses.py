from semantic_similarity import get_similarity_score


def by_similarity(responses: list[str], original: str, method: str = 'cosine') -> list[tuple[str, float]]:
    scores = []
    for response in responses:
        similarity = get_similarity_score(response, original, method=method)
        scores.append(similarity)
    ranked = sorted(zip(responses, scores), key=lambda x: x[1], reverse=True)
    return ranked
