import evaluate

def bleu_score(pred_cypher: str,
               target_cypher: str,
               neo4j_connector,
               timeout: int = 30) -> float:
    """Whether the predicted Cypher query is executable"""
    google_bleu = evaluate.load("google_bleu")
    if type(pred_cypher) != str:
        return 0.0
    try:
        result = google_bleu.compute(predictions=[pred_cypher], references=[[target_cypher]])
    except Exception as e:
        return 0.0
    return result["google_bleu"]