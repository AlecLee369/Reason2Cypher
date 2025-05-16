import neo4j
from neo4j import Query, GraphDatabase
import evaluate
from neo4j_graph import neo4jGraph



def error_analysis(pred_cypher: str,
               target_cypher: str,
               neo4j_connector,
               timeout: int = 60) -> float:
    """Whether the predicted Cypher query is executable"""
    if type(pred_cypher) != str:
        return 'BadCypher'
    if pred_cypher == target_cypher:
        return 'NoError'
    if '<think>' in pred_cypher:
        return 'Overthink'
    try:
        with neo4j_connector.driver.session(database=neo4j_connector.username) as session:
            query = Query(pred_cypher, timeout=timeout)
            result = session.run(query)
    except (
            neo4j.exceptions.CypherSyntaxError,
            neo4j.exceptions.DatabaseError,
            neo4j.exceptions.CypherTypeError,
            neo4j.exceptions.ClientError,
    ) as e:
        # print(f"Warning: Exception {e} occurred while executing the predicted Cypher query {pred_cypher}")
        return type(e).__name__
    except Exception as e:
        # print(f"Warning: Exception {e} occurred while executing the predicted Cypher query {pred_cypher}")
        return type(e).__name__
    try:
        summary = result.consume()
        notifications = summary.notifications
        if notifications:
            return notifications[0]['code']
        else:
            return 'NoError'
    except Exception as e:
        return type(e).__name__
