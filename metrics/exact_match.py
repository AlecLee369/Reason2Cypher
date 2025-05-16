"""
Some code are adapted from https://github.com/taoyds/test-suite-sql-eval and https://github.com/megagonlabs/cypherbench/blob/main/cypherbench/
"""
import neo4j
from neo4j import Query, GraphDatabase
import evaluate
from neo4j_graph import neo4jGraph
from utils import compare_execution, to_hashable

# execution accuracy score
def exact_match(pred_cypher: str,
               target_cypher: str,
               neo4j_connector,
               timeout: int = 10) -> float:
    """Whether the predicted Cypher query is executable"""
    if type(pred_cypher) != str:
        return 0.0
    if pred_cypher == target_cypher:
        return 1.0
    if '<think>' in pred_cypher:
        return 0.0
    try:
        generated_result = neo4j_connector.run_query(pred_cypher, timeout=timeout, convert_to_hashable=True)
        generated_result = [{k: to_hashable(v) for k, v in record.items()} for record in generated_result]


    except (
            neo4j.exceptions.CypherSyntaxError,
            neo4j.exceptions.DatabaseError,
            neo4j.exceptions.CypherTypeError,
            neo4j.exceptions.ClientError,
    ) as e:
        # print(f"Warning: Exception {e} occurred while executing the predicted Cypher query {pred_cypher}")
        return 0.0
    except TypeError as e:

        return 0.0
    except Exception as e:
        # print(f"Warning: Exception {e} occurred while executing the predicted Cypher query {pred_cypher}")
        return 0.0
    # con.close()
    
    try: 
        target_result = neo4j_connector.run_query(target_cypher, timeout=timeout, convert_to_hashable=True)
        target_result = [{k: to_hashable(v) for k, v in record.items()} for record in target_result]
    except TypeError as e:
        # some items in target cypher output is not hashable
        return 0.0
    except Exception as e:
        return 0.0
    try:
        output = compare_execution(generated_result, target_result, order_matters=False)
    except Exception as e:
        return 0.0
    return output


