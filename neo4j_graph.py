from neo4j import GraphDatabase, Query
from typing import Literal
from utils import to_hashable, compare_execution



class neo4jGraph:

    def __init__(self, URI, username, password):
        self.driver = GraphDatabase.driver(URI, 
                                        auth=(username, password),
                                        # connection_timeout=120,
                                        # notifications_min_severity='OFF',  # or 'WARNING' to enable entirely
                                        # notifications_disabled_classifications=['HINT', 'GENERIC'],
                                        )
        self.URI = URI
        self.username = username
        
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.driver.close()


    def run_query(self, cypher, timeout=None, convert_to_hashable=True, convert_func: Literal['data', 'graph'] = 'data'):
        
        with self.driver.session(database=self.username) as session:
            query = Query(cypher, timeout=timeout)
            result = session.run(query)
            if convert_func == 'data':
                result = result.data()
            elif convert_func == 'graph':
                result = result.graph()
            else:
                raise ValueError(f"Invalid convert_func: {convert_func}")
            # if convert_to_hashable == True:
            #     hashable_result = [{k: to_hashable(v) for k, v in record.items()} for record in result]
            #     return hashable_result
        return result
        
    def get_num_entities(self):
        return self.driver.execute_query(
            "MATCH (n) RETURN count(n) as num",
            database_= self.username,
            )[0][0]['num']

    def get_num_relations(self):
        return self.driver.execute_query(
            "MATCH ()-[r]->() RETURN count(r) as num",
            database_= self.username,
            )[0][0]['num']