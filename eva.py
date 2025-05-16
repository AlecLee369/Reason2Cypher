import argparse
import copy
import json
import os
import math
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from metrics import *
from neo4j_graph import neo4jGraph
from neo4j import GraphDatabase
from utils import *

RETURN_PATTERN_MAPPING = {
    "n_name": "n_name",
    "n_prop": "n_prop_combined",
    "n_name_prop": "n_prop_combined",
    "n_prop_distinct": "n_prop_combined",
    "n_prop_array_distinct": "n_prop_combined",
    "n_order_by": "n_order_by",
    "n_argmax": "n_argmax",
    "n_where": "n_where",
    "n_agg": "n_agg",
    "n_group_by": "n_group_by"
}

METRIC_FUNC_MAPPING = {
    'execution_accuracy': execution_accuracy,
    'psjs': provenance_subgraph_jaccard_similarity,
    'executable': executable,
    'exact_match': exact_match,
    'bleu_score': bleu_score,
    'error_analysis': error_analysis
}

# item: csv row
def compute_metrics(item, metrics, driver):
    item = copy.deepcopy(item)
    for m in metrics:
        pred_cypher = item['generated_cypher']
        ref_cypher = item['cypher']
        result = METRIC_FUNC_MAPPING[m](
            pred_cypher=pred_cypher,
            target_cypher=ref_cypher,
            neo4j_connector=driver,
        )
        item[m] = result
    return item


def avg_and_round(nums: list[float], n: int = 4):
    return round(sum(nums) / len(nums), n) if nums else math.nan


def aggregate(results: list[tuple[str, float]]):
    res = {}
    for key, value in results:
        if key not in res:
            res[key] = []
        res[key].append(value)
    for key, values in res.items():
        res[key] = avg_and_round(values)
    return res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_threads', type=int, default=16)
    # parser.add_argument('--metrics', nargs='+', default=['error_analysis'])
    parser.add_argument('--metrics', nargs='+', default=['bleu_score', 'exact_match', 'error_analysis'])
    # parser.add_argument('--metric_for_agg', default='execution_accuracy')
    parser.add_argument('--URI', default='neo4j+s://demo.neo4jlabs.com:7687')


    args = parser.parse_args()
    print(args)
    print()
    # open csv file
    absolute_path = '/evaluated_dataset/'
    target_file_name = 'test_infer_llama-r16-WholeCot'
    NEO4J_DATASET = os.path.join(absolute_path, f'{target_file_name}.csv')
    print("CSV Path:", NEO4J_DATASET)
    if os.path.exists(NEO4J_DATASET):
        df = pd.read_csv(NEO4J_DATASET)
        df = df.sort_values(by='database_reference_alias', ascending=True)
        # df = df[df['database_reference_alias'] == 'neo4jlabs_demo_db_offshoreleaks']
    else:
        print(f"Error: File not found - {NEO4J_DATASET}")

    df = df.sort_values(by='database_reference_alias', ascending=True)
    unique_graphs = df['database_reference_alias'].unique()
    # unique_graphs = ['neo4jlabs_demo_db_offshoreleaks', 'neo4jlabs_demo_db_recommendations']
    result_dfs = []
    failed_connections = []
    
    for grpah in unique_graphs:
        print(f"Processing grpah: {grpah}")
        df_category = df[df['database_reference_alias'] == grpah].copy()
        graph_name = grpah.replace("neo4jlabs_demo_db_", "")
        # AUTH = (graph_name, graph_name)
        # try:    
        with neo4jGraph(args.URI, graph_name, graph_name) as connector:
            connector.driver.verify_connectivity()
            print(f"Connection established in {graph_name} graph.")
                    # Use ThreadPoolExecutor for multithreading
        results = []
        with ThreadPoolExecutor(max_workers=args.num_threads) as executor:
        # with ThreadPoolExecutor(max_workers=1) as executor:
            futures = [executor.submit(compute_metrics, row, args.metrics, connector) for _, row in df_category.iterrows()]
            for future in tqdm(as_completed(futures), total=len(df_category)):
                results.append(future.result())
            df_results = pd.DataFrame(results)
            result_dfs.append(df_results)

        # except Exception as e:
        #     print(f"Fail to connect to {graph_name} graph: {e}")
        #     failed_connections.append(graph_name)
        #     continue
    if result_dfs:
        final_df = pd.concat(result_dfs, ignore_index=True)
        absolute_path = '/work/pi_wenlongzhao_umass_edu/9/bloomberg_project/evaluated_dataset/'
        evaluated_name = f'{target_file_name}_evaluated'
        final_df.to_csv(os.path.join(absolute_path, f'{evaluated_name}.csv'), index=False)
        print(f"Results saved to {evaluated_name}.csv")
        print(f'failed graph name: {failed_connections}')
        # Convert results back to a DataFrame and update 'age' column
        # df_category = pd.DataFrame(results)
        # result_dfs.append(df_category)




if __name__ == '__main__':
    main()
