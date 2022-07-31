from pathlib import Path
if __name__ == "__main__":
    import sys
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from functools import partial
import concurrent.futures

from tqdm import tqdm
import networkx as nx
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from data.database import MilvusClient, MongoClient, StoryDataset
from data.utils import get_date_nday_before


def get_links_to_event(global_event_id, milvus_client: MilvusClient, mongo_client: MongoClient, r_thresh=0.8, p_thresh=0.05):
    if global_event_id is None:
        return []
    query_event, events_candidate = get_candidate_events(milvus_client, mongo_client, global_event_id)
    if query_event is None:
        return []
    links = link_predict(query_event, events_candidate, r_thresh, p_thresh)
    return links


def get_candidate_events(milvus_client: MilvusClient, mongo_client: MongoClient, global_event_id):
    query_event = milvus_client.query(expr=f"global_event_id == {global_event_id}",
                                      output_fields=["global_event_id", "date_added", "embeddings"], consistency_level="Strong")
    if len(query_event) == 0:
        return None, []
    
    query_event = query_event[0]
    date_added = query_event['date_added']
    events_candidate = mongo_client.find({
        "date_added": {"$gt": get_date_nday_before(date_added, days=3),
                       "$lte": get_date_nday_before(date_added, hours=1)},
        "embedded": True}, 
        projection=["global_event_id"]
    )
    ids_candidate = [event['global_event_id'] for event in events_candidate]
    
    search_params = {"metric_type": "IP", "params": {"nprobe": 10}}
    hit = milvus_client.search(
        data=[query_event["embeddings"]],
        anns_field="embeddings", 
        param=search_params, 
        limit=10, 
        expr=f"global_event_id in {ids_candidate}",
        consistency_level="Strong")[0]
    ids_candidate = [id for id in hit.ids]

    events_candidate = milvus_client.query(expr=f"global_event_id in {ids_candidate}",
                                           output_fields=["global_event_id", "embeddings"], consistency_level="Strong")
    events_candidate = [event for event in events_candidate if tuple(event["embeddings"]) != tuple(query_event["embeddings"])]

    return query_event, events_candidate


def link_predict(query_event, events_candidate, r_thresh=0.8, p_thresh=0.05):
    # tqdm.write(f"{query_event['global_event_id']}: len(events_candidate) = {len(events_candidate)}")
    if len(events_candidate) < 10:
        return []
    
    to_id = query_event['global_event_id']
    query_event = wrap_event(query_event)
    events_candidate = [wrap_event(event) for event in events_candidate]

    embeddings = [query_event['embeddings']]
    embeddings += [event['embeddings'] for event in events_candidate]
    columns = [query_event['global_event_id']]
    columns += [event['global_event_id'] for event in events_candidate]
    df2fit = pd.DataFrame({column: embeddings[i] for i, column in enumerate(columns)})
    fit_expr = f"{query_event['global_event_id']} ~ {' + '.join([event['global_event_id'] for event in events_candidate])}"
    model = smf.ols(fit_expr, data=df2fit).fit()

    if model.rsquared_adj < r_thresh:
        return []
    
    from_ids = [event_id for event_id in model.pvalues.index if event_id != "Intercept" and model.pvalues[event_id] < p_thresh]
    weights = 1 - model.pvalues[from_ids]
    from_ids = [dewrap_event_id(event_id) for event_id in from_ids]
    links = [(from_id, to_id, {"weight": weight, "r_thresh": r_thresh, "p_thresh": p_thresh}) for from_id, weight in zip(from_ids, weights)]
    return links


def wrap_event(event):
    new_event = {}
    new_event['global_event_id'] = wrap_event_id(event['global_event_id'])
    new_event["embeddings"] = event["embeddings"]
    return new_event


def wrap_event_id(global_event_id):
    return f"e_{global_event_id}"


def dewrap_event_id(wrapped_event_id):
    return int(wrapped_event_id.split("_")[1])


class StoryConstructor:
    def __init__(self, milvus_client: MilvusClient = None, mongo_client: MongoClient = None, story_dataset: StoryDataset = None):
        self.milvus_client = milvus_client
        self.mongo_client = mongo_client
        self.story_dataset = story_dataset

    def construct_by_tranverse(self, r_thresh=0.8, p_thresh=0.05, multi_process=False):
        DG = nx.DiGraph()
        events = self.mongo_client.get_news_to_storyfy(projection=["global_event_id"])
        global_event_ids = [event['global_event_id'] for event in events]
        
        get_links_to_event_ = partial(get_links_to_event, milvus_client=self.milvus_client, mongo_client=self.mongo_client, r_thresh=r_thresh, p_thresh=p_thresh)
        if multi_process:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                for global_event_id, links in tqdm(zip(global_event_ids, executor.map(get_links_to_event_, global_event_ids)), total=len(global_event_ids)):
                    if links:
                        DG.add_edges_from(links)
                    self.mongo_client.record_storified(global_event_id)
        else:
            for global_event_id in tqdm(global_event_ids):
                links = get_links_to_event_(global_event_id)
                if links:
                    DG.add_edges_from(links)
                self.mongo_client.record_storified(global_event_id)

        if self.story_dataset is not None:
            self.story_dataset.update(DG)
        
        return DG
    
    def hyperparameter_test(self):
        r_thresh = np.arange(0.5, 1.1, 0.1)
        p_thresh = 0.05
        all_stats = []
        for r_thresh_ in tqdm(r_thresh, desc="r_thresh"):
            DG = self.construct_by_tranverse(r_thresh=r_thresh_, p_thresh=p_thresh, multi_process=True)
            G = DG.to_undirected()
            ccs_stats = pd.Series([len(cc) for cc in nx.connected_components(G)]).describe()
            ccs_stats.loc["r_thresh"] = r_thresh_
            ccs_stats.loc["p_thresh"] = p_thresh
            all_stats.append(ccs_stats)
        df_stats = pd.DataFrame(all_stats)
        df_stats.to_csv("hyperparameter_test_new.csv")


if __name__ == "__main__":
    r_thresh = 0.8
    p_thresh = 0.05

    milvus_client = MilvusClient()
    mongo_client = MongoClient()
    story_dataset = StoryDataset()
    story_constructor = StoryConstructor(milvus_client=milvus_client, mongo_client=mongo_client, story_dataset=story_dataset)
    DG = story_constructor.construct_by_tranverse(r_thresh=r_thresh, p_thresh=p_thresh, multi_process=True)

    # story_constructor.hyperparameter_test()
    
