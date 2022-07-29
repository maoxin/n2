from pathlib import Path
if __name__ == "__main__":
    import sys
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from datetime import datetime

from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)
from tqdm import tqdm
import numpy as np
import pandas as pd
import pymongo
import torch

from data.utils import to_date, get_date_range
from data import gdelt_download


class MilvusClient:
    def __init__(self):
        connections.connect("default", host="127.0.0.1", port="19530")
        if not utility.has_collection("realm_news"):
            self.create_table()
        self.collection = Collection("realm_news")
        
    def create_table(self):
        fields = [
            FieldSchema(name="global_event_id", dtype=DataType.INT64, is_primary=True, auto_id=False),
            FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=128),
            FieldSchema(name="date_added", dtype=DataType.INT64, is_primary=False)
        ]
        schema = CollectionSchema(fields, "realm_news stores the REALM embeddings of GDELT news")
        realm_news = Collection("realm_news", schema, consistency_level="Strong")

    def _reset(self):
        if utility.has_collection("realm_news"):
            utility.drop_collection("realm_news")
        self.create_table()
        self.build_index()

    def build_index(self):
        index_params = {
            "metric_type":"IP",
            "index_type":"IVF_FLAT",
            "params":{"nlist":128}
        }
        self.collection.create_index(
            field_name="embeddings", 
            index_params=index_params
        )
    
    def insert(self, global_event_id, embedding, date_added):
        if isinstance(embedding, torch.Tensor):
            embedding = embedding.detach().cpu().numpy()
        embedding = embedding.flatten()
        insert_result = self.collection.insert([
            [global_event_id], [embedding], [date_added]
        ])
        return insert_result

    def query(self, *args, **kwargs):
        self.collection.load()
        return self.collection.query(*args, **kwargs)

    def search(self, *args, **kwargs):
        self.collection.load()
        return self.collection.search(*args, **kwargs)


class MongoClient:
    def __init__(self):
        self.client = pymongo.MongoClient("mongodb://localhost:27017/")
        self.db = self.client["news"]
        self.collection = self.db["news"]

    def _reset(self):
        self.collection.drop()
        self.build_index()
    
    def build_index(self):
        self.collection.create_index([('global_event_id', pymongo.ASCENDING)], unique=True)
        self.collection.create_index("date_added")
    
    def insert(self, global_event_id, date_added, title, text, url):
        self.collection.insert_one({
            "global_event_id": int(global_event_id),
            "date_added": to_date(date_added),
            "title": title,
            "text": text,
            "url": url,
            "embedded": False,
            "storified": False,
        })

    def find(self, *args, **kwargs):
        return self.collection.find(*args, **kwargs)
    
    def find_one(self, *args, **kwargs):
        return self.collection.find_one(*args, **kwargs)

    def record_embedded(self, global_event_id):
        self.collection.update_one(
            {"global_event_id": global_event_id},
            {"$set": {"embedded": True}}
        )

    def record_storified(self, global_event_id):
        self.collection.update_one(
            {"global_event_id": global_event_id},
            {"$set": {"storified": True}}
        )

    def get_news_to_embed(self):
        return self.collection.find({"embedded": False})
    
    def get_news_to_storyfy(self):
        return self.collection.find({"storified": False, "embedded": True})


class GDELTDataset:
    def __init__(self, gdelt_dir="/Volumes/Extreme SSD/gdelt_archive"):
        self.gdelt_dir = Path(gdelt_dir)
        self.csv_paths = [path for path in self.gdelt_dir.glob("*.csv") if not path.name.startswith("._")]
        self.csv_paths = np.array(sorted(self.csv_paths, key=lambda path: to_date(path.name.split(".")[0])))
        self.dates = np.array([to_date(path.name.split(".")[0]) for path in self.csv_paths])
    
    def __len__(self):
        return len(self.csv_paths)
    
    def __getitem__(self, index):
        if isinstance(index, int):
            return pd.read_csv(str(self.csv_paths[index]))
        elif isinstance(index, slice):
            return [pd.read_csv(csv_path) for csv_path in self.csv_paths[index]]
       
        elif isinstance(index, datetime):
            return pd.read_csv(str(self.csv_paths[self.dates == index][0]))
        elif isinstance(index, str):
            return pd.read_csv(str(self.csv_paths[self.dates == to_date(index)][0]))
        
        else:
            raise NotImplementedError
    
    def update_database(self, start_date=None, end_date=None):
        if start_date is None:
            try:
                start_date = self.dates[-1]
            except IndexError:
                raise Exception("start_date is None and there are no csv files in the GDELT directory")        
        if end_date is None:
            end_date = datetime.utcnow()
            end_date = datetime(year=end_date.year, month=end_date.month, day=end_date.day)

        csv_paths = list(self.csv_paths)
        dates = list(self.dates)
        for date in tqdm(get_date_range(start_date, end_date), desc="Updating GDELT database"):
            gdelt_download.download_(date, output_dir=self.gdelt_dir)
            if to_date(date) not in self.dates:
                csv_paths.append(self.gdelt_dir / f"{date}.csv")
                dates.append(to_date(date))
        
        
        self.csv_paths = np.array(sorted(self.csv_paths, key=lambda path: to_date(path.name.split(".")[0])))
        self.dates = np.array([to_date(path.name.split(".")[0]) for path in self.csv_paths])
