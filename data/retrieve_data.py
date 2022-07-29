from pathlib import Path
if __name__ == "__main__":
    import sys
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from newspaper import Article
import concurrent.futures


from tqdm import tqdm
import numpy as np
import pandas as pd
import yaml

from data.database import MilvusClient, MongoClient, GDELTDataset
from data.embedder import NewsEmbedder
from data.utils import to_datestr


def download_news(url):
        article = Article(url)
        try:
            article.download()
            article.parse()
            return article.title, article.text
        except:
            return None, None


class NewsRetriever:
    def __init__(self, config, embedder: NewsEmbedder = None, milvus_client: MilvusClient = None, mongo_client: MongoClient = None):
        self.config = config
        self.embedder = embedder
        self.milvus_client = milvus_client
        self.mongo_client = mongo_client

    def filter_event_from_important_sources(self, event_meta_df):
        event_meta_df = event_meta_df[event_meta_df.SOURCEURL.str.contains("|".join(self.config['news_sources']))].drop_duplicates(subset=['SOURCEURL'])
        return event_meta_df

    def filter_event_undownloaded(self, event_meta_df):
        ids = event_meta_df.GLOBALEVENTID.to_numpy()
        is_in = np.array([self.mongo_client.find_one({"global_event_id": int(id)}) is not None for id in ids])
        return event_meta_df[~is_in]

    def retrieve(self, event_meta_df, filter_by_source=True, filter_by_downloaded=True):
        if filter_by_source:
            event_meta_df = self.filter_event_from_important_sources(event_meta_df)
        if filter_by_downloaded:
            event_meta_df = self.filter_event_undownloaded(event_meta_df)
        ids, urls, dates_added = event_meta_df.GLOBALEVENTID.to_numpy(), event_meta_df.SOURCEURL.to_numpy(), event_meta_df.DATEADDED.to_numpy()

        with concurrent.futures.ThreadPoolExecutor() as executor:
            for global_event_id, date_added, url, (title, text) in tqdm(zip(ids, dates_added, urls, executor.map(download_news, urls)), total=len(ids),
                                                                        desc="Downloading news"):
                if title is not None and text is not None:
                    self.mongo_client.insert(global_event_id, date_added, title, text, url)

        unembedded_news = list(self.mongo_client.get_news_to_embed())
        for news in tqdm(unembedded_news, desc="Embedding news"):
            global_event_id, title, text, date_added = news['global_event_id'], news['title'], news['text'], news['date_added']
            date_added = int(to_datestr(date_added))
            embedding = self.embedder.embedding_from_title_text(title, text)
            if embedding is not None:
                self.milvus_client.insert(global_event_id, embedding, date_added)
            self.mongo_client.record_embedded(global_event_id)



if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--config", default="./config.yaml")
    parser.add_argument("--gdelt_dir", default="/Volumes/Extreme SSD/gdelt_archive")
    parser.add_argument("--reset_milvus", action="store_true")
    parser.add_argument("--reset_mongo", action="store_true")
    args = parser.parse_args()
    
    config_path = Path(args.config)
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    embedder = NewsEmbedder(with_title=True, only_first_paragraph=True)
    milvus_client = MilvusClient()
    if args.reset_milvus:
        milvus_client._reset()
    mongo_client = MongoClient()
    if args.reset_mongo:
        mongo_client._reset()
    news_retriever = NewsRetriever(config, embedder, milvus_client, mongo_client)

    gdelt_dataset = GDELTDataset(args.gdelt_dir)
    # gdelt_dataset.update_database()
    for event_meta_df in tqdm(gdelt_dataset, desc="Retrieving news"):
        _ = news_retriever.retrieve(event_meta_df)
