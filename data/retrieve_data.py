from pathlib import Path
if __name__ == "__main__":
    import sys
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from math import ceil

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


def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


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
        is_in = np.array([self.mongo_client.find_one({"global_event_id": int(id)}, projection=["global_event_id"]) is not None for id in ids])
        return event_meta_df[~is_in]

    def retrieve(self, event_meta_df, filter_by_source=True, filter_by_downloaded=True, embed_batch_size=1):
        if filter_by_source:
            event_meta_df = self.filter_event_from_important_sources(event_meta_df)
        if filter_by_downloaded:
            event_meta_df = self.filter_event_undownloaded(event_meta_df)
        ids, urls, dates_added = event_meta_df.GLOBALEVENTID.to_numpy(), event_meta_df.SOURCEURL.to_numpy(), event_meta_df.DATEADDED.to_numpy()

        with concurrent.futures.ThreadPoolExecutor() as executor:
            for global_event_id, date_added, url, (title, text) in tqdm(zip(ids, dates_added, urls, executor.map(download_news, urls)), total=len(ids),
                                                                        desc="Downloading news"):
                if title is not None and text is not None:
                    # query_event = self.milvus_client.query(expr=f"global_event_id == {global_event_id}",
                                    #   output_fields=["global_event_id"], consistency_level="Strong")
                    # embedded = (len(query_event) != 0)
                    # self.mongo_client.insert(global_event_id, date_added, title, text, url, embedded=embedded)
                    self.mongo_client.insert(global_event_id, date_added, title, text, url)

        unembedded_news = list(self.mongo_client.get_news_to_embed())
        for news_batch in tqdm(batch(unembedded_news, embed_batch_size),
                               total=ceil(len(unembedded_news) / embed_batch_size), desc="Embedding news"):
            global_event_ids = [news["global_event_id"] for news in news_batch]
            titles = [news["title"] for news in news_batch]
            texts = [news["text"] for news in news_batch]
            dates_added = [int(to_datestr(news["date_added"])) for news in news_batch]
            embedding = self.embedder.embedding_from_title_text(titles, texts)
            if embedding is not None:
                insert_results = self.milvus_client.insert(global_event_ids, embedding, dates_added)
                for global_event_id in insert_results.primary_keys:
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
    gdelt_dataset.update_database()
    for event_meta_df in tqdm(gdelt_dataset, desc="Retrieving news"):
        _ = news_retriever.retrieve(event_meta_df)
    
    milvus_client.release()
