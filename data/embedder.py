from newspaper import Article
from transformers import RealmTokenizer, RealmEmbedder
import torch


class NewsEmbedder:
    def __init__(self, with_title=True, only_first_paragraph=True):
        self.with_title = with_title
        self.only_first_paragraph = only_first_paragraph

        self.tokenizer = RealmTokenizer.from_pretrained("google/realm-cc-news-pretrained-embedder")
        self.model = RealmEmbedder.from_pretrained("google/realm-cc-news-pretrained-embedder")
        self.model.eval()
        
    def inference(self, title, text):
        with torch.no_grad():
            if self.with_title:
                inputs = self.tokenizer(title, text, return_tensors="pt", max_length=512, truncation=True)
            else:
                inputs = self.tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
            outputs = self.model(**inputs)
            scores = outputs.projected_score
        
        return scores.cpu().numpy()
        
    def embedding_from_title_text(self, title=None, text=None):
        if title is None or text is None:
            return None
        
        title = title.strip()
        if self.only_first_paragraph:
            text = text.split("\n")[0].strip()
        return self.inference(title, text)