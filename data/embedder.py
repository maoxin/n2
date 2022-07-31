from newspaper import Article
from transformers import RealmTokenizer, RealmEmbedder
import torch


class NewsEmbedder:
    def __init__(self, with_title=True, only_first_paragraph=True):
        self.with_title = with_title
        self.only_first_paragraph = only_first_paragraph

        # model_path = "google/realm-cc-news-pretrained-embedder"
        model_path = "/Users/maoxin/.cache/huggingface/realm-cc-news-pretrained-embedder"
        self.tokenizer = RealmTokenizer.from_pretrained(model_path)
        self.model = RealmEmbedder.from_pretrained(model_path)
        self.model.eval()
        
    def inference(self, title, text):
        with torch.no_grad():
            if self.with_title:
                inputs = self.tokenizer(title, text, return_tensors="pt", truncation="only_second", padding=True, max_length=512)
            else:
                inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            outputs = self.model(**inputs)
            scores = outputs.projected_score
        
        return scores.cpu().numpy()
        
    def embedding_from_title_text(self, title=None, text=None):
        if title is None or text is None:
            return None

        if isinstance(title, str):
            title = title.strip()
        else:
            title = [t.strip() for t in title]

        if self.only_first_paragraph:
            if isinstance(text, str):
                text = " ".join(self.split_paragraph(text)[:2]).strip()
            else:
                text = [" ".join(self.split_paragraph(t)[:2]).strip() for t in text]
        
        return self.inference(title, text)

    def split_paragraph(self, text):
        text = [t.strip() for t in text.split("\n") if t.strip()]
        return text