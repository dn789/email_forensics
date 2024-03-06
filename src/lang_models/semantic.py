from pathlib import Path
import pickle

from sentence_transformers import SentenceTransformer, util

from utils.doc_ref import DocRef
from utils.doc import get_body_text


class SemanticModel:
    def __init__(self, model_name: str, embeds_path: Path, doc_ref: DocRef) -> None:
        self.model_name = model_name
        self.embeds_path = embeds_path
        self.model = SentenceTransformer(model_name)
        self.doc_ref = doc_ref

        if not embeds_path.is_file():
            self.encode_docs()
        else:
            self.load_doc_embeds()

    def encode_docs(self) -> None:
        print('\nEncoding docs. This might take a while...')

        paths = self.doc_ref.df[self.doc_ref.df['embed_index']
                                >= 0].index.tolist()
        docs = []
        for path in paths:
            docs.append(get_body_text(path, incl_subj=True))

        self.doc_embeds = self.model.encode(docs)
        with open(self.embeds_path, "wb") as fOut:
            pickle.dump(self.doc_embeds, fOut,
                        protocol=pickle.HIGHEST_PROTOCOL)

    def load_doc_embeds(self) -> None:
        with open(self.embeds_path, 'rb') as f:
            self.doc_embeds = pickle.load(f)

    def query_docs(self, query: str | list[str], query_label: str, save: bool = False) -> None:
        if type(query) == str:
            query = [query]
        all_scores = []
        for query_str in query:
            query_embed = self.model.encode(query_str)
            query_scores = util.cos_sim(query_embed, self.doc_embeds)[  # type: ignore
                0].cpu().tolist()  # type: ignore
            all_scores.append(query_scores)
        max_scores = [max(x) for x in zip(*all_scores)]
        df = self.doc_ref.df
        df[query_label] = None
        df[query_label] = [max_scores[embed_index] if embed_index >=
                           0 else None for embed_index in df['embed_index']]
        if save:
            self.doc_ref.save()
