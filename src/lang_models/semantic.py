from pathlib import Path
import pickle

from nltk import sent_tokenize
from sentence_transformers import SentenceTransformer, util

from utils.doc_ref import DocRef
from utils.doc import get_body_text
from utils.io import dump_json, load_json


class SemanticModel:
    def __init__(self, name: str, embeds_folder: Path, doc_ref: DocRef) -> None:
        self.name = name
        self.embeds_folder = embeds_folder
        self.embeds_path = embeds_folder / f'{self.name}.pkl'
        self.sent_embeds_path = embeds_folder / f'{self.name}_sents.pkl'
        self.sent_embeds_ref_path = embeds_folder / 'sent_embeds_ref.json'
        self.model = SentenceTransformer(name)
        self.doc_ref = doc_ref

        if not self.embeds_path.is_file():
            self.encode_docs()
        else:
            self.load_doc_embeds()

    def encode_docs(self) -> None:
        print('\nEncoding docs. This might take a while...')

        docs = []
        for path in self.doc_ref.get_paths():
            docs.append(get_body_text(path, incl_subj=True))

        self.doc_embeds = self.model.encode(docs)
        with open(self.embeds_path, "wb") as fOut:
            pickle.dump(self.doc_embeds, fOut,
                        protocol=pickle.HIGHEST_PROTOCOL)

    def encode_sents(self) -> None:
        print('\nEncoding docs by sentence. This might take a while...')

        self.sent_embeds_ref = {}
        i = 0
        all_sents = []
        for path in self.doc_ref.get_paths():
            text = get_body_text(path)
            sents = sent_tokenize(text)
            for sent in sents:
                self.sent_embeds_ref[i] = {'sent': sent, 'path': path}
                all_sents.append(sent)
                i += 1
        self.sent_embeds = self.model.encode(all_sents)

        dump_json(self.sent_embeds_ref, self.sent_embeds_ref_path)
        with open(self.sent_embeds_path, "wb") as fOut:
            pickle.dump(self.sent_embeds, fOut,
                        protocol=pickle.HIGHEST_PROTOCOL)

    def load_doc_embeds(self) -> None:
        with open(self.embeds_path, 'rb') as f:
            self.doc_embeds = pickle.load(f)

    def load_sent_embeds(self) -> None:
        with open(self.sent_embeds_path, 'rb') as f:
            self.sent_embeds = pickle.load(f)
        self.sent_embeds_ref = load_json(self.sent_embeds_ref_path)

    def query_docs(self,
                   query: str | list[str],
                   query_label: str,
                   save: bool = False,
                   show_top_n: bool | int = False,
                   show_score: bool = True) -> None:
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

        if show_top_n:
            rows = df.nlargest(show_top_n, query_label)
            for i, row in enumerate(list(rows.itertuples())):
                body = get_body_text(row.Index)  # type: ignore
                print(f'RESULT {i + 1}:')
                if show_score:
                    print(f'Score: {row.__getattribute__(query_label)}')
                print('---------\n')
                print(body.strip())
                print(f'\n{"="*75}\n')

    def query_sents(self, query: str | list[str], top_n: int = 10, window: int = 0) -> None:
        all_scores = []
        if type(query) == str:
            query = [query]
        for query_str in query:
            query_embed = self.model.encode(query_str)
            query_scores = util.cos_sim(query_embed, self.sent_embeds)[  # type: ignore
                0].cpu().tolist()  # type: ignore
            all_scores.append(query_scores)
        max_scores = [max(x) for x in zip(*all_scores)]
        indices_scores = [(i, score) for i, score in enumerate(max_scores)]
        indices_scores.sort(key=lambda x: x[1], reverse=True)

        for i, score in indices_scores[:top_n]:
            last_i = len(self.sent_embeds) - 1
            window_l, window_r = [], []
            if window:
                for num in range(1, window + 1):
                    i_l = i - num
                    if i_l >= 0:
                        window_l.append(self.sent_embeds_ref[str(i_l)]['sent'])
                    i_r = i + num
                    if i_r <= last_i:
                        window_r.append(self.sent_embeds_ref[str(i_r)]['sent'])
            sent = self.sent_embeds_ref[str(i)]['sent']
            sent = '\u0332'.join(f'  {sent}  ')
            sents_to_show = ' '.join(window_l) + \
                f'\n{sent}\n' + ' '.join(window_r)

            print(f'RESULT {i + 1}:')
            print(f'score: {score}')
            print('---------\n')
            print(sents_to_show)
            print(f'\n{"="*75}\n')
