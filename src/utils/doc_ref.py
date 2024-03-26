"""
DocRef class that keeps track of file paths of PST items, named entities,
senders/recipients, etc.

"""
from collections import Counter
from pathlib import Path
import pandas as pd

from utils.doc import get_body_text, get_sender, get_recipients, check_if_folder_is_sent
from utils.io import load_json


class DocRef:
    def __init__(self, docs_folder: Path, ref_path: Path) -> None:
        self.docs_folder = docs_folder
        self.path = ref_path
        self.source_dict = load_json(docs_folder / 'sources.json')
        if ref_path.is_file():
            self.df = pd.read_pickle(ref_path)
        else:
            self.make_df()
            self.save()

    def make_df(self):
        d = {
            'path': [],
            'user_folder': [],
            'embed_index': [],
            'duplicate': [],
            'empty': [],
            'sender': [],
            'recipients': [],
            'ORG': None,
            'PER': None,
            'LOC': None,
            'checked_ORG': False,
            'checked_other_NER': False,
        }

        i = 0
        body_texts = set()
        for user_folder in self.docs_folder.iterdir():
            for path in user_folder.rglob('*.json'):
                if not path.is_file():
                    continue
                d['path'].append(path)
                d['user_folder'].append(user_folder)
                doc_d = load_json(path)
                body = get_body_text(doc_d, preprocessed=False)
                d['sender'].append(get_sender(doc_d))
                d['recipients'].append(get_recipients(doc_d))

                empty = not body
                duplicate = body in body_texts
                embed_index = -1

                d['empty'].append(empty)
                d['duplicate'].append(duplicate)

                if not empty and not duplicate:
                    embed_index = i
                    i += 1
                d['embed_index'].append(embed_index)
                if not duplicate:
                    body_texts.add(body)

        df = pd.DataFrame(d)
        df.set_index('path', inplace=True)
        df['embed_index'] = df['embed_index'].astype(int)
        self.df = df

    def get_users(self) -> set[str]:
        return set(self.df['user'].to_list())

    def get_user_folders(self) -> set[Path]:
        return set(self.df['user_folder'].to_list())

    def get_paths(self, encoded_only: bool = True) -> list[Path]:
        if encoded_only:
            paths = self.df[self.df['embed_index'] >= 0].index.to_list()
        else:
            paths = self.df.index.to_list()
        return paths

    def get_paths_by_query(self, query_label: str, query_threshold: float) -> list[Path]:
        paths_by_query = self.df[self.df[query_label]
                                 >= query_threshold].index.to_list()
        return paths_by_query

    def get_paths_by_user_folder(self, user_folder: Path, sent_only: bool = False) -> list[Path]:
        paths = self.df[self.df['user_folder'] == user_folder].index.to_list()
        if sent_only:
            paths = [path for path in paths if check_if_folder_is_sent(path)]
        return paths

    def get_paths_to_process(self, query_label: str, query_threshold: float) -> list[Path]:
        paths_to_process = self.df[(self.df[query_label] >= query_threshold) & (
            -self.df['checked_ORG'] | -self.df['checked_other_NER'])].index.to_list()
        return paths_to_process

    def add_ents(self, path: Path, ents: list[str] | set[str] | dict[str, set[str]], orgs_only: bool = False) -> None:
        if isinstance(ents, dict):
            for tag, ents_ in ents.items():
                self.df.at[path, tag] = ents_
        elif orgs_only:
            self.df.at[path, 'ORG'] = ents
        checked = 'checked_ORG' if orgs_only else 'checked_other_NER'
        self.df.loc[path, checked] = True  # type: ignore

    def get_ents(self, path: Path, ent_type: str) -> set[str] | dict[str, set[str]]:
        if ent_type != 'ORG':
            raise NotImplementedError
        return self.df.loc[path, ent_type]  # type: ignore

    def is_doc_tagged(self, path: Path, ent_type: str = 'other') -> bool:
        checked = 'checked_ORG' if ent_type == 'ORG' else 'checked_other_NER'
        return self.df.loc[path, checked]  # type: ignore

    def get_sender(self, path: Path) -> dict[str, str | None]:
        return self.df.loc[path, 'sender']  # type: ignore

    def get_recipients(self, path: Path) -> list[dict[str, str | None]]:
        return self.df.loc[path, 'recipients']  # type: ignore

    def save(self):
        self.df.to_pickle(self.path)
