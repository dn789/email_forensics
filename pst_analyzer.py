"""
Analyzes a company or organization's emails and other PST contents and saves results.
"""

from collections import Counter, namedtuple
from pathlib import Path
import pickle

import pandas as pd
from sentence_transformers import SentenceTransformer, util
import tldextract
from tqdm import tqdm

from preprocess import preprocess_doc_bodies
from get_entities import get_entities
from utils import (find_email_addresses,
                   find_phone_nums,
                   find_urls,
                   load_json,
                   dump_json,
                   get_sender,
                   get_recipients,
                   get_contact
                   )


class PSTAnalyzer():
    """
    Analyzes a company or organization's emails and other PST contents and 
    saves results.

    When initialized with a source and output for the first time:
    (1) preprocesses docs
    (2) gets doc embeddings, 
    (3) gets user (employee, etc.), email address, and url info
    (4) gets vendor names

    search_docs provides semantic search capability.
    """

    def __init__(self, source: str, output: str, args: dict = {}) -> None:
        """
        Args:
            source (str): Folder of extracted PSTs (from process_pst)from 
                employees or members of a a single company or organization.
            output (str): Output folder. 
            args (dict, optional): Kwargs for clean_docs and get_vendors:
                {
                'proprocess_args': {...},
                'get_vendors_args: {...}
                }
                Defaults to {}.

        Raises:
            FileNotFoundError: source_folder doesn't exist.
        """

        # Set up paths

        self.source = Path(source)
        if not self.source.is_dir():
            raise FileNotFoundError(f'Input folder {self.source} not found.')
        self.output = Path(output)
        self.args = args
        self.semantic_model_name = args.get(
            'semantic_model_name', 'msmarco-distilbert-base-v4')

        folders = {
            'util': '_util',
            'preprocessing': '_util/preprocessing',
            'doc_embeds': '_util/doc_embeds',
            'users': '',
            'email_addrs': '',
            'urls': '',
            'misc': '',
            'vendors': ''
        }

        util_files = {
            'first_run_checklist': 'first_run_checklist.json',
            'doc_ref': 'doc_ref.pkl',
        }

        paths = {}

        for name, folder in folders.items():
            folder = folder or name
            paths[name] = path = self.output / folder
            path.mkdir(parents=True, exist_ok=True)
        paths['output'] = self.output

        for name, path in util_files.items():
            paths[name] = paths['util'] / path

        Paths = namedtuple('Paths', paths)
        self.paths = Paths(**paths)

        # Check if running on folder for first time
        self.first_run_checklist = load_json(
            self.paths.first_run_checklist) or {}
        if not self.first_run_checklist.get('first_run_complete'):
            self._first_run()
        else:
            self.doc_ref = pd.read_pickle(self.paths.doc_ref)
            self.load_semantic_model()

    def _first_run(self) -> None:
        first_run_items = {
            '_make_doc_ref': None,
            'preprocess': self.args.get('preprocess'),
            'load_semantic_model': None,
            'get_user_email_and_url_info': None,
            'get_vendors': self.args.get('get_vendors'),
        }

        for func_name, args in first_run_items.items():
            if not self.first_run_checklist.get(func_name):
                if args:
                    self.__getattribute__(func_name)(args)
                else:
                    self.__getattribute__(func_name)()
                self.first_run_checklist[func_name] = True
                dump_json(self.first_run_checklist,
                          self.paths.first_run_checklist)
            self.first_run_checklist['first_run_complete'] = True

            dump_json(self.first_run_checklist,
                      self.paths.first_run_checklist)

    def _make_doc_ref(self) -> None:
        d = {
            'path': [],
            'embed_index': [],
            'duplicate': [],
            'empty': [],
            'orgs': None,
            'got_orgs': False,
            'filters': []
        }

        i = 0
        body_texts = set()
        for path in self.source.rglob('*.json'):
            if not path.is_file():
                continue
            d['path'].append(path)
            d['filters'].append({})
            body = load_json(path).get('bodyText')
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
        df['embed_index'] = df['embed_index'].astype(int)
        df.to_pickle(self.paths.doc_ref)
        self.doc_ref = df

    def preprocess(self, kwargs: dict | None = None) -> None:
        print('\nPreprocessing docs...')

        kwargs = kwargs or {}
        preprocess_doc_bodies(self.doc_ref, self.paths.preprocessing, **kwargs)

    def load_semantic_model(self,  model_name: str | None = None) -> None:
        self.semantic_model_name = model_name or self.semantic_model_name
        self.semantic_model = SentenceTransformer(self.semantic_model_name)
        embeds_path = self.paths.doc_embeds / f'{self.semantic_model_name}.pkl'
        if not embeds_path.is_file():
            self._encode_docs()
        else:
            self._load_doc_embeds(embeds_path)

    def _encode_docs(self) -> None:
        print('\nEncoding docs. This might take a while...')

        paths = self.doc_ref[self.doc_ref['embed_index'] >= 0]['path'].tolist()
        docs = []
        for embeds_path in paths:
            doc_dict = load_json(embeds_path)
            docs.append(doc_dict.get('subject', '') + '\n\n' +
                        doc_dict.get('bodyTextPreprocessed', '').strip())

        self.doc_embeds = self.semantic_model.encode(docs)
        embeds_path = self.paths.doc_embeds / f'{self.semantic_model_name}.pkl'
        with open(embeds_path, "wb") as fOut:
            pickle.dump(self.doc_embeds, fOut,
                        protocol=pickle.HIGHEST_PROTOCOL)

    def _load_doc_embeds(self, path: Path) -> None:
        with open(path, 'rb') as f:
            self.doc_embeds = pickle.load(f)

    def get_user_email_and_url_info(self) -> None:
        print('\nGetting user, email address, and url info...')

        self.usernames = set()
        self.email_addr_dict = {}
        self.urls = set()
        self.org_domains = set()

        # Processes each user's contents, gets email and url info
        for user_folder in tqdm(list(self.source.iterdir())):
            user = {
                'name': user_folder.name,
                'contacts': [],
                'communicators': Counter()
            }

            self.usernames.add(user['name'])

            for path in user_folder.rglob('*.json'):
                if path.is_file():
                    self._process_doc(path, user)

            dump_json(user['contacts'], self.paths.users /
                      f'{user["name"]}_contacts.json', )
            dump_json(dict(user['communicators'].most_common(20)),
                      self.paths.users / f'{user["name"]}_top_communicatiors.json')

        # Gets email addrs with organization's domain, saves addrs.
        org_email_dict = {}
        for email_addr, v in self.email_addr_dict.items():
            if '@' in email_addr:
                domain = email_addr.split('@')[1]
                if domain in self.org_domains:
                    org_email_dict[email_addr] = v
                elif set(self.usernames) & v['names']:
                    if domain not in ('gmail.com', 'yahoo.com', 'hotmail.com'):
                        org_email_dict[email_addr] = v
                        self.org_domains.add(domain)

        dump_json(self.email_addr_dict,
                  self.paths.email_addrs / 'all_emails.json', )
        dump_json(org_email_dict,
                  self.paths.email_addrs / 'org_emails.json', )

        # Same as above but for urls.
        org_urls = set()
        for url in self.urls:
            parsed = tldextract.extract(url)
            domain = parsed.domain + '.' + parsed.suffix
            if domain in self.org_domains:
                org_urls.add(url)

        dump_json(self.urls, self.paths.urls / 'all_urls.json', )
        dump_json(org_urls, self.paths.urls / 'org_urls.json', )

        # Gets signatures from frequent text blocks that contain a phone number
        # and email address.
        signatures = set()
        for text in load_json(self.paths.preprocessing / 'freq_deduped_matches.json'):
            if find_email_addresses(text) and find_phone_nums(text):
                signatures.add(text)
        dump_json(signatures, self.paths.misc /
                  'signatures.json', )

    def _process_doc(self, path: Path, user: dict) -> None:
        doc = load_json(path)
        try:
            doc_type = doc['messageClass']
        except KeyError:
            print(path)

        if doc_type == 'IPM.Note':
            sender = get_sender(doc)
            recipients = get_recipients(doc)
            for name, addr in [sender] + recipients:
                if addr:
                    self.email_addr_dict.setdefault(addr, {'names': set()})
                    if name:
                        self.email_addr_dict[addr]['names'].add(name)
            # Checks if this is a sent item. Might need more robust method.
            if 'Sent Items' in path.__str__():
                user['communicators'].update([r[1] for r in recipients])
                # Get company/organization domains from sender in sent items.
                if '@' in sender[1]:
                    self.org_domains.add(sender[1].split('@')[1])
            else:
                user['communicators'][sender[1]] += 1

        elif doc_type == 'IPM.Contact':
            if contact := get_contact(doc):
                user['contacts'].append(contact)

        if body := doc.get('bodyText'):
            email_addrs = find_email_addresses(body)
            for addr in email_addrs:
                self.email_addr_dict.setdefault(addr, {'names': set()})
            self.urls.update(find_urls(body))

    def get_orgs(self, filter_terms: list | None = None, output: Path | None = None, kwargs: dict | None = None) -> None:
        output = output or self.output
        kwargs = kwargs or {}
        filter_model_name = kwargs.get(
            'filter_model_name', self.semantic_model_name)
        embeds_path = self.paths.doc_embeds / f'{filter_model_name}.pkl'
        if filter_terms:
            kwargs['filter_terms'] = filter_terms
        get_entities(self.doc_ref, self.paths.doc_ref, output, self.paths.util,
                     doc_embeds_path=embeds_path, **kwargs)

    def get_vendors(self, kwargs: dict | None = None) -> None:
        print('\nGetting vendor names...')
        kwargs = kwargs or {}
        kwargs.setdefault('filter_terms', ['invoice', 'payment', 'vendor'])
        self.get_orgs(output=self.paths.vendors, kwargs=kwargs)

    def search_docs(self, query: str, top_n: int = 10) -> None:
        query_embed = self.semantic_model.encode(query)
        scores = util.cos_sim(query_embed, self.doc_embeds)[  # type: ignore
            0].cpu().tolist()  # type: ignore
        embed_indices = range(len(self.doc_embeds))
        doc_score_pairs = list(zip(embed_indices, scores))
        doc_score_pairs = sorted(
            doc_score_pairs, key=lambda x: x[1], reverse=True)

        for i, (embed_i, score) in enumerate(doc_score_pairs[:top_n]):
            path = self.doc_ref.loc[self.doc_ref['embed_index']
                                    == embed_i].iloc[0]['path']
            body = load_json(path).get('bodyTextPreprocessed', '')
            print(f'RESULT {i + 1}:')
            print('---------\n')
            print(body.strip())
            print(f'\n{"="*75}\n')
