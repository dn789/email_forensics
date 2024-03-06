"""
Analyzes a company or organization's emails and other PST contents and 
saves results.

"""

from collections import namedtuple
from pathlib import Path

from get_entities import get_entities_by_doc_query
from get_contact_info import main as get_contact_info, get_signatures_and_save
from preprocess import preprocess_doc_bodies
from lang_models.semantic import SemanticModel
from utils.io import load_json, dump_json
from utils.doc_ref import DocRef
from utils.doc import get_body_text


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
        self.semantic_model_name = args.get(
            'semantic_model_name', 'msmarco-distilbert-base-v4')

        folders = {
            'util': '_util',
            'preprocessing': '_util/preprocessing',
            'doc_embeds': '_util/doc_embeds',
            'entities_util': '_util/entities',
            'contact_info': '',
            'entities': ''
        }

        util_files = {
            'checklist': 'checklist.json',
            'doc_ref': 'doc_ref.pkl',
            'doc_embeds_file': f'doc_embeds/{self.semantic_model_name}.pkl'
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

        self.args = args
        if self.paths.checklist.exists():
            self.checklist = load_json(self.paths.checklist)
        else:
            self.checklist = {}

        # Makes or loads doc ref
        self.doc_ref = DocRef(self.source, self.paths.util / 'doc_ref.pkl')

        # Preprocessing
        if not self.checklist.get('preprocess'):
            self.preprocess(**args.get('preprocess', {}))
            self.checklist['preprocess'] = True
            dump_json(self.checklist, self.paths.checklist)

        # Loads semantic model and get doc embeddings if they donn't exist

        self.semantic_model = SemanticModel(
            self.semantic_model_name, self.paths.doc_embeds_file, self.doc_ref)

        # Other checklist items

        if not self.checklist.get('get_contact_info'):
            self.get_contact_info()
            self.checklist['get_contact_info'] = True
            dump_json(self.checklist, self.paths.checklist)

        if not self.checklist.get('get_vendors'):
            get_vendors_args = args.get('get_vendors', {})
            self.get_vendors(**get_vendors_args)
            self.checklist['get_vendors'] = True
            dump_json(self.checklist, self.paths.checklist)

    def preprocess(self, **kwargs) -> None:
        print('\nPreprocessing docs...')
        preprocess_doc_bodies(
            self.doc_ref, self.paths.preprocessing, **kwargs)

    def get_contact_info(self) -> None:
        print('\nGetting contact info...')
        get_contact_info(self.source, self.paths.contact_info)
        freq_deduped_matches = load_json(
            self.paths.preprocessing / 'freq_deduped_matches.json')
        get_signatures_and_save(freq_deduped_matches,  # type: ignore
                                self.paths.contact_info / 'signatures.json')

    def query_docs(self, query: str | list, top_n: int = 10, query_label: str = 'query') -> None:
        query_label = query_label.replace(' ', '_')
        self.semantic_model.query_docs(query, query_label=query_label)
        df = self.doc_ref.df
        paths = df.nlargest(top_n, query_label).index
        for i, path in enumerate(paths):
            body = get_body_text(path)
            print(f'RESULT {i + 1}:')
            print('---------\n')
            print(body.strip())
            print(f'\n{"="*75}\n')

    def get_entities_from_relevant_docs(self,
                                        filter_query: str | list,
                                        query_label: str = 'query',
                                        query_threshold: float = .3,
                                        orgs_only: bool = True,
                                        n_docs_sample_freq_orgs=500,
                                        occurence_threshold_freq_orgs=.05
                                        ) -> None:
        self.semantic_model.query_docs(
            filter_query, query_label=query_label, save=True)

        get_entities_by_doc_query(self.doc_ref,
                                  self.paths.entities_util,
                                  self.paths.entities,
                                  query_label=query_label,
                                  query_threshold=query_threshold,
                                  orgs_only=orgs_only,
                                  n_docs_sample_freq_orgs=n_docs_sample_freq_orgs,
                                  occurence_threshold_freq_orgs=occurence_threshold_freq_orgs)

    def get_vendors(self, **kwargs):
        kwargs['filter_query'] = kwargs.get(
            'filter_query', ['invoice', 'payment', 'vendor'])
        kwargs['query_label'] = kwargs.get(
            'query_label', 'vendors_default')
        self.get_entities_from_relevant_docs(**kwargs)
