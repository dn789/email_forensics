"""
Analyzes a company or organization's emails and other PST contents and 
saves results.

"""

from collections import namedtuple
from pathlib import Path

from clean_body_text import clean_body_text
from get_entities import get_entities_by_doc_query
from get_contact_info import main as get_contact_info, get_signatures_and_save
from lang_models.semantic import SemanticModel
from preprocess.preprocess import preprocess
from utils.io import load_json, dump_json
from utils.doc_ref import DocRef


class Project():
    """
    """

    def __init__(self,
                 source: Path,
                 project_folder: Path,
                 **kwargs
                 ) -> None:
        """
        """
        # Set up paths
        self.source = source
        self.project_folder = project_folder
        self.semantic_model_name = kwargs['semantic_model_name']

        folders = {
            'docs': '',
            'output': '',
            'util': '_util',
            'clean_body_text': '_util/clean_body_text',
            'doc_embeds': '_util/doc_embeds',
            'entities_util': '_util/entities',
            'contact_info': 'output/contact_info',
            'entities': 'output/entities'
        }

        paths = {}
        for name, folder in folders.items():
            folder = folder or name
            paths[name] = path = self.project_folder / folder
            path.mkdir(parents=True, exist_ok=True)
        paths['project_folder'] = self.project_folder
        paths['checklist'] = paths['util'] / 'checklist.json'
        Paths = namedtuple('Paths', paths)
        self.paths = Paths(**paths)

        if self.paths.checklist.is_file():
            checklist = load_json(self.paths.checklist)
        else:
            checklist = {}

        if not checklist.get('preprocess'):
            self.preprocess()
            checklist['preprocess'] = True
            dump_json(checklist, self.paths.checklist)

        # Makes or loads doc ref
        self.doc_ref = DocRef(self.paths.docs,
                              self.paths.util / 'doc_ref.pkl')

        if not checklist.get('clean_body_text'):
            self.clean_body_text(**kwargs.get('clean_body_text', {}))
            checklist['clean_body_text'] = True
            dump_json(checklist, self.paths.checklist)

        if not checklist.get('get_contact_info'):
            self.get_contact_info()
            checklist['get_contact_info'] = True
            dump_json(checklist, self.paths.checklist)

        # Loads semantic model and get doc embeddings if they donn't exist
        self.semantic_model = SemanticModel(
            self.semantic_model_name, self.paths.doc_embeds, self.doc_ref)

        if not checklist.get('get_vendors'):
            self.get_vendors(**kwargs.get('get_vendors', {}))
            checklist['get_vendors'] = True
            dump_json(checklist, self.paths.checklist)

    def preprocess(self):
        print('\nPreprocessing input...')
        preprocess(self.source, self.paths.docs)

    def clean_body_text(self, **kwargs) -> None:
        print('\nCleaning body text...')
        clean_body_text(
            self.doc_ref, self.paths.clean_body_text, **kwargs)

    def get_contact_info(self) -> None:
        print('\nGetting contact info...')
        get_contact_info(self.paths.docs, self.paths.contact_info)
        freq_deduped_matches = load_json(
            self.paths.clean_body_text / 'freq_deduped_matches.json')
        get_signatures_and_save(freq_deduped_matches,  # type: ignore
                                self.paths.contact_info / 'signatures.json')

    def query_docs(self, query: str | list, top_n: int = 10, query_label: str = 'query') -> None:
        query_label = query_label.replace(' ', '_')
        self.semantic_model.query_docs(
            query, query_label=query_label, show_top_n=top_n)

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
