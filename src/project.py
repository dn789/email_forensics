"""
Analyzes a company or organization's emails and other PST contents.

"""

from collections import namedtuple
from pathlib import Path

from clean_body_text import clean_body_text
from get_entities import get_entities_by_doc_query
from get_contact_info import main as get_contact_info, save_signatures_from_all_docs
from lang_models.semantic import SemanticModel
from preprocess.preprocess import preprocess
from utils.io import load_json, dump_json
from utils.doc_ref import DocRef


class Project:
    """
    Analyzes a company or organization's emails and other PST contents.
    """

    def __init__(self,
                 source: Path,
                 project_folder: Path,
                 **kwargs
                 ) -> None:
        """
        Preprocessing, cleaning text and getting vendors is done automatically 
        when a project is initalized in a folder for the first time.

        Args:
            source (Path): Path of folder containing PSTs or email files, or 
                path of a single PST file. Each PST file or subfolder should
                represent an individual's communications. 
            project_folder (Path): Path for output files.
            kwargs: {
                "already_preprocessed" (bool): Skips preprocessing step. Only
                    need this if you're initializing a project in a new folder
                    with already preprocessed docs in docs subfolder. 
                "semantic_model_name" (str): Model name from 
                    https://www.sbert.net/docs/pretrained_models.html.
                "clean_body_text" (dict): {clean_body_text kwargs}.
                "get_vendors" (dict): {get_vendors kwargs}.
            }
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

        if not checklist.get('preprocess') and not kwargs.get('already_preprocessed'):
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
        """
        Normailizes spacing and removes redundant text blocks (footers, etc.) 
        from a set of emails.

        Kwargs:
            doc_ref (DocRef): Document reference.
            util_folder (Path): Path for utility files. 
            min_match_size (int, optional): Min. char count for redundant text 
                blocks. Defaults to 100.
            match_ratio_threshold (float, optional): Min. similarity ratio for 
                redundant text blocks. Defaults to .9.
            match_prevalence_threshold (float, optional): Text blocks that occur 
                in at least this proportion of docs will be considered redunant. 
                Defaults to .1.
            sample_n_docs (int, optional): N docs to sample for redundant text 
                blocks. Defaults to 50.
            sequence_matcher_autojunk (bool, optional): Faster if True. Defaults 
                to True.
        """
        print('\nCleaning body text...')

        clean_body_text(
            self.doc_ref, self.paths.clean_body_text, **kwargs)

    def get_contact_info(self) -> None:
        print('\nGetting contact info...')
        get_contact_info(self.doc_ref, self.paths.contact_info)
        freq_deduped_matches = load_json(
            self.paths.clean_body_text / 'freq_deduped_matches.json')
        save_signatures_from_all_docs(freq_deduped_matches,  # type: ignore
                                      self.paths.contact_info / 'signatures.json')

    def query_docs(self, query: str | list, top_n: int = 10, query_label: str = 'query', show_score: bool = True) -> None:
        query_label = query_label.replace(' ', '_')
        self.semantic_model.query_docs(
            query, query_label=query_label, show_top_n=top_n, show_score=show_score)

    def get_entities_from_relevant_docs(self,
                                        filter_query: str | list,
                                        query_label: str = 'query',
                                        query_threshold: float = .3,
                                        orgs_only: bool = True,
                                        n_docs_sample_freq_orgs=250,
                                        occurence_threshold_freq_orgs=.05
                                        ) -> None:
        """Gets entities from docs relevant to query.

        Args:
            filter_query (str | list): Query string or list of query strings. 
                Highest-scoring query is used to check relevance. 
        query_label (str, optional): Used to name output file. Defaults to 
            'query'.
        query_threshold (float, optional): Minumum similarity score for 
            relevant documents. Defaults to .3.
        orgs_only (bool, optional): Only get organization entities. Defaults to 
            True.
        n_docs_sample_freq_orgs (int, optional): Number of random docs to get
            org entities from. Used for filtering out irrelevant entities from 
            results. Defaults to 250.
        occurence_threshold_freq_orgs (float, optional): Entities occuring in 
            at least this proportion of random docs will be filtered out . 
            Defaults to .05.
        """
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
        """
        Gets vendor names by getting entities from docs relevant to invoices,
        payment, etc.

        Kwargs:
            filter_query (str | list): Query string or list of query strings. 
                Highest-scoring query is used to check relevance. Defaults
                to ['invoice', 'payment', 'vendor']
            query_label (str, optional): Used to name output file. Defaults to 
                'vendors_default'.
            query_threshold (float, optional): Minumum similarity score for 
                relevant documents. Defaults to .3.
            orgs_only (bool, optional): Only get organization entities. Defaults to 
                True.
            n_docs_sample_freq_orgs (int, optional): Number of random docs to get
                org entities from. Used for filtering out irrelevant entities from 
                results. Defaults to 250.
            occurence_threshold_freq_orgs (float, optional): Entities occuring in 
                at least this proportion of random docs will be filtered out . 
                Defaults to .05.
        """
        print('\nGetting vendor names...')
        kwargs['filter_query'] = kwargs.get(
            'filter_query', ['invoice', 'payment', 'vendor'])
        kwargs['query_label'] = kwargs.get(
            'query_label', 'vendors_default')
        self.get_entities_from_relevant_docs(**kwargs)
