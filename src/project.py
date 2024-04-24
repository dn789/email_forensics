"""
Analyzes a company or organization's emails and other PST contents.

"""

from collections import namedtuple
from pathlib import Path
import logging

from clean_body_text import clean_body_text
from get_entities import get_entities_by_doc_query
from get_contact_info import main as get_contact_info, save_signatures_from_all_docs
from lang_models.semantic import SemanticModel
from preprocess.preprocess import preprocess
from utils.io import load_json, dump_json
from utils.doc_ref import DocRef
from security_scan.find_secrets import find_secrets_in_docs


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

        logging.basicConfig(filename=self.paths.util / 'errors.log',
                            level=logging.ERROR)

        if self.paths.checklist.is_file():
            checklist = load_json(self.paths.checklist)
        else:
            checklist = {}

        if not checklist.get('preprocess') and not kwargs.get('already_preprocessed'):
            self.preprocess()
            checklist['preprocess'] = True
            dump_json(checklist, self.paths.checklist)

        # Makes or loads doc ref.
        self.doc_ref = DocRef(self.paths.docs,
                              self.paths.util / 'doc_ref.pkl')

        checklist_items = [
            'clean_body_text',
            'get_contact_info',
            'find_secrets',
            'semantic_model',
            'get_vendors'
        ]

        for item in checklist_items:
            if item == 'semantic_model':
                # Loads semantic model and makes or laods doc embeddings.
                self.semantic_model = SemanticModel(
                    self.semantic_model_name, self.paths.doc_embeds, self.doc_ref)
                continue
            if not checklist.get(item):
                getattr(self, item)(**kwargs.get(item, {}))
                checklist[item] = True
                dump_json(checklist, self.paths.checklist)

    def preprocess(self):
        print('\nPreprocessing input...')
        try:
            preprocess(self.source, self.paths.docs)
        except Exception as e:
            logging.exception(f'{type(e).__name__}: {e}')
            raise e

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
        try:
            clean_body_text(
                self.doc_ref, self.paths.clean_body_text, **kwargs)
        except Exception as e:
            logging.exception(f'{type(e).__name__}: {e}')
            raise e

    def get_contact_info(self) -> None:
        try:
            print('\nGetting contact info...')
            get_contact_info(self.doc_ref, self.paths.contact_info)
            freq_deduped_matches = load_json(
                self.paths.clean_body_text / 'freq_deduped_matches.json')
            save_signatures_from_all_docs(freq_deduped_matches,  # type: ignore
                                          self.paths.contact_info / 'signatures.json')
        except Exception as e:
            logging.exception(f'{type(e).__name__}: {e}')
            print(f'{type(e).__name__}: {e}')

    def query_docs(self, query: str | list, top_n: int = 10, show_score: bool = True, save: bool = False) -> None:
        query_label = query if isinstance(query, str) else '_'.join(query)
        query_label = query_label.replace(' ', '_')
        self.semantic_model.query_docs(
            query, query_label=query_label, show_top_n=top_n, show_score=show_score, save=save)

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
        try:
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
        except Exception as e:
            logging.exception("Exception occurred: %s", str(e))
            print(f'{type(e).__name__}: {e}')

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
        try:
            kwargs['filter_query'] = kwargs.get(
                'filter_query', ['invoice', 'payment', 'vendor'])
            kwargs['query_label'] = kwargs.get(
                'query_label', 'vendors_default')
            self.get_entities_from_relevant_docs(**kwargs)
        except Exception as e:
            logging.exception(f'{type(e).__name__}: {e}')
            print(f'{type(e).__name__}: {e}')

    def find_secrets(self, gitleaks_path: Path | None = None, gitleaks_config: Path | None = None) -> None:
        """Uses gitleaks to find passwords, API keys, etc. in body text.

        Args:
            gitleaks_path (Path | None, optional): Path to gitleaks executable. 
                Defaults to None.
            gitleaks_config (Path | None, optional): Path to gitleaks config 
                toml file. Defaults to None.
        """
        if gitleaks_path:
            print('\nFinding secrets...')
            try:
                find_secrets_in_docs(gitleaks_path,
                                     self.doc_ref, self.project_folder, config=gitleaks_config)
            except Exception as e:
                logging.exception(f'{type(e).__name__}: {e}')
                print(f'{type(e).__name__}: {e}')
