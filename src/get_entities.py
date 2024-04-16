"""Gets named entities from relevant PST docs."""

from collections import Counter
from difflib import SequenceMatcher
from itertools import combinations
from pathlib import Path
import random
import pandas as pd

from tqdm import tqdm

from lang_models.ner import NerCompanyTagger, FlairTagger
from utils.io import dump_json,  load_json
from utils.doc_ref import DocRef
from utils.doc import get_body_text


def group_strs_by_similarity(entities_with_paths: dict[str, set[str]]) -> dict[str, dict[str, set[str] | int]]:
    grouped = {ent: {'paths': paths, 'ents': set(
        [ent])} for ent, paths in entities_with_paths.items()}
    ignore = set()
    for (ent1, paths1), (ent2, paths2) in combinations(entities_with_paths.items(), 2):
        if any((ent1 in ignore, ent2 in ignore, len(ent1) < 5, len(ent2) < 5)):
            continue
        ref_ent,  other_ent,  = (
            ent1, ent2, )if len(paths1) >= len(paths2) else (ent2,  ent1)

        if SequenceMatcher(None, ref_ent.lower(), other_ent.lower(), autojunk=False).ratio() >= .9:
            grouped[ref_ent]['paths'].update(grouped[other_ent]['paths'])
            grouped[ref_ent]['ents'].update(grouped[other_ent]['ents'])
            grouped.pop(other_ent)
            ignore.add(other_ent)
    for ref_ent, d in grouped.items():
        d['count'] = len(d['paths'])  # type: ignore
        d.pop('paths')
    return grouped  # type: ignore


def add_entities_to_doc_ref(paths: list[Path],
                            doc_ref: DocRef,
                            org_tagger: NerCompanyTagger,
                            general_tagger: FlairTagger | None = None
                            ) -> None:

    updated = 0
    for path in tqdm(paths):
        text = get_body_text(path, incl_subj=True)
        if not doc_ref.is_doc_tagged(path, 'ORG'):
            orgs = org_tagger(text)
            doc_ref.add_ents(path, orgs, orgs_only=True)
        if general_tagger and not doc_ref.is_doc_tagged(path):
            entities_by_tag = general_tagger(text)
            doc_ref.add_ents(path,  entities_by_tag)
        updated += 1
        if updated == 100:
            doc_ref.save()
            updated = 0
    doc_ref.save()


def get_freq_entities_to_exclude(doc_ref: DocRef,
                                 util_folder: Path,
                                 org_tagger: NerCompanyTagger,
                                 general_tagger: FlairTagger | None,
                                 n_docs_sample_freq_orgs: int,
                                 occurence_threshold_freq_orgs: float
                                 ) -> set[str]:

    grouped_freq_orgs_path = util_folder/'grouped_freq_orgs.json'
    if not grouped_freq_orgs_path.is_file():
        print(
            f'\nGetting entities from {n_docs_sample_freq_orgs} random docs...')

        paths = doc_ref.get_paths()
        random.shuffle(paths)
        paths = paths[:n_docs_sample_freq_orgs]

        add_entities_to_doc_ref(
            paths, doc_ref, org_tagger=org_tagger, general_tagger=general_tagger)

        orgs_with_paths = {}
        for path in paths:
            for org in doc_ref.get_ents(path, 'ORG'):  # type: ignore
                orgs_with_paths.setdefault(org, set())
                orgs_with_paths[org].add(path)

        grouped_freq_orgs = group_strs_by_similarity(orgs_with_paths)
        dump_json(grouped_freq_orgs, grouped_freq_orgs_path)

    else:
        grouped_freq_orgs = load_json(grouped_freq_orgs_path)

    min_count = n_docs_sample_freq_orgs * occurence_threshold_freq_orgs
    ents_to_exclude = set()
    for ref_org, d in grouped_freq_orgs.items():
        if d['count'] >= min_count:  # type: ignore
            ents_to_exclude.update(d['ents'])  # type: ignore
    return ents_to_exclude


def rank_and_save_orgs_and_related_info(doc_ref: DocRef, doc_ref_paths: list[Path], output: Path, orgs_to_exclude: set[str]) -> None:
    # Ranks and gets associated email addrs
    orgs_by_doc_query = Counter()
    orgs_with_associated_email_addrs = {}
    for path in doc_ref_paths:
        for org in doc_ref.get_ents(path, 'ORG'):  # type: ignore
            if org not in orgs_to_exclude:
                orgs_by_doc_query[org] += 1
                orgs_with_associated_email_addrs.setdefault(org, Counter())
                for email_addr_d in (
                        [doc_ref.get_sender(path)] + doc_ref.get_recipients(path)):  # type: ignore
                    if isinstance(email_addr_d, dict):
                        orgs_with_associated_email_addrs[org][email_addr_d['addr']] += 1
    save_dict = {}
    # Save to HTML table

    for org, doc_count in orgs_by_doc_query.most_common():
        email_addrs = [addr for addr, count in orgs_with_associated_email_addrs[org].most_common(
            5) if '@' in addr]
        save_dict[org] = [doc_count, ', '.join(email_addrs)]

    cols = ['doc count', 'associated emails']
    save_df = pd.DataFrame.from_dict(save_dict, orient='index', columns=cols)
    save_df['doc count'] = save_df['doc count'].astype(int)
    save_df.index.name = 'Organization'
    save_df.to_markdown(output.with_suffix('.md'))
    save_df.to_json(output.with_suffix('.json'), index=False)


def get_entities_by_doc_query(doc_ref: DocRef,
                              util_folder: Path,
                              output_folder:  Path,
                              query_label: str = 'query',
                              query_threshold: float = .25,
                              orgs_only: bool = True,
                              n_docs_sample_freq_orgs: int = 250,
                              occurence_threshold_freq_orgs: float = .05
                              ) -> None:
    """Gets entities from docs relevant to query.

    Args:
        doc_ref (DocRef): Document reference 
        util_folder (Path): Path for utility files.
        output_folder (Path): Output path.
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

    Raises:
        NotImplementedError: Only implemented for org entities.
    """
    if not orgs_only:
        raise NotImplementedError

    # Initializes taggers
    org_tagger = NerCompanyTagger()
    general_tagger = None if orgs_only else FlairTagger()
    # Gets docs that meet query_threshold and have not yet been tagged
    paths_to_process = doc_ref.get_paths_to_process(
        query_label, query_threshold)
    # Gets entities from docs
    add_entities_to_doc_ref(
        paths_to_process, doc_ref, org_tagger=org_tagger, general_tagger=general_tagger)
    # Gets all paths that match queries.
    query_paths = doc_ref.get_paths_by_query(query_label, query_threshold)
    # Gets frequent entities in a random sample of docs to exclude from results
    orgs_to_exclude = get_freq_entities_to_exclude(
        doc_ref, util_folder, org_tagger, general_tagger, n_docs_sample_freq_orgs, occurence_threshold_freq_orgs)
    # Ranks orgs by doc frequency and gets associated people; saves to csv
    output = output_folder / f'{query_label}.csv'
    rank_and_save_orgs_and_related_info(
        doc_ref, query_paths,  output, orgs_to_exclude)
