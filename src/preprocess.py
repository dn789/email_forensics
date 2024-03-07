"""Preprocesses PST docs' body text, esp. by removing redunant text blocks."""

from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any
import random
import re


from tqdm import tqdm
from utils.io import dump_json, load_json
from utils.doc_ref import DocRef
from utils.doc import add_preprocessed_body_text, get_body_text


SPACE_CHARS = '\t\x0b\x0c\r\x1c\x1d\x1e\x1f \x85\xa0\u1680\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u2028\u2029\u202f\u205f\u3000'


def normalize_body_text_for_matching(body: str) -> str:
    """Adds leading and trailing '\n\n' to text."""

    body = re.sub(rf'[{SPACE_CHARS}]+', ' ', body)
    body = re.sub(r'\n(> )+', '\n', body)
    body = '\n\n'.join([block.strip()
                        for block in body.split('\n\n') if block.strip()])
    body = '\n\n' + body + '\n\n'
    return body


def normalize_body_text_after_dedupe(body: str) -> str:
    body = '\n\n'.join([block.strip()
                        for block in body.split('\n\n') if block.strip()])
    return body


def get_matches_in_body_pair(text1: str,
                             text2: str,
                             min_match_size: int = 100,
                             sequence_matcher_autojunk: bool = True) -> list[str]:
    """Pass text to normalize_body_text_for_matching first."""

    matches = []

    match_blocks = SequenceMatcher(
        None, text1, text2, autojunk=sequence_matcher_autojunk).get_matching_blocks()

    for match in match_blocks:
        if match.size < min_match_size:
            continue
        match_str = text1[match.a:match.a+match.size]

        if match_str.startswith('\n\n') and match_str.endswith('\n\n'):
            matches.append(match_str.strip())
            continue

        if not match_str.startswith('\n\n'):
            try:
                match_str = match_str[match_str.index('\n\n'):]
            except ValueError:
                pass

        if not match_str.endswith('\n\n'):
            try:
                match_str = match_str[:match_str.rindex('\n\n')]
            except ValueError:
                pass

        if len(match_str) >= min_match_size:
            matches.append(match_str.strip())

    return matches


def get_matches_in_bodies(paths: list[Path], min_match_size=100, sequence_matcher_autojunk: bool = True) -> list[tuple[str, list]]:
    completed_doc_pairs = set()
    matches_with_paths_d = {}

    for path1 in tqdm(paths):
        body1 = get_body_text(path1, preprocessed=False)
        if not body1:
            continue
        text1 = normalize_body_text_for_matching(body1)
        for path2 in paths:
            if path1 == path2 or set([path1, path2]) in completed_doc_pairs:
                continue
            completed_doc_pairs.add(frozenset([path1, path2]))
            body2 = get_body_text(path2, preprocessed=False)
            if not body2:
                continue
            text2 = normalize_body_text_for_matching(body2)
            matches = get_matches_in_body_pair(
                text1, text2,
                min_match_size=min_match_size,
                sequence_matcher_autojunk=sequence_matcher_autojunk)
            for match in matches:
                matches_with_paths_d.setdefault(match, set())
                matches_with_paths_d[match].update([path1, path2])

    matched_with_paths_ranked = Counter(matches_with_paths_d).most_common()
    return matched_with_paths_ranked  # type: ignore


def get_freq_deduped_matches(matches_and_paths: list[tuple[Any, Any]],
                             match_ratio_threshold: float = .9,
                             doc_count_threshold: int = 10,
                             sequence_matcher_autojunk: bool = True) -> list[str]:
    deduped_matches = {}
    completed_pairs = set()
    for i1, (match1, path_set1) in tqdm(list(enumerate(matches_and_paths))):
        if not match1:
            continue
        deduped_match = match1
        deduped_match_paths = set(path_set1)  # type: ignore
        matches_and_paths[i1] = None, None
        for i2, (match2, path_set2) in enumerate(matches_and_paths):
            if i1 == i2 or set([i1, i2]) in completed_pairs or not match2:
                continue
            completed_pairs.add(frozenset([i1, i2]))
            ratio = SequenceMatcher(
                None, deduped_match, match2, autojunk=sequence_matcher_autojunk).ratio()
            if ratio >= match_ratio_threshold:
                deduped_match = min((deduped_match, match2), key=len)
                deduped_match_paths.update(path_set2)
                matches_and_paths[i2] = None, None

        deduped_matches[deduped_match] = deduped_match_paths

    freq_deduped_matches = []
    for match, paths in Counter(deduped_matches).most_common():
        if len(paths) >= doc_count_threshold:  # type: ignore
            freq_deduped_matches.append(match)
    return freq_deduped_matches


def remove_freq_matches_from_bodies(matches_to_remove: list[str],
                                    paths: list[Path],
                                    min_match_size: int = 100,
                                    sequence_matcher_autojunk: bool = True) -> None:
    for path in tqdm(paths):
        body = get_body_text(path, preprocessed=False)
        if not body:
            continue
        body = normalize_body_text_for_matching(body)

        for match_to_remove in matches_to_remove:
            for match_block in SequenceMatcher(None, match_to_remove, body, autojunk=sequence_matcher_autojunk).get_matching_blocks():
                if match_block.size < min_match_size:
                    continue
                match_str = match_to_remove[match_block.a:
                                            match_block.a + match_block.size]
                body = body.replace(match_str, '')

        body = normalize_body_text_after_dedupe(body)
        add_preprocessed_body_text(path, body)


def preprocess_doc_bodies(
        doc_ref: DocRef,
        util_folder: Path,
        min_match_size=100,
        match_ratio_threshold: float = .9,
        match_prevalence_threshold: float = .1,
        sample_n_docs: int = 50,
        sequence_matcher_autojunk: bool = True) -> None:
    """_summary_

    Args:
        doc_ref (DocRef): _description_
        util_folder (Path): _description_
        min_match_size (int, optional): _description_. Defaults to 100.
        match_ratio_threshold (float, optional): _description_. Defaults to .9.
        match_prevalence_threshold (float, optional): _description_. Defaults to .1.
        sample_n_docs (int, optional): _description_. Defaults to 50.
        sequence_matcher_autojunk (bool, optional): _description_. Defaults to True.
    """
    paths_d = {'sent': [], 'received': []}

    for path in doc_ref.get_paths():
        if 'Sent Items' in path.__str__():
            paths_d['sent'].append(path)
        else:
            paths_d['received'].append(path)

    matches_with_paths = []

    for label, paths_subset in paths_d.items():

        print(f'Getting matching text blocks in {label} emails...')

        random.shuffle(paths_subset)
        paths_subset = paths_subset[:sample_n_docs]

        matches_with_paths_subset = get_matches_in_bodies(
            paths_subset,
            min_match_size=min_match_size,
            sequence_matcher_autojunk=sequence_matcher_autojunk)
        dump_json(matches_with_paths_subset,
                  util_folder / f'{label}_matches.json', )
        matches_with_paths.extend(matches_with_paths_subset)

    doc_count_threshold = int(sample_n_docs * match_prevalence_threshold)

    print('Deduping frequent text blocks...')

    freq_matches = get_freq_deduped_matches(
        matches_with_paths,
        match_ratio_threshold=match_ratio_threshold,
        doc_count_threshold=doc_count_threshold,
        sequence_matcher_autojunk=sequence_matcher_autojunk)
    dump_json(freq_matches, util_folder /
              'freq_deduped_matches.json', )

    paths = paths_d['sent'] + paths_d['received']

    print('Removing frequent text blocks from email bodies...')

    remove_freq_matches_from_bodies(freq_matches,
                                    paths,
                                    min_match_size=min_match_size,
                                    sequence_matcher_autojunk=sequence_matcher_autojunk)
