"""
Removes frequently occurring redundant blocks of text, such as e-mail footers,
signature blocks, etc.

Samples n email's body text from sent and received folders to find frequently 
occurring sections of text and adds a version of the body with those sections
removed to each email JSON.

Procedure : 

(1) Fuzzy matches text strings in sample emails.
(2) Excludes matched groups that don't occur frequently enough in the sample
(3) Dedupes the remaining text strings.
(4) Uses the deduped strings to match text in the full set of emails, expanding
    selection beyond matched text to the nearest paragraph boundary ('\n\n')

Parameters :

    docs_folder : Folder with email JSONs. 
    util : Folder for util files.
    kwargs : 
        min_match_size': Minimum character length to consider text for removal,
            default 100
        n_docs_to_sample': How many docs to sample to find frequently occuring
            blocks, default 100
        match_threshold': SequenceMatcher.ratio threshold for matching blocks 
            when sampling , default .9
        min_prop_occurrence : Minimum proportion of occurence of matched set
            in sample docs to qualify for removal.
        dedupe_threshold': thefuzz.process.dedupe ratio for deduping blocks 
            of text, default 90 
        dedupe_scorer': thefuzz.process.dedupe scoring function, default 
            fuzz.ratio
        do_get_freq_blocks': Step 1, default True
        do_group_matches': Step 2, default True
        do_dedupe': Step 3, default True
        do_remove_text_from_docs': Step 4, default True
        
"""

from collections import Counter
from difflib import SequenceMatcher
import os
import random
import re

from thefuzz.process import dedupe as fuzz_dedupe
from thefuzz import fuzz
from tqdm import tqdm

from utils import dump_json, load_json


def strip_bad_chars(doc, split_paras=False):
    doc = doc.replace('\u00a0', ' ')
    doc = re.sub(r'[ \t]+', ' ', doc)
    doc = re.sub(r'\n>+', '\n', doc)
    final_doc = []
    for block in doc.split('\n\n'):
        block.strip()
        block.strip('\n')
        final_doc.append(block)
    if not split_paras:
        final_doc = '\n\n'.join(final_doc)
    return final_doc


def get_matches_in_pair(doc1, doc2, min_match_size, match_threshold=.9):
    match_pairs = []
    matched_doc1_indices = set()
    matched_doc2_indices = set()
    for i1, line1 in enumerate(doc1):
        if i1 in matched_doc1_indices:
            continue
        for i2,  line2 in enumerate(doc2):
            if i2 in matched_doc2_indices:
                continue
            current_match1 = []
            current_match2 = []
            if SequenceMatcher(None, line1, line2, autojunk=True).ratio() < match_threshold:
                continue
            matched_doc1_indices.add(i1)
            matched_doc2_indices.add(i2)
            current_match1.append(line1)
            current_match2.append(line2)
            for prev_i1, prev_i2 in reversed(list(zip(range(0, i1), range(0, i2)))):
                prev_line1 = doc1[prev_i1]
                prev_line2 = doc2[prev_i2]
                if SequenceMatcher(None, prev_line1, prev_line2, autojunk=True).ratio() >= match_threshold:
                    current_match1.insert(0, prev_line1)
                    current_match2.insert(0, prev_line2)
                    matched_doc1_indices.add(i1 - 1)
                    matched_doc2_indices.add(i2 - 1)
                else:
                    break
            for next_i1, next_i2 in zip(range(i1 + 1, len(doc1)), range(i2 + 1, len(doc2))):
                next_line1 = doc1[next_i1]
                next_line2 = doc2[next_i2]
                if SequenceMatcher(None, next_line1, next_line2, autojunk=True).ratio() >= match_threshold:
                    current_match1.append(next_line1)
                    current_match2.append(next_line2)
                    matched_doc1_indices.add(i1 + 1)
                    matched_doc2_indices.add(i2 + 1)
                else:
                    break
            match1 = '\n\n'.join(current_match1).strip('\n')
            match2 = '\n\n'.join(current_match2).strip('\n')
            if len(match1) >= min_match_size or len(match2) >= min_match_size:
                match_pairs.append([match1, match2])
    return match_pairs


def add_to_match_dict(match_dict, match_pairs, paths):
    for match_pair in match_pairs:
        target_match_sets = set()
        for match in match_pair:
            for match_set in match_dict:
                if match in match_set:
                    target_match_sets.add(match_set)
                    break
        if not target_match_sets:
            match_dict[frozenset(match_pair)] = set(paths)
            continue
        else:
            new_match_set = frozenset().union(*target_match_sets, match_pair)
            new_match_paths = set(paths)
            for match_set in target_match_sets:
                new_match_paths.update(match_dict.pop(match_set))
            match_dict[new_match_set] = new_match_paths


def group_matches(match_sets, match_threshold=.9):
    grouped_matches = {}
    completed_pairs = set()
    for i1, (match_set1, path_set1) in tqdm(list(enumerate(match_sets)), desc='Grouping matches...'):
        new_results_set = set(match_set1)
        new_path_set = set(path_set1)
        tester = max(match_set1)
        for i2, (match_set2, path_set2) in enumerate(match_sets):
            if i2 == i1 or set([i1, i2]) in completed_pairs or not match_set2:
                continue
            completed_pairs.add(frozenset([i1, i2]))
            ratio = SequenceMatcher(
                None, tester, max(match_set2), autojunk=False).ratio()
            if ratio >= match_threshold:
                new_results_set.update(match_set2)
                new_path_set.update(path_set2)
                match_sets[i2] = [None, None]
        grouped_matches[frozenset(new_results_set)] = new_path_set
    grouped_matches = sorted(
        [(x, y) for x, y in grouped_matches.items()], key=lambda x: len(x[1]), reverse=True)
    return grouped_matches


def dedupe_matches(match_sets, min_doc_ocurrence, dedupe_threshold=70, scorer=fuzz.ratio):
    print('Deduping matches...')
    to_dedupe = []
    for result, path_set in match_sets:
        if len(path_set) >= min_doc_ocurrence:
            to_dedupe.extend(result)
    deduped_matches = fuzz_dedupe(to_dedupe, dedupe_threshold, scorer=scorer)
    return deduped_matches


def remove_text_from_docs(doc_paths,  ref_matches, min_match_size):
    for path in tqdm(doc_paths, desc='Removing text from emails...'):
        email_dict = load_json(path)
        doc = email_dict.get('bodyText', '')
        doc = strip_bad_chars(doc)
        if not doc:
            continue
        for ref_match in ref_matches:
            matcher = SequenceMatcher(None, ref_match, doc)
            for match in matcher.get_matching_blocks():
                if match.size >= min_match_size:
                    match_str = ref_match[match.a:match.a+match.size]
                    if match.b > 1 and doc[match.b-2: match.b] != '\n\n':
                        try:
                            match_str = doc[doc.rindex(
                                '\n\n'):match.b] + match_str
                        except ValueError:
                            # Just go to beginning of file?
                            pass
                    if match.b + match.size < len(doc) - 1 and doc[match.b + match.size: match.b + match.size + 2] != '\n\n':
                        try:
                            match_str = match_str + doc[match.b +
                                                        match.size:doc.index('\n\n')]
                        except ValueError:
                            # Just go to end of file?
                            pass
                    doc = doc.replace(match_str, '\n\n')

        doc = re.sub(r'\n{3,}', '\n', doc)
        email_dict['bodyTextFiltered'] = doc
        dump_json(email_dict, path)


def main(doc_paths, util_folder, **kwargs):

    min_match_size = kwargs.get('min_match_size', 100)
    n_docs_to_sample = kwargs.get('n_docs_to_sample', 100)
    match_threshold = kwargs.get('match_threshold', .9)
    min_prop_occurrence = kwargs.get('min_prop_occurrence', .10)
    dedupe_threshold = kwargs.get('dedupe_threshold', 70)
    dedupe_scorer = kwargs.get('dedupe_scorer', fuzz.ratio)
    do_get_freq_blocks = kwargs.get('do_get_freq_blocks', True)
    do_group_matches = kwargs.get('do_group_matches', True)
    do_dedupe = kwargs.get('do_dedupe', True)
    do_remove_text_from_docs = kwargs.get('do_remove_text_from_docs', True)

    os.makedirs(util_folder, exist_ok=True)

    sent_paths, received_paths = [], []
    for path in doc_paths:
        if 'Sent Items' in path:
            sent_paths.append(path)
        else:
            received_paths.append(path)

    for paths in (sent_paths, received_paths):
        label = 'sent' if paths == sent_paths else 'received'
        print(f'Working on {label} emails...')

        if do_get_freq_blocks:
            random.shuffle(paths)
            paths = paths[:n_docs_to_sample]
            match_dict = {}
            completed_doc_pairs = set()
            for path1 in tqdm(paths, desc='Getting matches...'):
                doc1 = load_json(path1).get('bodyText', '').strip()
                doc1 = strip_bad_chars(doc1, split_paras=True)
                for path2 in paths:
                    if path1 == path2 or set([path1, path2]) in completed_doc_pairs:
                        continue
                    completed_doc_pairs.add(frozenset([path1, path2]))
                    doc2 = load_json(path2).get('bodyText', '').strip()
                    doc2 = strip_bad_chars(doc2, split_paras=True)
                    match_pairs = get_matches_in_pair(
                        doc1, doc2, min_match_size, match_threshold)
                    add_to_match_dict(match_dict, match_pairs, [path1, path2])

            counter = Counter()
            for match_set, path_set in match_dict.items():
                counter[match_set] = len(path_set)
            counter = Counter(match_dict).most_common()
            grouped = []
            for match_set, path_set in counter:
                grouped.append(
                    [match_set, match_dict[frozenset(match_set)]])
            path = os.path.join(util_folder, f'{label}_matches.json')
            dump_json(grouped, path, default=list)
        else:
            grouped = load_json(
                os.path.join(util_folder, f'{label}_matches.json'))

        if do_group_matches:
            grouped_matches = group_matches(
                grouped, match_threshold)
            path = os.path.join(util_folder, f'{label}_grouped_matches.json')
            dump_json(grouped_matches, path, default=list)

    if do_dedupe:
        grouped_matches = load_json(
            os.path.join(util_folder, 'sent_grouped_matches.json')) + load_json(os.path.join(util_folder, 'received_grouped_matches.json'))
        min_doc_ocurrence = n_docs_to_sample * min_prop_occurrence
        deduped_matches = dedupe_matches(
            grouped_matches, min_doc_ocurrence, dedupe_threshold, dedupe_scorer)
        path = os.path.join(util_folder, f'text_to_remove.json')
        dump_json(deduped_matches, path)
    else:
        deduped_matches = load_json(
            os.path.join(util_folder, f'text_to_remove.json'))

    if do_remove_text_from_docs:
        remove_text_from_docs(
            doc_paths, deduped_matches, min_match_size)
