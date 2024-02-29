from ast import literal_eval
from collections import Counter, namedtuple
from difflib import SequenceMatcher
from pathlib import Path
from typing import Callable
import pickle
import random

import pandas as pd
from sentence_transformers import SentenceTransformer, util
from thefuzz.process import fuzz
from tqdm import tqdm

from models import KwModel, NerCompanyTagger
from utils import load_json, get_sender, get_recipients


def semantic_search_kw(keywords, model, ref_embeds, metric=util.cos_sim, **kwargs):
    best_score = 0
    for kw in keywords:
        words = kw.split()
        word_embeds = model.encode(words)
        for ref_embed in ref_embeds:
            scores = metric(ref_embed, word_embeds)[0].cpu().tolist()
            max_scores = max(scores)
            if best_score < max_scores:
                best_score = max_scores
    return best_score


def semantic_search_doc(doc_embed, ref_embeds, metric=util.cos_sim, **kwargs):
    scores = []
    for ref_embed in ref_embeds:
        scores.append(metric(ref_embed, doc_embed)[0].cpu().tolist()[0])
    return max(scores)


def get_keywords(args):

    kw_model = KwModel(args.kw_kwargs)

    for col_name in ('keywords', 'got_keywords'):
        if col_name not in args.doc_ref:
            args.doc_ref[col_name] = None

    filtered_ref = args.doc_ref[(args.doc_ref['keywords'].isnull()) & (
        args.doc_ref['embed_index'] >= 0)]

    updated = False
    if not args.batch_size:
        for row in tqdm(list(filtered_ref.itertuples()), 'Getting keywords'):
            if row.embed_index % 500 and updated:
                args.doc_ref.to_pickle(args.doc_ref_path)
                updated = False
            email_d = load_json(row.path)
            subject = email_d.get('subject', '')
            body = email_d.get('bodyTextPreprocessed', '')
            doc = '\n\n'.join([subject, body]).strip()

            try:
                kw_sets = kw_model(doc)
                args.doc_ref.at[row.Index, 'keywords'] = kw_sets
                args.doc_ref.loc[row.Index, 'got_keywords'] = True
                updated = True
            except Exception:
                args.doc_ref.loc[row.Index, 'got_keywords'] = True
        args.doc_ref.to_pickle(args.doc_ref_path)

    else:
        paths = [(row.Index, row.path)
                 for row in args.filtered_ref.itertuples()]
        batches = [paths[i:i + args.batch_size]
                   for i in range(0, len(paths), args.batch_size)]
        for batch in tqdm(batches, f'Getting keywords from batches of {args.batch_size} docs'):
            docs = []
            for i, file_path in batch:
                email_d = load_json(file_path) or {}
                subject = email_d.get('subject', '')
                body = email_d.get('bodyTextPreprocessed', '')
                doc = '\n\n'.join([subject, body]).strip()
                if not doc:
                    args.doc_ref[file_path] = {'skip': True}
                    continue
                docs.append(doc)
            kw_sets = kw_model(docs)
            for file_path, keywords in zip(batch, kw_sets):
                args.doc_ref.at[i, 'keywords'] = keywords
                args.doc_ref.loc[i, 'got_keywords'] = True
            args.doc_ref.to_pickle(args.doc_ref_path)


def _get_entities(args):

    if not args.filter_using_keywords:
        with open(args.doc_embeds_path, 'rb') as f:
            doc_embeds = pickle.load(f)

    ner_model = NerCompanyTagger()

    filtered_ref = args.doc_ref[args.doc_ref['embed_index'] >= 0]

    updated = False
    for row in tqdm(list(filtered_ref.itertuples()), 'Getting entities'):
        if not row.filters.get(args.filter_label):
            item = load_json(row.path)
            doc = item.get('subject', '') + '\n\n' + \
                item.get('bodyTextPreprocessed', '')

            if args.filter_using_keywords:
                score = semantic_search_kw(
                    row.keywords, args.filter_model, args.ref_embeds, metric=args.filter_metric)
            else:
                score = semantic_search_doc(
                    doc_embeds[row.embed_index],  args.ref_embeds, metric=args.filter_metric)
            args.doc_ref.loc[row.Index, 'filters'][args.filter_label] = score
            updated = True

        else:
            score = args.doc_ref.loc[row.Index, 'filters'][args.filter_label]
        if score < args.filter_threshold:
            continue
        if not row.got_orgs:
            item = load_json(row.path) or {}
            doc = item.get('subject', '') + '\n\n' + \
                item.get('bodyTextPreprocessed', '')
            orgs = ner_model(doc)['ORG']
            if orgs:
                args.doc_ref.at[row.Index, 'orgs'] = orgs
            args.doc_ref.loc[row.Index, 'got_orgs'] = True
            updated = True
        if row.embed_index and not row.embed_index % 500 and updated:
            args.doc_ref.to_pickle(args.doc_ref_path)

    args.doc_ref.to_pickle(args.doc_ref_path)


def load_ranked_orgs_by_min_proportion(file_path, proportion=.05):

    df = pd.read_csv(file_path, converters={'orgs': literal_eval})
    orgs_list = df.loc[df['proportion'] > proportion, 'orgs'].tolist()
    org_reps = df.loc[df['proportion'] > proportion, 'org_rep'].tolist()
    orgs_set = set()
    for orgs in orgs_list:
        orgs_set.update(orgs)
    return {'org_reps': org_reps, 'exclude': orgs_set}


def get_most_freq_orgs(args):

    tagger = NerCompanyTagger()
    ref_tuples = list(
        args.doc_ref[args.doc_ref['embed_index'] >= 0].itertuples())
    random.shuffle(ref_tuples)
    ref_tuples = ref_tuples[:args.n_random_docs_for_freq_orgs]

    orgs_dict = Counter()

    updated = False
    for i, row in tqdm(list(enumerate(ref_tuples)), f'Getting entities from {args.n_random_docs_for_freq_orgs} random docs'):
        item = load_json(row.path) or {}
        doc = item.get('subject', '') + '\n\n' + \
            item.get('bodyTextPreprocessed', '')
        if row.orgs:
            orgs = row.orgs
        else:
            orgs = tagger(doc)['ORG']
            args.doc_ref.at[row.Index, 'orgs'] = orgs
            updated = True
        if not i % 500 and updated:
            args.doc_ref.to_pickle(args.doc_ref_path)
            updated = False
        for orgs_set in orgs:
            orgs_dict[orgs_set] += 1
    args.doc_ref.to_pickle(args.doc_ref_path)

    merged_orgs = {}
    for org, count in orgs_dict.most_common():
        if org in merged_orgs:
            continue
        if len(org) < 5:
            merged_orgs[org] = {'orgs': {org}, 'count': count}
        else:
            for org_rep, org_dict in merged_orgs.items():
                if len(org_rep) < 5:
                    continue
                if SequenceMatcher(None, org, org_rep).ratio() >= .8:
                    org_dict['orgs'].add(org)
                    org_dict['count'] += count
                    break
            else:
                merged_orgs[org] = {'orgs': {org}, 'count': count}

    d = {'org_rep': [], 'orgs': [], 'count': [], 'proportion': []}
    for org_rep, v in merged_orgs.items():
        d['org_rep'].append(org_rep)
        d['orgs'].append(list(v['orgs']))
        d['count'].append(v['count'])
        d['proportion'].append(v['count']/args.n_random_docs_for_freq_orgs)
    df = pd.DataFrame(d)
    df.sort_values('proportion')
    df.to_csv(args.freq_orgs_path)


def get_relevant_entities(args):

    doc_ref = pd.read_pickle(args.doc_ref_path)
    exclude_dict = load_ranked_orgs_by_min_proportion(
        args.freq_orgs_path, .05)
    excluded_orgs = exclude_dict['exclude'] or []
    excluded_org_reps = exclude_dict['org_reps']

    orgs_counter = Counter()
    per_counter = Counter()
    orgs_kws_counter = {}
    orgs_pers_counter = {}

    for row in doc_ref.itertuples():
        if row.filters.get(args.filter_label, 0) < args.filter_threshold:
            continue
        orgs = row.orgs
        people = []
        if not orgs:
            continue
        item = load_json(row.path)
        sender_name = get_sender(item)[0]
        recipient_names = [r[0] for r in get_recipients(item) if r[0]]
        if sender_name:
            people.append(sender_name)
        people.extend(recipient_names)
        if 'keywords' in args.doc_ref:
            keywords = row.keywords
        else:
            keywords = None
        for org in orgs:
            if org in excluded_orgs:
                continue
            else:
                cont = False
                for org_rep in excluded_org_reps:
                    if SequenceMatcher(None, org, org_rep).ratio() >= .75:
                        cont = True
                        break
                if cont:
                    continue
            orgs_counter[org] += 1
            orgs_kws_counter.setdefault(org, Counter())
            if keywords:
                orgs_kws_counter[org].update(keywords)
            orgs_pers_counter.setdefault(org, Counter())
            orgs_pers_counter[org].update(people)

    d = {x: [] for x in ['org', 'count', 'people', 'keywords']}
    for org, count in orgs_counter.most_common():
        if orgs_counter[org] < per_counter[org]:
            continue
        d['org'].append(org)
        d['count'].append(count)
        top_people = [x[0] for x in orgs_pers_counter[org].most_common(10)]
        d['people'].append('; '.join(top_people))
        top_keywords = [x[0]
                        for x in orgs_kws_counter[org].most_common(10)]
        d['keywords'].append('; '.join(top_keywords))
    if not any(d['keywords']):
        d.pop('keywords')
    df = pd.DataFrame(d)

    output_f = f'{args.filter_label} threshold_{args.filter_threshold}.csv'
    df.to_csv(args.output / output_f)


def get_entities(doc_ref: pd.DataFrame,
                 doc_ref_path: Path,
                 output: Path,
                 util_folder: Path,
                 filter_terms: list = ['invoice', 'payment', 'vendor'],
                 filter_model_name: str = 'msmarco-distilbert-base-v4',
                 filter_metric: Callable = util.cos_sim,
                 filter_threshold: float = .3,
                 filter_using_keywords: bool = False,
                 doc_embeds_path: Path | None = None,
                 kw_kwargs: dict = {'top_n': 10},
                 kw_batch_size: int | None = None,
                 n_random_docs_for_freq_orgs: int = 500
                 ):

    # Use 'all-mpnet-base-v2' if filter_using_keywords'
    filter_model = SentenceTransformer(filter_model_name)
    ref_embeds = filter_model.encode(filter_terms)

    filter_label = (
        filter_model_name,
        f'filter_terms-{";".join(filter_terms)}',
        str(filter_metric).split()[1],
        f'by_keyword-{filter_using_keywords}'
    )
    filter_label = ' '.join(filter_label)

    freq_orgs_path = util_folder / 'most_freq_orgs.csv'

    args = locals()
    Args = namedtuple('Args', args)
    args = Args(**args)

    if not freq_orgs_path.is_file():
        get_most_freq_orgs(args)

    if filter_using_keywords:
        get_keywords(args)
    _get_entities(args)
    get_relevant_entities(args)
