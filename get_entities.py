import argparse
from ast import literal_eval
from collections import Counter
from difflib import SequenceMatcher
import os
import random

from nltk.corpus import wordnet as wn
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

from models import KwModel, NerTagger
from utils import dump_json, load_json, get_sender, get_recipients


def make_ref_json(docs_folder, output):
    d = {}
    for root, dirs, files in os.walk(docs_folder):
        for f in files:
            d[os.path.join(root, f)] = {}
    dump_json(d, output)


def similarity_wordnet(keywords, ref_synsets, metric=wn.wup_similarity, file_path=None):
    best_score = 0
    for kw in keywords:
        for word in kw.split():
            synsets = wn.synsets(word)
            if not synsets:
                continue
            synset = synsets[0]
            for ref_synset in ref_synsets:
                similarity = metric(synset, ref_synset)
                if best_score < similarity:
                    best_score = similarity
    return best_score


def similarity_sbert_kw(keywords, model, ref_embeds, metric=util.dot_score, file_path=None):
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


def symmetric_similarity(phrase1, phrase2, model, metric=util.dot_score):
    phrase1emb = model.encode(phrase1)
    phrase2emb = model.encode(phrase2)
    score_temp = metric(phrase1emb, phrase2emb)[0].cpu()
    score = score_temp.tolist()[0]
    return score


def get_keywords(ref_path, kw_kwargs={}, batch_size=None):
    """
    Get keywords for docs in folder using KeyBERT.

    Parameters
    ----------
    ref_path : str
    kw_kwargs : dict
        KeyBERT model args. 
    batch_size : int
        Pass batches of n docs to KeyBERT at once (faster).
    """
    ref_dict = load_json(ref_path)
    kw_model = KwModel(kw_kwargs)

    updated = False

    if not batch_size:
        for i, (file_path, data) in tqdm(list(enumerate(ref_dict.items())), 'Getting keywords...'):
            if i and not i % 500 and updated:
                dump_json(ref_dict, ref_path, default=list)
                updated = False
            if 'keywords' in data or data.get('skip'):
                continue
            item = load_json(file_path) or {}
            doc = item.get('bodyText', '').strip()
            if not doc:
                ref_dict[file_path] = {'skip': True}
                continue
            try:
                kw_sets = kw_model(doc)
                data['keywords'] = kw_sets
                updated = True
            except Exception:
                data['keywords'] = []
        dump_json(ref_dict, ref_path, default=list)

    else:
        paths = list(ref_dict.keys())
        paths = [p for p in paths if not ref_dict[p]['keywords']]
        batches = [paths[i:i + batch_size]
                   for i in range(0, len(paths), batch_size)]
        for batch in tqdm(batches, f'Getting keywords from batches of {batch_size} docs...'):
            docs = []
            for file_path in batch:
                item = load_json(file_path) or {}
                doc = item.get('bodyText', '').strip()
                if not doc:
                    ref_dict[file_path] = {'skip': True}
                    continue
                docs.append(doc)

            kw_sets = kw_model(docs)
            for file_path, keywords in zip(batch, kw_sets):
                data['keywords'] = keywords

            dump_json(ref_dict, ref_path, default=list)


def get_entities(ref_path,
                 filter_label,
                 filter_terms,
                 filter_func=similarity_wordnet,
                 filter_func_args={},
                 filter_threshold=.75,
                 target_tags=('ORG', 'PER')
                 ):

    ref_dict = load_json(ref_path)

    ner_model = NerTagger()

    if filter_func == similarity_sbert_kw:
        sbertmodel = SentenceTransformer('all-mpnet-base-v2')
        filter_func_args['model'] = sbertmodel
        filter_func_args['ref_embeds'] = sbertmodel.encode(
            filter_terms)

    if filter_func == similarity_wordnet:
        filter_func_args['ref_synsets'] = [
            wn.synset(name) for name in filter_terms]

    updated = False

    for i, (file_path, data) in tqdm(list(enumerate(ref_dict.items())), 'Getting entities...'):
        if filter_label in data.get('filter', {}) or data.get('skip'):
            continue

        data.setdefault('filter', {})
        item = load_json(file_path) or {}
        doc = item.get('bodyText', '').strip()
        if not doc:
            ref_dict[file_path] = {'skip': True}
            continue

        filter_func_args['file_path'] = file_path
        score = filter_func(data['keywords'], **filter_func_args)
        data['filter'][filter_label] = score
        updated = True
        if score < filter_threshold:
            continue
        if not data.get('entities'):
            data['entities'] = ner_model(doc, target_tags=target_tags)
            updated = True
        if i and not i % 500 and updated:
            dump_json(ref_dict, ref_path, default=list)

    dump_json(ref_dict, ref_path, default=list)


def load_ranked_orgs_by_min_proportion(file_path, proportion=.05):
    df = pd.read_csv(file_path, converters={'orgs': literal_eval})
    orgs_list = df.loc[df['proportion'] > proportion, 'orgs'].tolist()
    org_reps = df.loc[df['proportion'] > proportion, 'org_rep'].tolist()
    orgs_set = set()
    for orgs in orgs_list:
        orgs_set.update(orgs)
    return {'org_reps': org_reps, 'exclude': orgs_set}


def get_ranked_orgs_random(ref_dict, ref_path, output, n_docs=1000):

    paths = list(ref_dict.keys())
    random.shuffle(paths)
    tagger = NerTagger()
    orgs_dict = Counter()
    paths = paths[:n_docs]

    if not n_docs:
        n_docs = len(paths)

    updated = False
    for i, file_path in tqdm(list(enumerate(paths)), f'Getting entities from {n_docs} random docs...'):
        data = ref_dict[file_path]
        if data.get('skip'):
            continue
        item = load_json(file_path) or {}
        doc = item.get('bodyText', '').strip()
        if not doc:
            data['skip'] = True
            continue
        if 'entities' in ref_dict[file_path]:
            orgs = data['entities']['ORG']
        else:
            entities = tagger(doc)
            data['entities'] = entities
            orgs = entities['ORG']
            updated = True
        if not i % 500 and updated:
            dump_json(ref_dict, ref_path)
            updated = False
        for orgs_set in orgs:
            orgs_dict[orgs_set] += 1
    dump_json(ref_dict, ref_path)

    merged_orgs = {}
    for org, count in orgs_dict.most_common():
        if org in merged_orgs:
            continue
        else:
            for org_rep, org_dict in merged_orgs.items():
                if SequenceMatcher(None, org, org_rep).ratio() >= .75:
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
        d['proportion'].append(v['count']/n_docs)
    df = pd.DataFrame(d)
    df.sort_values('proportion')
    df.to_csv(output)


def get_orgs_and_related_info(ref_path, filter_label, filter_threshold, output):
    ref_dict = load_json(ref_path)
    ranked_entities_path = os.path.join(output, 'ranked_orgs_random.csv')
    if not os.path.isfile(ranked_entities_path):
        get_ranked_orgs_random(ref_dict, ref_path, ranked_entities_path)
    exclude_dict = load_ranked_orgs_by_min_proportion(
        ranked_entities_path, .05)
    excluded_orgs = exclude_dict['exclude'] or []
    excluded_org_reps = exclude_dict['org_reps']

    orgs_counter = Counter()
    per_counter = Counter()
    orgs_kws_counter = {}
    orgs_pers_counter = {}

    for file_path, data in ref_dict.items():
        if data.get('filter', {}).get(filter_label, 0) < filter_threshold:
            continue

        entities = data['entities']
        orgs, people = entities.get('ORG', []), list(
            entities.get('PER', []).keys())
        per_counter.update(people)

        if not orgs:
            continue

        item = load_json(file_path)
        sender_name = get_sender(item)[0]
        recipient_names = [r[0] for r in get_recipients(item)]
        people.append(sender_name)
        people.extend(recipient_names)

        keywords = data['keywords']
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
        top_keywords = [x[0] for x in orgs_kws_counter[org].most_common(10)]
        d['keywords'].append('; '.join(top_keywords))
    df = pd.DataFrame(d)

    output_f = f'orgs {filter_label} threshold_{filter_threshold}.csv'
    os.makedirs(os.path.join(output, 'output'), exist_ok=True)
    df.to_csv(os.path.join(output, 'output', output_f))


def main(docs_folder,
         output,
         filter_label='wordnet_invoice_related',
         filter_terms=['invoice.n.01'],
         filter_func=similarity_wordnet,
         filter_func_args={},
         filter_threshold=.75,
         kw_kwargs={'top_n': 10},
         kw_batch_size=None,
         ):

    os.makedirs(output, exist_ok=True)
    ref_path = os.path.join(output, 'ref.json')
    if not os.path.isfile(ref_path):
        make_ref_json(docs_folder, ref_path)
    get_keywords(ref_path, kw_kwargs=kw_kwargs, batch_size=kw_batch_size)
    get_entities(ref_path, filter_label, filter_terms,
                 filter_func, filter_func_args=filter_func_args, filter_threshold=filter_threshold)
    get_orgs_and_related_info(ref_path, filter_label, filter_threshold,
                              output)
    print('\nFinished.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='')

    parser.add_argument(
        'docs_folder', help='Folder containing extracted PST items (emails, etc.)')
    parser.add_argument(
        'output', help='Output folder')
    parser.add_argument(
        '-kw_batch_size', help='Keywords batch size', default=None)
    parser.add_argument(
        '-filter_label', help='Filter label', default='wordnet invoice related')
    parser.add_argument(
        '-filter_terms', help='Filter terms', default=['invoice.n.01'])
    parser.add_argument(
        '-filter_func', help='Filter function', default=similarity_wordnet)
    parser.add_argument(
        '-filter_func_args', help='Filter function arguments', default=similarity_wordnet)
    parser.add_argument(
        '-filter_threshold', help='Filter function threshold', default=similarity_wordnet)
    parser.add_argument(
        '-kw_kwargs', help='Args for KeyBERT', default={'top_n': 10})
    parser.add_argument(
        '-get_kws', help='Whether to get keywords', default=True)
    args = parser.parse_args()

    if args.get_kws:
        main(
            args.docs_folder,
            args.output,
            args.filter_label,
            args.filter_terms,
            args.filter_func,
            args.filter_func_args,
            args.filter_threshold,
            kw_kwargs=args.kw_kwargs,
            kw_batch_size=args.kw_batch_size,
        )
