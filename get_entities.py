import argparse
from ast import literal_eval
from collections import Counter
from difflib import SequenceMatcher
import os
import pickle
import random

from nltk.corpus import wordnet as wn
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from thefuzz.process import fuzz
from tqdm import tqdm

from models import KwModel, NerCompanyTagger
from utils import dump_json, load_json, get_sender, get_recipients


def make_ref_json(paths, output):
    d = {}
    for i, path in enumerate(paths):
        d[path] = {'index': i}
    dump_json(d, output)


def similarity_wordnet(keywords, ref_synsets, metric=wn.wup_similarity, **kwargs):
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


def get_keywords(ref_path, kw_kwargs={}, batch_size=None):
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
            doc = item.get('subject', '') + '\n\n' + \
                item.get('bodyTextFiltered', '').strip()
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
                doc = item.get('subject', '') + '\n\n' + \
                    item.get('bodyTextFiltered', '').strip()
                if not doc:
                    ref_dict[file_path] = {'skip': True}
                    continue
                docs.append(doc)

            kw_sets = kw_model(docs)
            for file_path, keywords in zip(batch, kw_sets):
                data['keywords'] = keywords

            dump_json(ref_dict, ref_path, default=list)


def get_entities(ref_path,
                 util_folder,
                 filter_label,
                 filter_func,
                 filter_func_args,
                 ):
    ref_dict = load_json(ref_path)

    ner_model = NerCompanyTagger()

    if filter_func == semantic_search_doc:
        doc_embeds_path = os.path.join(
            util_folder,  f'{filter_func_args["model_name"]}_doc_embeds.pkl')

        if os.path.isfile(doc_embeds_path):
            with open(doc_embeds_path, 'rb') as f:
                doc_embeds = pickle.load(f)
        else:
            file_paths = ref_dict.keys()
            docs = []
            for f in file_paths:
                item = load_json(f)
                docs.append(item.get('subject', '') + '\n\n' +
                            item.get('bodyTextFiltered', '').strip())

            print('Encoding docs. This might take a while...')
            doc_embeds = filter_func_args['model'].encode(docs)
            with open(doc_embeds_path, "wb") as fOut:
                pickle.dump(doc_embeds, fOut, protocol=pickle.HIGHEST_PROTOCOL)

    updated = False

    for i, (file_path, data) in tqdm(list(enumerate(ref_dict.items())), 'Getting entities...'):
        if data.get('skip'):
            continue

        if filter_label not in data.get('filter', {}):
            data.setdefault('filter', {})
            item = load_json(file_path) or {}
            doc = item.get('subject', '') + '\n\n' + \
                item.get('bodyTextFiltered', '')
            if not doc:
                ref_dict[file_path] = {'skip': True}
                continue

            if filter_func == semantic_search_doc:
                score = filter_func(
                    doc_embeds[data['index']], **filter_func_args)
            else:
                score = filter_func(data['keywords'], **filter_func_args)
            data['filter'][filter_label] = score
            updated = True
        else:
            score = data['filter'][filter_label]
        if score < filter_func_args['threshold']:
            continue
        if not data.get('entities'):
            item = load_json(file_path) or {}
            doc = item.get('subject', '') + '\n\n' + \
                item.get('bodyTextFiltered', '')
            if not doc:
                ref_dict[file_path] = {'skip': True}
                continue
            data['entities'] = ner_model(doc)
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
    tagger = NerCompanyTagger()
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
        doc = item.get('subject', '') + '\n\n' + \
            item.get('bodyTextFiltered', '')
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
        d['proportion'].append(v['count']/n_docs)
    df = pd.DataFrame(d)
    df.sort_values('proportion')
    df.to_csv(output)


def get_orgs_and_related_info(ref_path, filter_label, filter_threshold, output, util_folder):
    ref_dict = load_json(ref_path)
    ranked_entities_path = os.path.join(
        util_folder, 'ranked_orgs_random.csv')
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
        entities = data.get('entities')
        if not entities:
            print(file_path)
            continue
        orgs, people = entities.get('ORG', {}), list(
            entities.get('PER', {}).keys())

        per_counter.update(people)

        if not orgs:
            continue

        item = load_json(file_path)
        sender_name = get_sender(item)[0]
        recipient_names = [r[0] for r in get_recipients(item) if r[0]]
        if sender_name:
            people.append(sender_name)
        people.extend(recipient_names)

        keywords = data.get('keywords', [])
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
        top_keywords = [x[0]
                        for x in orgs_kws_counter[org].most_common(10)]
        d['keywords'].append('; '.join(top_keywords))
    if not any(d['keywords']):
        d.pop('keywords')
    df = pd.DataFrame(d)

    output_f = f'{filter_label} threshold_{filter_threshold}.csv'
    df.to_csv(os.path.join(output,  output_f))


def main(paths,
         output,
         util_folder,
         filter_terms=['invoice', 'payment', 'vendor'],
         filter_func=semantic_search_doc,
         filter_func_args={},
         kw_kwargs={'top_n': 10},
         kw_batch_size=None,
         ):
    """Gets named organization entities from relevant emails.
    Saves output to a csv file with the name of the config. 

    Parameters
    ----------
    paths : list
        Doc paths.
    output : str
        Path.
    util_folder : str
        Path.
    filter_terms : list, optional
        Terms used to find relevant emails, 
        default ['invoice', 'payment', 'vendor']
    filter_func : func, optional
        Function used to compare filter_terms and email text/keywords:
            - semantic_search_doc : Asymmetric semantic search comparing
                filter_terms with email text.
            - semantic_search_kw : Symmetric semantic search comparing
                filter_terms with email keywords.
            - similarity_wordnet : Compares filter_terms with WordNet synsets
                of email keywords. filter_terms must be a list of WordNet
                synsets (see https://www.nltk.org/howto/wordnet.html)
    filter_func_args : dict, optional
        Each filter_func has its own default args (see beginning of function). 
        Only use this when overriding those :
            - model_name (semantic_search_*) : Use asymmetric semantic
                search models for semantic_search_doc and symmetric ones
                for semantic_search_kw. Make sure to use the appropriate 
                metric for the specified model (see sentence_transformer 
                documentation for more info).
            - metric : Similarity metric.
                wn.wup_similarity (similarity_wordnet) : 0 to 1
                util.cos_sim (semantic_search_*) : -1 to 1
                util.dot_score (semantic_search_*) : Test model on huggingface 
                to determine a reasonable threshold.
            - threshold : Threshold to determine relevance.

    kw_kwargs : dict, optional
        Args for keywords extraction, default {'top_n': 10}
    kw_batch_size : int, optional
        Batch size for keyword extraction, by default None
    """

    if filter_func == similarity_wordnet:
        filter_func_args['model_name'] = 'wordnet_synsets'
        filter_func_args['ref_synsets'] = [
            wn.synset(name) for name in filter_terms]
        filter_func_args['metric'] = filter_func_args.get(
            'metric', wn.wup_similarity,)
        filter_func_args['threshold'] = filter_func_args.get(
            'threshold', .9)

    if filter_func == semantic_search_kw:
        filter_func_args['model_name'] = filter_func_args.get(
            'model_name', 'all-mpnet-base-v2')
        filter_func_args['model'] = SentenceTransformer(
            filter_func_args['model_name'])
        filter_func_args['metric'] = filter_func_args.get(
            'metric', util.cos_sim)
        filter_func_args['threshold'] = filter_func_args.get(
            'threshold', .9)
        filter_func_args['ref_embeds'] = filter_func_args['model'].encode(
            filter_terms)

    if filter_func == semantic_search_doc:
        filter_func_args['model_name'] = filter_func_args.get(
            'model_name', 'msmarco-distilbert-base-v4')
        filter_func_args['model'] = SentenceTransformer(
            filter_func_args['model_name'])
        filter_func_args['metric'] = filter_func_args.get(
            'metric', util.cos_sim)
        filter_func_args['threshold'] = filter_func_args.get(
            'threshold', .3)
        filter_func_args['ref_embeds'] = filter_func_args['model'].encode(
            filter_terms)

    filter_label = f'{str(filter_func).split()[1]} {filter_func_args["model_name"]} filter_terms-{";".join(filter_terms)} {str(filter_func_args["metric"]).split()[1]}'
    ref_path = os.path.join(util_folder, 'doc_ref.json')

    if not os.path.isfile(ref_path):
        make_ref_json(paths, ref_path)
    if filter_func in (semantic_search_kw, similarity_wordnet):
        get_keywords(ref_path,
                     kw_kwargs,
                     kw_batch_size)
    get_entities(ref_path,
                 util_folder,
                 filter_label,
                 filter_func,
                 filter_func_args)
    get_orgs_and_related_info(ref_path,
                              filter_label,
                              filter_func_args['threshold'],
                              output,
                              util_folder)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='See file for documentation')
    parser.add_argument('paths')
    parser.add_argument('output')
    parser.add_argument('util_folder')
    parser.add_argument(
        '-filter_terms', default=['invoice', 'payment', 'vendor'])
    parser.add_argument('-filter_func', default=semantic_search_doc)
    parser.add_argument('-filter_func_args', default={})
    parser.add_argument('-kw_kwargs', default={'top_n': 10})
    parser.add_argument('-kw_batch_size', default=None)
    args = parser.parse_args()

    if args.get_kws:
        main(
            args.paths,
            args.output,
            args.util_folder,
            args.filter_terms,
            args.filter_func,
            args.filter_func_args,
            kw_kwargs=args.kw_kwargs,
            kw_batch_size=args.kw_batch_size,
        )
