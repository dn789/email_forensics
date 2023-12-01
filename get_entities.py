import argparse
from collections import Counter
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


def similarity_wordnet(keywords, ref_synsets, threshold=.8, metric=wn.wup_similarity, file_path=None):
    for kw in keywords:
        for word in kw.split():
            synsets = wn.synsets(word)
            if not synsets:
                continue
            synset = synsets[0]
            for ref_synset in ref_synsets:
                similarity = metric(synset, ref_synset)
                if similarity >= threshold:
                    return True


def similarity_sbert_kw(keywords, model, ref_embeds, threshold=.6, metric=util.dot_score, file_path=None):
    for kw in keywords:
        words = kw.split()
        word_embeds = model.encode(words)
        for ref_embed in ref_embeds:
            scores = metric(ref_embed, word_embeds)[0].cpu().tolist()
            if max(scores) >= threshold:
                return True


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
        if not filter_func(data['keywords'], **filter_func_args):
            data['filter'][filter_label] = False
            updated = True
            continue
        updated = True
        data['filter'][filter_label] = True

        if not data.get('entities'):
            data['entities'] = ner_model(doc, target_tags=target_tags)
            updated = True
        if i and not i % 500 and updated:
            dump_json(ref_dict, ref_path, default=list)

    dump_json(ref_dict, ref_path, default=list)


def load_ranked_orgs_by_min_proportion(file_path, proportion=.05):
    df = pd.read_csv(file_path)
    return df.loc[df['proportion'] > proportion, 'org'].tolist()


def get_ranked_orgs_random(ref_dict, output):

    paths = list(ref_dict.keys())
    random.shuffle(paths)
    tagger = NerTagger()
    orgs_dict = Counter()
    paths = paths[:1000]

    for file_path in tqdm(paths, f'Getting entities from 1000 random docs...'):

        item = load_json(file_path) or {}
        doc = item.get('bodyText', '').strip()
        if not doc:
            continue
        if 'entities' in ref_dict[file_path]:
            orgs = ref_dict[file_path]['entities']['ORG']
        else:
            orgs = tagger(doc)['ORG']
        for e in orgs:
            orgs_dict[e] += 1
    d = {'org': [], 'count': [], 'proportion': []}
    for e, count in orgs_dict.most_common():
        d['org'].append(e)
        d['count'].append(count)
        d['proportion'].append(count/1000)
    df = pd.DataFrame(d)
    df.to_csv(output)


def get_orgs_and_related_info(ref_path, filter_label, output, output_label=None):
    ref_dict = load_json(ref_path)
    ranked_entities_path = os.path.join(output, 'ranked_orgs_random.csv')
    if not os.path.isfile(ranked_entities_path):
        get_ranked_orgs_random(ref_dict, ranked_entities_path)
    exclude = load_ranked_orgs_by_min_proportion(
        ranked_entities_path, .05)
    exclude = exclude or []

    orgs_counter = Counter()
    orgs_kws_counter = {}
    orgs_pers_counter = {}

    for file_path, data in ref_dict.items():
        if not data.get('filter', {}).get(filter_label):
            continue

        entities = data['entities']
        orgs, people = entities.get('ORG', []), list(
            entities.get('PER', []).keys())

        if not orgs:
            continue

        item = load_json(file_path)
        sender_name = get_sender(item)[0]
        recipient_names = [r[0] for r in get_recipients(item)]
        people.append(sender_name)
        people.extend(recipient_names)

        keywords = data['keywords']
        for org in orgs:
            if org in exclude:
                continue
            orgs_counter[org] += 1
            orgs_kws_counter.setdefault(org, Counter())
            orgs_kws_counter[org].update(keywords)
            orgs_pers_counter.setdefault(org, Counter())
            orgs_pers_counter[org].update(people)

    d = {x: [] for x in ['org', 'count', 'people', 'keywords']}
    for org, count in orgs_counter.most_common():
        d['org'].append(org)
        d['count'].append(count)
        top_people = [x[0] for x in orgs_pers_counter[org].most_common(10)]
        d['people'].append('; '.join(top_people))
        top_keywords = [x[0] for x in orgs_kws_counter[org].most_common(10)]
        d['keywords'].append('; '.join(top_keywords))
    df = pd.DataFrame(d)

    if not output_label:
        output_label = filter_label
    os.makedirs(os.path.join(output, 'output'), exist_ok=True)
    df.to_csv(os.path.join(output, 'output', f'orgs {output_label}.csv'))


def main(docs_folder,
         output,
         output_label=None,
         filter_label='wordnet_invoice_related',
         filter_terms=['invoice.n.01'],
         filter_func=similarity_wordnet,
         filter_func_args={},
         kw_kwargs={'top_n': 10},
         kw_batch_size=None,
         ):

    os.makedirs(output, exist_ok=True)
    ref_path = os.path.join(output, 'ref.json')
    if not os.path.isfile(ref_path):
        make_ref_json(docs_folder, ref_path)
    get_keywords(ref_path, kw_kwargs=kw_kwargs, batch_size=kw_batch_size)
    get_entities(ref_path, filter_label, filter_terms,
                 filter_func, filter_func_args=filter_func_args)
    get_orgs_and_related_info(ref_path, filter_label,
                              output, output_label=output_label)
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
            kw_kwargs=args.kw_kwargs,
            kw_batch_size=args.kw_batch_size,
        )
