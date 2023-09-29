""" 
See bottom of script.
Specify DOCS_FOLDER and then:
(1) get_keywords
(2) get_entities (and relevance_info)
(3) get_person_org_pairs --> csv of person/org pairs, counts, associated docs and keywords, 
etc.
(4) get_all_sender_recipient_pairs --> csv of sender/recipient pairs and counts. 
"""
import argparse
from collections import Counter
import json
import os
import re

from flair.data import Sentence
from flair.models import SequenceTagger
from keybert import KeyBERT
from keyphrase_vectorizers import KeyphraseCountVectorizer
from nltk import sent_tokenize
from nltk.corpus import wordnet as wn
import pandas as pd
import plotly.express as px
import random
# from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm


def make_email_dict_from_string(email):
    """Makes dict of headers, body from text string (eg. from Enron corpus)."""
    email_dict = {}
    header_lines, body = email.split('\n\n', 1)
    header_lines = header_lines.split('\n')
    headers = []
    for line in header_lines:
        if line.startswith('\t') or line.startswith(' '):
            headers[-1] = headers[-1] + f' {line.strip()}'
        else:
            headers.append(line)
    for header in headers:
        k, v = header.split(':', 1)
        email_dict[k] = v.strip()
    email_dict['body'] = body
    return email_dict


class KwModel():
    """Uses KeyBERT to get keywords."""

    def __init__(self, kwargs={}):
        self.kwargs = kwargs
        self.kw_model = KeyBERT()

    def __call__(self, docs):
        vectorizer = self.kwargs.get('vectorizer')
        if vectorizer == 'keyphrase':
            self.kwargs['vectorizer'] = KeyphraseCountVectorizer()
        elif vectorizer == 'count':
            self.kwargs['vectorizer'] = CountVectorizer(
                ngram_range=self.kwargs.get(
                    'keyphrase_ngram_range', (1, 1)),
                stop_words=self.kwargs['stop_words']
            )
        results = []
        results = self.kw_model.extract_keywords(docs, **self.kwargs)
        if type(docs) == str:
            results = [x[0] for x in results]
        else:
            for i, kw_set in enumerate(results):
                results[i] = [x[0] for x in kw_set]
        return results


class NerTagger():
    """Uses flair ontonotes NER to get named entities."""

    def __init__(self):
        self.tagger = SequenceTagger.load("flair/ner-english-fast")

    def __call__(self, text, target_tags=('ORG', 'PER')):
        tag_dict = {tag: {} for tag in target_tags}
        sents = sent_tokenize(text)
        for sent in sents:
            sent = Sentence(sent)
            self.tagger.predict(sent)
            for entity in sent.get_spans('ner'):
                if entity.tag in target_tags:
                    tag_dict[entity.tag][entity.text] = None
        text = text.replace('\n', ' ')
        for tag, entity_dict in tag_dict.items():
            for entity in entity_dict:
                entity_pattern = re.escape(entity)
                entity_dict[entity] = {
                    'count': len(re.findall(
                        rf'\b{entity_pattern}\b', text)),
                    'tf-idf': None
                }
        return tag_dict


def check_keywords_for_relevance(filename, keywords, ref_synsets, threshold=.8, metric=wn.wup_similarity):
    """Checks keyword synsets for similarity with synsets ref_synset_file."""
    ref_synsets = [wn.synset(name) for name in ref_synsets]
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


def check_keywords_for_entities(keywords, entities):
    """Checks if any keyword is a word-by-word subset of any entity or vice-versa."""
    matches = set()
    for keyword in keywords:
        keyword = ' ' + keyword.lower() + ' '
        for entity in entities:
            entity = ' ' + entity.lower() + ' '
            if keyword in entity or entity in keyword:
                matches.add(entity.strip())
    return matches


def get_keywords(filename, docs_folder, kw_kwargs={}, batch_size=None):
    """
    Get keywords for docs in folder using KeyBERT.

    Parameters
    ----------
    filename : str
    docs_folder : str
    kw_kwargs : dict
        KeyBERT model args. 
    batch_size : int
        Pass batches of n docs to KeyBERT at once (faster).
    """
    if os.path.isfile(filename):
        kw_dict = json.load(open(filename))
    else:
        kw_dict = {f: None for f in os.listdir(docs_folder)}

    kw_model = KwModel(kw_kwargs)
    added = False
    if not batch_size:
        for i, (f, v) in tqdm(list(enumerate(kw_dict.items())), 'Getting keywords...'):
            if i and not i % 500 and added:
                json.dump(kw_dict, open(filename, 'w'))
            if v:
                continue
            with open(os.path.join(docs_folder, f), encoding='utf-8') as f_:
                try:
                    if f.endswith('json'):
                        doc = json.load(f_)['body'].strip()
                    else:
                        doc = f_.read().split('\n\n')[-1].strip()
                except UnicodeDecodeError:
                    kw_dict[f] = {'skip': True}
                    continue
            try:
                kw_sets = kw_model(doc)
                kw_dict[f] = kw_sets
                added = True
            except Exception:
                kw_dict[f] = []
        json.dump(kw_dict, open(filename, 'w'))
    else:
        filenames = list(kw_dict.keys())
        filenames = [f for f in filenames if not kw_dict[f]]
        f_batches = [filenames[i:i + batch_size]
                     for i in range(0, len(filenames), batch_size)]
        for f_batch in tqdm(f_batches, f'Getting keywords from batches of {batch_size} docs...'):
            docs = [open(os.path.join(docs_folder, f)).read() for f in f_batch]
            kw_sets = kw_model(docs)
            for f, keywords in zip(f_batch, kw_sets):
                kw_dict[f] = keywords
            json.dump(kw_dict, open(filename, 'w'))


def get_entities(filename,
                 docs_folder,
                 kw_json,
                 relevance_json,
                 relevance_label,
                 relevance_func=check_keywords_for_relevance,
                 relevance_func_args={},
                 target_tags=('ORG', 'PER')
                 ):
    """
    Makes or adds to a JSON of docs and their named entities, and a JSON of docs and
    their relevance for each relevance label. Only gets entities for docs that pass
    current relevance_func. 

    Parameters
    ----------
    filename : str
    docs_folder : str
    kw_json : str
        keywords JSON
    relevance_json : str
        JSON with relevance info for each relevance_label for each doc. Will be created
        if it doesn't exist. 
    relevance_label : str
        label for relevance function
    relevance_func : func
        Function to determine relevance of a doc. Should return True if doc is relevant.
    relevance_func_args : dict
        Args for relevance_func aside from keywords and filename.
    target_tags : tuple
        target tags for entities, by default ('ORG', 'PER') for comparing organization-
        person pairs.
    """
    if os.path.isfile(filename):
        entities_dict = json.load(
            open(filename, encoding='utf-8'))
    else:
        entities_dict = {f: {} for f in os.listdir(docs_folder)}

    if os.path.isfile(relevance_json):
        relevance_dict = json.load(
            open(relevance_json, encoding='utf-8'))
    else:
        relevance_dict = {filename: {} for filename in os.listdir(docs_folder)}

    kw_dict = json.load(
        open(kw_json, encoding='utf-8'))

    ner_model = NerTagger()
    added_entities, added_relevance = False, False
    for i, (f, v) in tqdm(list(enumerate(entities_dict.items())), 'Getting entities...'):
        if relevance_label in relevance_dict[f] or v.get('skip'):
            continue
        with open(os.path.join(docs_folder, f), encoding='utf-8') as f_:
            try:
                if f.endswith('json'):
                    doc = json.load(f_)['body'].strip()
                else:
                    doc = f_.read().split('\n\n')[-1].strip()
            except UnicodeDecodeError:
                v['skip'] = True
                continue
        if not doc:
            v['skip'] = True
            continue
        if not relevance_func(f, kw_dict[f], **relevance_func_args):
            added_relevance = True
            relevance_dict[f][relevance_label] = False
            continue
        added_relevance = True
        relevance_dict[f][relevance_label] = True
        if not v:
            entities_dict[f] = ner_model(doc, target_tags=target_tags)
            added_entities = True
        if i and not i % 500 and (added_entities or added_relevance):
            if added_entities:
                json.dump(entities_dict, open(
                    filename, 'w', encoding='utf-8'), default=list)
            if added_relevance:
                json.dump(relevance_dict, open(
                    relevance_json, 'w', encoding='utf-8'), default=list)
    json.dump(entities_dict, open(
        filename, 'w', encoding='utf-8'), default=list)
    json.dump(relevance_dict, open(
        relevance_json, 'w', encoding='utf-8'), default=list)


def get_ranked_entities_from_file(proportion, entities_file):
    entities = []
    for line in open(entities_file).read().strip().split('\n'):
        try:
            entity, prop = line.split('\t')
            if float(prop) >= proportion:
                entities.append(entity)
            else:
                break
        except ValueError:
            pass
    return entities


def graph_entities(entities_json, top_n_entities=None, exclude=None):
    exclude = exclude or []
    results_dict = json.load(open(entities_json, encoding='utf-8'))
    all_entities = []
    for k, v in results_dict.items():
        entities = list(
            [x for x in v.get('ORG', {}).keys() if x not in exclude])
        all_entities.extend(entities)
    counts = Counter(all_entities)
    counts = counts.most_common(top_n_entities)
    df = pd.DataFrame(list(counts))
    df = df.rename(columns={0: 'word', 1: 'count'})
    fig = px.bar(df, x='word', y='count')
    fig.show()


def get_ranked_entities(folder, output):
    """
    Creates file of entities and their % of occurence in 1000 docs in folder.
    """
    fs = os.listdir(folder)
    random.shuffle(fs)
    tagger = NerTagger()
    entities_d = Counter()
    fs = fs[:5000]
    doc_count = len(fs)
    for f in tqdm(fs, f'Getting entities from {doc_count} random emails...'):
        f_ = open(os.path.join(folder, f))
        try:
            if f.endswith('json'):
                doc = json.load(f_)['body'].strip()
            else:
                doc = f_.read().split('\n\n')[-1].strip()
        except UnicodeDecodeError:
            pass
        entities = tagger(doc)
        for tag in entities:
            for e in entities[tag]:
                entities_d[e] += 1
    to_write = []

    for e, count in entities_d.most_common():
        to_write.append(f'{e}\t{count/doc_count}')
    with open(output, 'w', encoding='utf-8') as f:
        f.write('\n'.join(to_write))


def get_person_org_pairs(relevance_label, docs_folder, relevance_json, entity_json, kw_json, ranked_entities_path=None):
    """
    Makes dataframe of:
    - person-org pairs
    - list of docs they're found in
    - keywords in those docs. 
    ...for each doc in relevance_json with relevance_label: True

    Parameters
    ----------
    kw_json : str
    relevance_label : str
    relevance_json : str
    entity_json : str
    kw_json : str
    ranked_entities_path : str
        Path to ranked entities % occurence in docs (all that occur in >= 5%
        of docs will bbe excluded)
    """
    if os.path.isfile(ranked_entities_path):
        exclude = get_ranked_entities_from_file(
            .05, ranked_entities_path)
    else:
        get_ranked_entities(docs_folder, ranked_entities_path)
        exclude = get_ranked_entities_from_file(
            .05, ranked_entities_path)
    exclude = exclude or []
    relevance_dict = json.load(
        open(relevance_json, encoding='utf-8'))
    entity_dict = json.load(
        open(entity_json, encoding='utf-8'))
    kw_dict = json.load(
        open(kw_json, encoding='utf-8'))
    counts = Counter()
    pairs_ref = {}
    for f, v in relevance_dict.items():
        if not v.get(relevance_label):
            continue
        entities = entity_dict[f]
        if not entities or not entities.get('ORG') or not entities.get('PER'):
            continue
        for org in entities['ORG']:
            if org in exclude:
                continue
            for per in entities['PER']:
                counts[(org, per)] += 1
                pairs_ref.setdefault(
                    (org, per), {'files': set(), 'keywords': Counter()})
                pairs_ref[(org, per)]['files'].add(f)
                pairs_ref[(org, per)]['keywords'].update(kw_dict[f])

    d = {x: [] for x in ['org', 'person', 'count', 'keywords', 'files']}
    for pair, count in counts.most_common():
        d['org'].append(pair[0])
        d['person'].append(pair[1])
        d['count'].append(count)
        top_keywords = [x[0]
                        for x in pairs_ref[pair]['keywords'].most_common(10)]
        d['keywords'].append(', '.join(top_keywords))
        d['files'].append(pairs_ref[pair]['files'])
    df = pd.DataFrame(d)
    return df


def get_email_sender_and_recipients(email):
    """Gets sender and recpients from email dict or string."""
    if type(email) == str:
        email = make_email_dict_from_string(email)
    if not email.get('From') or not email.get('To'):
        return
    sender_string = email['From']
    recepient_string = email['To']

    sender_and_recipients = {}
    for string, label, in zip((sender_string, recepient_string), ('sender', 'recipients')):
        names_emails = []
        for name_address in string.split(','):
            if len(name_address_split := name_address.split('<')) == 2:
                name, address = name_address_split
                name = name.strip('" ')
                address = address.strip('>')
            else:
                name = None
                address = name_address_split[0]
            names_emails.append({'name': name, 'address': address})
        if label == 'sender':
            names_emails = names_emails[0]
        sender_and_recipients[label] = names_emails
    return sender_and_recipients


def get_all_sender_recipient_pairs(relevance_json, docss_folder, relevance_label):
    pairs_dict = Counter()
    results_dict = json.load(open(relevance_json))
    for f, v in results_dict.items():
        if not v.get(relevance_label):
            continue
        try:
            path = os.path.join(docss_folder, f)
            if f.endswith('json'):
                email = json.load(open(path))
            else:
                email = open(path).read()
        except UnicodeDecodeError:
            continue
        sender_and_recipents = get_email_sender_and_recipients(email)
        if not sender_and_recipents:
            continue
        sender = sender_and_recipents['sender']
        for recipient in sender_and_recipents['recipients']:
            pairs_dict[(sender['address'], recipient['address'])] += 1
            # k = f'{sender["address"]}\t{recipient["address"]}'
            # pairs_dict[k] += 1  for pair, count in counts.most_common():
    d = {x: [] for x in ['sender', 'recipient', 'count']}
    for k, v in pairs_dict.most_common():
        d['sender'].append(k[0])
        d['recipient'].append(k[1])
        d['count'].append(v)
    df = pd.DataFrame(d)
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="""Makes (1) CSV of top co-occuring person and organization entities
    (2) top sender/recipient pairs in emails """)

    parser.add_argument('docs_folder', help='Folder containing emails')
    parser.add_argument(
        '-kw_path', help='Path for keywords JSON', default='keywords.json')
    parser.add_argument(
        '-kw_batch_size', help='Keywords batch size', default=None)
    parser.add_argument(
        '-entities_path', help='Path for entities json', default='entities.json')
    parser.add_argument(
        '-relevance_path', help='Path for relevance json', default='relevance.json')
    parser.add_argument(
        '-relevance_label', help='Relevance label', default='invoice synset only')
    parser.add_argument(
        '-ref_synsets', help='Reference synsets', default=['invoice.n.01'])
    parser.add_argument(
        '-entity_pairs_path', help='Path for entity-pairs csv', default='person-org pairs.csv')
    parser.add_argument(
        '-to_from_pairs_path', help='Path for sender-recipient csv', default='to_from_pairs.csv')
    parser.add_argument(
        '-ranked_entities_path', help='Path to ranked entities. Will create if not found.', default='ranked_entities.txt')
    parser.add_argument(
        '-kw_args', help='Args for KeyBERT', default={'top_n': 10})
    parser.add_argument(
        '-get_kws', help='Whether to get keywords', default=True)
    parser.add_argument(
        '-get_entities', help='Whether to get entities', default=True)
    parser.add_argument(
        '-get_pairs', help='Whether to get person/org and to/from pairs', default=True)
    args = parser.parse_args()
    if args.get_kws:
        get_keywords(args.kw_path,
                     args.docs_folder,
                     args.kw_args,
                     batch_size=args.kw_batch_size)

    if args.get_entities:
        get_entities(
            args.entities_path,
            args.docs_folder,
            args.kw_path,
            args.relevance_path,
            args.relevance_label,
            relevance_func=check_keywords_for_relevance,
            relevance_func_args={
                'ref_synsets': args.ref_synsets
            })

    if args.get_pairs:
        person_org_pairs = get_person_org_pairs(
            args.relevance_label,
            args.docs_folder,
            args.relevance_path,
            args.entities_path,
            args.kw_path,
            args.ranked_entities_path
        )
        person_org_pairs.head(5000).to_csv(args.entity_pairs_path)
        to_from_pairs = get_all_sender_recipient_pairs(
            args.relevance_path,
            args.docs_folder,
            args.relevance_label)
        to_from_pairs.head(5000).to_csv(args.to_from_pairs_path)
