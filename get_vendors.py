""" 
Use KeyBERT to get keywords from an email. If any of the keywords are related to
orders, invoices, etc. run ontonotes NER. If any ORG entities found by the NER
are in the keywords, consider them a vendor.
"""

from collections import Counter
import difflib
import json
import os

from flair.data import Sentence
from flair.models import SequenceTagger
from keybert import KeyBERT
from keyphrase_vectorizers import KeyphraseCountVectorizer
from nltk import sent_tokenize
from nltk import wordnet as wn
import pandas as pd
import plotly.express as px
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm

from utils import get_synsets


class KwModel():
    def __init__(self, kwargs={}):
        self.kwargs = {
            # Min, max word count for keywords
            'keyphrase_ngram_range': (1, 3),
            'use_mmr': True,  # Increases diversity of keywords
            'diversity': .5,  # Set diversity between 0 and 1 if using MMR
            # ("keyphrase", True, False) How to represent document. Keyphrase vectorizer should be more coherent
            'vectorizer': 'keyphrase',
            'stop_words': 'english'
        }
        self.kwargs.update(kwargs)
        self.kw_model = KeyBERT()

    def __call__(self, text):
        """Get keywords."""
        if self.kwargs.get('vectorizer'):
            if self.kwargs['vectorizer'] == 'keyphrase':
                self.kwargs['vectorizer'] = KeyphraseCountVectorizer()
            else:
                self.kwargs['vectorizer'] = CountVectorizer(
                    ngram_range=self.kwargs.get(
                        'keyphrase_ngram_range', (1, 1)),
                    stop_words=self.kwargs['stop_words']
                )
        keywords = self.kw_model.extract_keywords(text, **self.kwargs)
        keywords = [x[0] for x in keywords]
        return keywords


class NerTagger():
    def __init__(self):
        self.tagger = SequenceTagger.load("flair/ner-english-fast")

    def __call__(self, text, entities=('ORG',)):
        """Get entities from text."""
        entities = set()
        sents = sent_tokenize(text)
        for sent in sents:
            sent = Sentence(sent)
            self.tagger.predict(sent)
            for entity in sent.get_spans('ner'):
                if entity.tag in entities:
                    entities.add(entity.text)
        return entities


def get_entities_and_keywords(text_folder, output, n_files=None, entities=('ORG',), kw_kwargs={}):
    """
    Makes a JSON file with entity and keyword information for each file
    in text_folder. n_files = # of files in text_folder to process, default all.
    """
    kw_model = KwModel(kw_kwargs)
    ner_model = NerTagger()

    results_dict = {}

    for i, filename in tqdm(list(enumerate(os.listdir(text_folder)[:n_files]))):
        if filename in results_dict:
            continue
        if i and not i % 100:
            json.dump(results_dict, open(
                output, 'w', encoding='utf-8'), default=list)
        text = open(os.path.join(text_folder, filename),
                    encoding='utf-8').read()
        entities = ner_model(text)
        keywords = kw_model(text)
        results_dict[filename] = {
            'entities': entities,
            'keywords': keywords
        }
    json.dump(results_dict, open(
        output, 'w', encoding='utf-8'), default=list)


def check_keywords_for_relevance(keywords, ref_synsets, threshold=.8, metric=wn.wup_similarity):
    """Checks keywords for possible relevance to a vendor-related e-mail"""
    for kw in keywords:
        synset = wn.get_synsets(kw)[0]
        for ref_synset in ref_synsets:
            similarity = metric(synset, ref_synset)
            if similarity >= threshold:
                return True


def check_keywords_for_entities(keywords, entities, threshold=.8):
    """Return any keywords that match an entity found in the text."""
    matches = []
    for keyword in keywords:
        for entity in entities:
            s = difflib.SequenceMatcher(None, keyword, entity)
            if s.ratio() >= threshold:
                matches.append(keyword)
    return matches


def get_vendor_counts_and_emails(keywords_and_entities_json, output, ref_synsets, wn_threshold=.8, wn_metric=wn.wup_similarity, str_match_threshold=.8):
    """
    Uses keywords and entity results to find vendors and get email count for each.

    Parameters
    ----------
    keywords_and_entities_json : str
        Path to JSON file with keywords and entities per text file
    output : str
        Path to output JSON
    ref_synsets : str
        past to file of vendor-related wordnet synset names to use as reference
    wn_threshold : float, optional
        Threshold for determing wordnet sysnet similarity (for checking relevence of 
        keywords to ref_synsets), by default .75
    wn_metric : _type_, optional
        Metric for comparing wordnet sysnets (for checking relevence of keywords to 
        ref_synsets), by default wn.wup_similarity
    str_match_threshold : float, optional
        Threshold for string similarity (for checking if keyword match entity names in 
        text), by default .8
    """
    kw_entity_ref = json.load(
        open(keywords_and_entities_json, encoding='utf-8'))['files']
    results_dict = {}
    ref_synset_names = open(ref_synsets, encoding='utf-8').read().split()
    ref_synsets = [wn.synset(name) for name in ref_synset_names]

    for i, (filename, v) in tqdm(list(enumerate(kw_entity_ref.items()))):
        if filename in results_dict:
            continue
        if i and not i % 100:
            json.dump(results_dict, open(
                output, 'w', encoding='utf-8'), default=list)
        keywords = v['keywords']
        if not check_keywords_for_relevance(keywords, ref_synsets, threshold=wn_threshold, metric=wn_metric):
            continue
        vendors = check_keywords_for_entities(
            keywords, threshold=str_match_threshold)
        results_dict[filename] = vendors
    json.dump(results_dict, open(
        output, 'w', encoding='utf-8'), default=list)


def graph_entities(entities_json, top_n_entities=None):
    results_dict = json.load(open(entities_json, encoding='utf-8'))
    entities = [entity for entities in results_dict.items()
                for entity in entities]
    counts = Counter(entities)
    counts = counts.most_common(top_n_entities)
    df = pd.DataFrame(list(counts))
    df = df.rename(columns={0: 'word', 1: 'count'})
    fig = px.bar(df, x='word', y='count')
    fig.show()


get_entities_and_keywords('email_text', 'entities_kws_up_to_3.json',
                          n_files=None, entities=('ORG',), kw_kwargs={})
