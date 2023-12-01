import re

from flair.data import Sentence
from flair.models import SequenceTagger
from keybert import KeyBERT
from keyphrase_vectorizers import KeyphraseCountVectorizer
from nltk import sent_tokenize
from sklearn.feature_extraction.text import CountVectorizer


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
