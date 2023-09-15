from collections import Counter
import json
import os

from flair.data import Sentence
from flair.models import SequenceTagger
from nltk import sent_tokenize
import pandas as pd
import plotly.express as px


def get_entities(tags, folder, output):
    tagger = SequenceTagger.load("flair/ner-english-fast")
    entity_dict = {}
    for filename in os.listdir(folder):
        email = open(os.path.join(folder, filename), encoding='utf-8').read()
        sents = sent_tokenize(email)
        for sent in sents:
            sent = Sentence(sent)
            tagger.predict(sent)
            for entity in sent.get_spans('ner'):
                if entity.tag in tags:
                    entity_dict.setdefault(
                        entity.text, {'tags': set(), 'files': set(), 'count': 0})
                    entity_dict['tags'].add(entity.tag)
                    entity_dict['files'].add(filename)
                    entity_dict['count'] += 1

    json.dump(open(output, 'w', encoding='utf-8'))


def graph_entities(data, min_count=0, top_n_entities_only=None):
    data = json.load(open(data, encoding='utf-8'))
    counts = {word: v['count']
              for word, v in data.items() if v['count'] >= min_count}
    counts = Counter(counts)
    counts = counts.most_common(top_n_entities_only)
    df = pd.DataFrame(list(counts))
    df = df.rename(columns={0: 'word', 1: 'count'})
    fig = px.bar(df, x='word', y='count')
    fig.show()
