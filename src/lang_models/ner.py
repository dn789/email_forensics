from flair.data import Sentence
from flair.models import SequenceTagger
from nltk import sent_tokenize
from transformers import pipeline


class NerCompanyTagger:
    def __init__(self) -> None:
        self.tagger = pipeline(
            task='ner', model="nbroad/deberta-v3-base-company-names")

    def __call__(self, text: str, return_set: bool = True, do_sent_tokenize: bool = True) -> set[str] | list[str]:
        if do_sent_tokenize:
            text = sent_tokenize(text)
        else:
            text = [text]  # type: ignore
        orgs, current_org = set(), ''

        for batch in text:
            if len(batch) > 2400:
                continue
            output: list[dict] = self.tagger(batch)  # type: ignore
            for d in output:
                if d['entity'].startswith('B'):
                    if current_org:
                        orgs.add(current_org)
                        current_org = ''
                    current_org += batch[d['start']:d['end']]
                else:
                    current_org += batch[d['start']:d['end']]
            if current_org:
                orgs.add(current_org)
                current_org = ''

        orgs = [org.strip().replace('\n', '') for org in orgs]
        if return_set:
            orgs = set(orgs)
        return orgs


class FlairTagger:
    """Uses flair ontonotes NER."""

    def __init__(self):
        self.tagger = SequenceTagger.load("flair/ner-english")

    def __call__(self, text: str, target_tags: tuple = ('LOCATION', 'PER')) -> dict[str, set[str]]:
        entities_d = {tag: set() for tag in target_tags}
        sents = sent_tokenize(text)
        for sent in sents:
            sent = Sentence(sent)
            self.tagger.predict(sent)
            for entity in sent.get_spans('ner'):
                if entity.tag in target_tags:
                    entities_d[entity.tag].add(entity.text)
        return entities_d
