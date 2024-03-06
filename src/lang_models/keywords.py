# class KwModel():
#     """Uses KeyBERT to get keywords."""

#     def __init__(self, kwargs={}):
#         self.kwargs = kwargs
#         self.kw_model = KeyBERT()

#     def __call__(self, docs):
#         vectorizer = self.kwargs.get('vectorizer')
#         if vectorizer == 'keyphrase':
#             self.kwargs['vectorizer'] = KeyphraseCountVectorizer()
#         elif vectorizer == 'count':
#             self.kwargs['vectorizer'] = CountVectorizer(
#                 ngram_range=self.kwargs.get(
#                     'keyphrase_ngram_range', (1, 1)),
#                 stop_words=self.kwargs['stop_words']
#             )
#         results = []
#         results = self.kw_model.extract_keywords(docs, **self.kwargs)
#         if type(docs) == str:
#             results = [x[0] for x in results]
#         else:
#             for i, kw_set in enumerate(results):
#                 results[i] = [x[0] for x in kw_set]
#         return results
