from transformers import BertTokenizer, BertForTokenClassification, pipeline, Pipeline, PreTrainedModel

from split_documents import split_documents, split_text

class NER:
    _model_name = 'pierreguillou/ner-bert-large-cased-pt-lenerbr'
    _tokenizer: BertTokenizer
    _model: PreTrainedModel
    _ner: Pipeline

    def __init__(self, model_name=_model_name):
        self._tokenizer = BertTokenizer.from_pretrained(model_name)
        self._model = BertForTokenClassification.from_pretrained(model_name)
        self._ner = pipeline('ner', model=self._model, tokenizer=self._tokenizer)

    @property
    def model(self):
        return self._model

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def ner(self):
        return self._ner

    @staticmethod
    def split_text(text, chunk_size=512, chunk_overlap=32, split_by='character'):
        return split_text(text, chunk_size, chunk_overlap, split_by)

    @staticmethod
    def split_documents(file_path, page_start=None, page_end=None, chunk_size=512, chunk_overlap=32, split_by='character'):
        return split_documents(file_path, page_start, page_end, chunk_size=chunk_size, chunk_overlap=chunk_overlap, split_by=split_by)

    def multipage_ner(self, docs, flattened=False):
        pages = [doc.page_content for doc in docs]
        res = self.ner(pages)
        if flattened:
            res = self._flatten_matrix(res)
        return res

    @staticmethod
    def _flatten_matrix(matrix):
        flat_list = []
        for row in matrix:
            flat_list.extend(row)
        return flat_list

    @staticmethod
    def tokens_to_words(subwords):
        words = []
        for subword in subwords:
            w = subword['word']
            e = subword['entity']
            if len(w) < 3 or w[:2] != '##':
                words.append({'word': w, 'entity': e})
            else:
                words[-1]['word'] = words[-1]['word'] + w[2:]
        return words

    @staticmethod
    def words_to_topics(words):
        topics = []
        for word in words:
            if word['entity'][0] == 'B':
                topics.append({'word': word['word'], 'entity': word['entity'][2:]})
            elif word['entity'][0] == 'I':
                topics[-1]['word'] = topics[-1]['word'] + ' ' + word['word']
        return topics

    def get_topics(self, docs):
        topics = []
        for doc in docs:
            subwords = self.ner(doc.page_content)
            words = self.tokens_to_words(subwords)
            topics.extend(self.words_to_topics(words))
        return topics
