from functools import lru_cache
import spacy


@lru_cache(maxsize=None)
def process_text(text):
    nlp = spacy.load("pt_core_news_sm")
    doc = nlp(text)
    return doc

def extract_class(text, f):
    doc = process_text(text)
    tokens = list(filter(f, doc))
    return tokens

def extract_verbs(text):
    verbs = extract_class(text, lambda token: token.pos_ in {"VERB"} and token.dep_ in {"ROOT", "aux"})
    return verbs

def extract_impersonal_verbs(text):
    """Verbs without nsubj as child"""
    impersonal_verbs = extract_class(text, lambda token: token.pos_ in {"VERB"} and token.dep_ in {"ROOT", "aux"} and not any(child.dep_ == "nsubj" for child in token.children))
    return impersonal_verbs

def extract_nouns(text, deps={"nsubj", "obl:agent", "obl", "nsubjpass"}):
    nouns = extract_class(text, lambda token: token.pos_ in {"NOUN", "PROPN", "PRON"} and token.dep_ in deps)
    return nouns
