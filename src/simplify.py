from itertools import combinations, product
from pprint import pprint
import bert_score
from langchain.prompts import PromptTemplate
from ollama import Client, generate
import torch

from call_llms import call_llms
from llms import get_llama
from prompts.prompts import SimplePrompt
from segment_sentences import join_paragraphs, paragraph_mask, segment

from transformers import AutoModelForMaskedLM, AutoTokenizer, pipeline
from transformers.utils import logging

import segment_sentences
from syntatic_analysis import extract_impersonal_verbs, extract_nouns, extract_verbs
logging.set_verbosity_error()

from semantic_similarity import get_similarity_score

def simplify_segments(segments):
    options = {'temperature': 0.0, 'top_k': 2, 'top_p': 0.05}
    responses = call_llms([
        {
            'model': 'llama3.1:8b',
            'prompt': SimplePrompt(f'Reescreva a seguinte frase, mantendo o tom e o sentido, visando facilitar o entendimento. Responda apenas com a frase reescrita, e mais nada.\nFrase original: "{sentence}"\nFrase reescrita: '),
            'options': options,
            'key': i
        } for i, sentence in enumerate(segments)
    ])
    responses = list(sorted(responses, key=lambda x: x[1])) # type: ignore
    simplified = [res[0]['response'].replace("\"", "").strip() for res in responses]

    scores = []
    for simple, sentence in zip(simplified, segments):
        score = get_similarity_score(simple, sentence, method='bertscore')
        scores.append(score)
    return simplified, scores

def build_from_simplified(segments, scores, original, similarity_threshold=0.8) -> tuple[list[str], float, float, float, list[str]]:
    """
    De uma lista de segmentos simplificados, constrói um texto final, mantendo os segmentos originais cuja simplificação não está acima do limite.
    """
    new_segments = []
    swapped = 0
    score = []
    for i, (simple, sentence) in enumerate(zip(segments, original)):
        if scores[i] < similarity_threshold:
            new_segments.append(sentence)
            score.append(1.0)
        else:
            new_segments.append(simple)
            swapped += 1
            score.append(scores[i])
    score = sum(score) / len(score)
    ratio = swapped / len(segments)

    mask = paragraph_mask(segments)
    simplified_merged = join_paragraphs(new_segments, mask)
    # print("Simplified merged:", simplified_merged)
    original_merged = join_paragraphs(original, mask)
    # print("Original merged:", simplified_merged)

    scores = []
    for simple, sentence in zip(simplified_merged, original_merged):
        _score = get_similarity_score(simple, sentence, method='bertscore')
        scores.append(_score)
    overall_score = sum(scores) / len(scores)

    return new_segments, score, ratio, overall_score, simplified_merged

def simplify(text) -> tuple[str, float, float]:
    segment = segment_sentences.segment(text)
    simplified, scores = simplify_segments(segment)
    simplified_paragraphs, score, ratio, overall_score, simplified_merged = build_from_simplified(simplified, scores, segment)
    simplified = "".join(simplified_paragraphs)
    return simplified, overall_score, ratio

def list_complex(text):
    prompt = PromptTemplate.from_template(
        """Faça uma lista de palavras complexas do meio jurídico presentes na seguinte frase. Responda com a lista, e nada mais.

Frase: "{text}"

Lista de palavras complexas:"""
    )

    llm = get_llama(model_configuration={'temperature': 0.0, 'top_p': 0.1, 'top_k': 1, 'seed': 42})
    chain = prompt | llm
    res = chain.invoke({'text': text}).content
    return res

def sinonimos(word, text):
    prompt = PromptTemplate.from_template(
        f"""
Responda com uma lista de 10 sinônimos diferentes e mais simples da palavra complexa "{word}" na frase abaixo. Responda com a lista de sinônimos, e nada mais.

Frase: "{text}"

Lista de sinônimos:
"""
    )
    llm = get_llama(model_configuration={'temperature': 0.1, 'top_p': 0.75})
    chain = prompt | llm
    res = chain.invoke({'word': word, 'text': text}).content
    print(res)
    return res

def bert_sinonimos(word, text):
    masked_text = "[CLS]" + text + "[SEP]" + text.replace(word, "[MASK]") + "[SEP]"
    # model_name = 'google-bert/bert-base-multilingual-uncased'
    model_name = 'rufimelo/Legal-BERTimbau-large'
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    pipe = pipeline('fill-mask', model=model, tokenizer=tokenizer, top_k=10)
    res = pipe(masked_text)

    sequences = [seq['sequence'].replace('[CLS] ', '').replace('[SEP]', '').strip() for seq in res]
    bertscores = [get_similarity_score(seq, text, method='bertscore', model_name=model_name, model_num_layers=24) for seq in sequences]
    cosines = [get_similarity_score(seq, text, method='cosine') for seq in sequences]
    avg_scores = [(bertscore * cosine) ** 0.5 for bertscore, cosine in zip(bertscores, cosines)]

    scores = sorted(list(zip(sequences, bertscores, cosines, avg_scores)), key=lambda x: x[1], reverse=True)
    return scores

# TODO: choose from
def personalize(text):
    """Reescreve uma frase, trocando verbos impessoais por verbos pessoais."""
    impersonal_verbs = extract_impersonal_verbs(text)
    nouns = extract_nouns(text)
    candidates = list(product(impersonal_verbs, nouns))
    rewritten = call_llms([
        {
            'model': 'llama3.1:8b',
            'prompt': f"""Por favor, reescreva a frase "{text}", mas utilizando um verbo pessoal no lugar do verbo '{verbo}', onde o sujeito é '{sujeito}'. A nova frase deve manter o sentido original, mas indicando que '{sujeito}' realizou a ação. Responda apenas com a frase reescrita, e mais nada.""",
            'options': {'temperature': 0.25, 'top_p': 0.25, 'top_k': 5},
            'key': i,
        } for i, (verbo, sujeito) in enumerate(candidates)
    ])
    rewritten = [r[0]['response'] for r in rewritten]
    scores = [get_similarity_score(r[0]['response'], text, method='bertscore') for r in rewritten]
    print(scores)
    return zip(rewritten, scores)

def collapse(text):
    prompt = """Reescreva o seguinte texto, de modo que se torne coerente, mantendo o tom e o sentido original. Responda apenas com o texto reescrito, e mais nada.

{context}

Texto reescrito:"""
    res = call_llms([
        {
            'model': 'llama3.1:8b',
            'prompt': prompt.format(context=text),
            'options': {'temperature': 0.0 + i / 10, 'top_p': 0.1 + i/10, 'top_k': 2 + i},
            'key': i
        } for i in range(1, 4)
    ],
        n_workers=3,
    )
    collapsed = [r[0]['response'] for r in res]
    return collapsed

def long_period(text):
    prompt = """A frase "{text}" é muito longa. Escreva a mesma ideia em frases separadas, mantendo o tom e o sentido original. Mencione todas as entidades presentes na frase original. Responda apenas com a nova versão do texto, e mais nada.

Nova versão:"""
    res = call_llms([
        {
            'model': 'llama3.1:8b',
            'prompt': prompt.format(text=text),
            'options': {'temperature': 0.0 + i / 10, 'top_p': 0.1 + i/10, 'top_k': 10 + i},
            'key': i
        } for i in range(1, 4)
    ],
    )
    versions = [r[0]['response'] for r in res]
    scores = [get_similarity_score(s, text, method='bertscore') for s in versions]
    return versions, scores
