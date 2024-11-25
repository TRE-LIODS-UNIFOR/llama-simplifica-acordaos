from pprint import pprint
import bert_score

from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch

model_name = 'rufimelo/Legal-BERTimbau-large'

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)


def simplify_word(sentence, complex_word, top_k=5):
    # Mask the complex word in the sentence
    masked_sentence = sentence.replace(complex_word, tokenizer.mask_token)

    # Tokenize the masked sentence
    inputs = tokenizer(masked_sentence, return_tensors="pt")

    # Predict masked token
    with torch.no_grad():
        logits = model(**inputs).logits

    # Find the masked token position
    mask_token_index = torch.where(inputs.input_ids == tokenizer.mask_token_id)[1]

    # Get top K predicted tokens for the mask
    mask_token_logits = logits[0, mask_token_index, :]
    top_k_tokens = torch.topk(mask_token_logits, top_k, dim=1).indices[0].tolist()

    # Decode and filter the predicted tokens for simplicity
    suggestions = []
    for token in top_k_tokens:
        suggestion = tokenizer.decode([token]).strip()
        # Filter: exclude the complex word and ensure it’s a valid simplification
        if suggestion != complex_word and suggestion.isalpha():
            suggestions.append(suggestion)

    return suggestions if suggestions else [complex_word]

def simplify_sentence(sentence, words):
    simplified = sentence
    for word in words:
        suggestion = simplify_word(simplified, word)[0]
        simplified = simplified.replace(word, suggestion)
    return simplified


if __name__ == "__main__":
    sentence = "O recorrente Antônio Ilomar Vasconcelos Cruz impugna a sentença proferida pelo Juízo Eleitoral da 33ª Zona - Canindé/CE que julgou parcialmente procedente os pedidos da Representação Eleitoral por propaganda antecipada e condenou o recorrente ao pagamento de multa no valor de R$ 15.000,00."
    complex_word = ["impugna", "proferida", "julgou"]
    simplified = simplify_sentence(sentence, complex_word)
    [
    "recorrente",
    "impugna",
    "proferida",
    "juízo Eleitoral",
    "zona",
    "representação Eleitoral",
    "propaganda antecipada",
    "condenou",
    "multa",
    "parcialmente",
    "procedente",
    ]

    print(simplified)
    # scores = []
    # for i, suggestion in enumerate(simplified, 1):
    #     print(f"{i}. {suggestion}")
    #     fixed = sentence.replace(complex_word, suggestion)
    #     score = bert_score.score([fixed], [sentence], lang='pt', model_type=model_name)[2].item()
    #     scores.append((fixed, score, bert_score.score([suggestion], [complex_word], lang='pt', model_type=model_name)[2].item()))

    # sentence = scores[0][0]
    # complex_word = 'proferida'
    # simplified = simplify_word(sentence, complex_word, top_k=10)
    # scores = []
    # for i, suggestion in enumerate(simplified, 1):
    #     print(f"{i}. {suggestion}")
    #     fixed = sentence.replace(complex_word, suggestion)
    #     score = bert_score.score([fixed], [sentence], lang='pt', model_type=model_name)[2].item()
    #     scores.append((fixed, score, bert_score.score([suggestion], [complex_word], lang='pt', model_type=model_name)[2].item()))

    # pprint(scores)
