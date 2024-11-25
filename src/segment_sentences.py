import spacy


def segment(text: str) -> list:
    nlp = spacy.load("pt_core_news_sm")
    doc = nlp(text)

    sentences = [sent.text for sent in doc.sents]
    return sentences

def paragraph_mask(segments: list[str]) -> list[int]:
    mask = []
    for i, segment in enumerate(segments):
        if segment.endswith("\n"):
            mask.append(i)
    return mask

def join_paragraphs(segments: list[str], mask: list[int]) -> list[str]:
    paragraphs = [[]]
    for i, segment in enumerate(segments):
        paragraphs[-1].append(segment.replace("\n", "").strip())
        if i in mask:
            paragraphs.append([])

    for i, paragraph in enumerate(paragraphs):
        paragraphs[i] = " ".join([sentence for sentence in paragraph])

    return paragraphs
