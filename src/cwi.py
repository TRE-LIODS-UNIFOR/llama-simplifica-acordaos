from wordfreq import zipf_frequency

def clean(text):
    words = text.split(" ")
    words = [word.replace(",", "").replace(".", "").replace("(", "").replace(")", "") for word in words]
    return " ".join(words)

def frequencies(text, sorted=False):
    _text = clean(text)
    freqs = [(word, zipf_frequency(word, 'pt')) for word in _text.split()]
    if sorted:
        freqs.sort(key=lambda x: x[1])
    return freqs

def cwi(text, sorted=False, threshold=3.0):
    freqs = frequencies(text, sorted=sorted)
    return [(word, freq) for word, freq in freqs if freq <= threshold]
