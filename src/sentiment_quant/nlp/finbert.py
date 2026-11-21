from transformers import pipeline

MODEL = "ProsusAI/finbert"
_classifier = None  # loaded once, then reused


def get_classifier():
    global _classifier
    if _classifier is None:
        _classifier = pipeline("text-classification", model=MODEL, top_k=None)
    return _classifier


def score_texts(texts):
    results = get_classifier()(texts, truncation=True, max_length=512)
    rows = []
    for r in results:
        probs = {x["label"].lower(): x["score"] for x in r}
        rows.append({
            "p_positive": probs["positive"],
            "p_negative": probs["negative"],
            "p_neutral": probs["neutral"],
        })
    return rows


if __name__ == "__main__":
    headlines = [
        "Company reports record quarterly profit and raises guidance",
        "Shares plunge after company misses earnings estimates",
        "Company holds annual shareholder meeting on Tuesday",
    ]
    for text, s in zip(headlines, score_texts(headlines)):
        print(text)
        print("  ", {k: round(v, 3) for k, v in s.items()})