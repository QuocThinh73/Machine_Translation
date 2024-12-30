from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

models = [
    "Helsinki-NLP/opus-mt-en-vi",
    "Helsinki-NLP/opus-mt-vi-en",
    "Helsinki-NLP/opus-mt-en-jap",
    "Helsinki-NLP/opus-mt-jap-en",
]


def translate(text, model):
    if model not in models:
        raise ValueError(f"Mô hình '{model}' không được hỗ trợ.")
    tokenizer = AutoTokenizer.from_pretrained(model)
    model = AutoModelForSeq2SeqLM.from_pretrained(model)
    batch = tokenizer([text], return_tensors="pt")
    generated_ids = model.generate(**batch)
    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
