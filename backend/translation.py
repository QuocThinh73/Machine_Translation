from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

models = {
    "English->Vietnamese": "Helsinki-NLP/opus-mt-en-vi",
    "Vietnamese->English": "Helsinki-NLP/opus-mt-vi-en",
    "English->Japanese": "Helsinki-NLP/opus-mt-en-jap",
    "Japanese->English": "Helsinki-NLP/opus-mt-jap-en",
    "Custom Model (English -> Vietnamese)": "phanlehieu27/vinai-en2vi-MedEV",
}


def translate(text, model):
    if model not in models.keys():
        raise ValueError(f"Mô hình '{model}' không được hỗ trợ.")

    tokenizer = AutoTokenizer.from_pretrained(models[model])
    model = AutoModelForSeq2SeqLM.from_pretrained(models[model])

    batch = tokenizer([text], return_tensors="pt")
    generated_ids = model.generate(**batch)
    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
