from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

models = {
    "Custom Model (English -> Vietnamese)": "phanlehieu27/vinai-en2vi-MedEV",
    "English->Vietnamese": "Helsinki-NLP/opus-mt-en-vi",
    "Vietnamese->English": "Helsinki-NLP/opus-mt-vi-en",
    "English->Japanese": "Helsinki-NLP/opus-mt-en-jap",
    "Japanese->English": "Helsinki-NLP/opus-mt-jap-en",
}


def translate(text, model):
    if model not in models.keys():
        raise ValueError(f"Mô hình '{model}' không được hỗ trợ.")

    if (model != "Custom Model (English -> Vietnamese)"):
        tokenizer = AutoTokenizer.from_pretrained(models[model])
        model = AutoModelForSeq2SeqLM.from_pretrained(models[model])

        batch = tokenizer([text], return_tensors="pt")
        generated_ids = model.generate(**batch)
        return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    else:
        tokenizer = AutoTokenizer.from_pretrained(models[model])
        model = AutoModelForSeq2SeqLM.from_pretrained(models[model])

        input_ids = tokenizer(text, padding=True, return_tensors="pt")
        input_ids = {k: v for k, v in input_ids.items()}

        output_ids = model.generate(
            **input_ids,
            decoder_start_token_id=tokenizer.lang_code_to_id["vi_VN"],
            num_return_sequences=1,
            num_beams=1,
        )

        vi_texts = tokenizer.batch_decode(
            output_ids, skip_special_tokens=True)
        return vi_texts[0]
