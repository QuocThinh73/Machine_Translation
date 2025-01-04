import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model = "./backend/best_model"
tokenizer_en2vi = AutoTokenizer.from_pretrained(model)
model_en2vi = AutoModelForSeq2SeqLM.from_pretrained(model)

device_en2vi = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_en2vi.to(device_en2vi)


def translate_en2vi(en_texts: str) -> str:
    input_ids = tokenizer_en2vi(en_texts, padding=True, return_tensors="pt")
    input_ids = {k: v.to(device_en2vi) for k, v in input_ids.items()}

    output_ids = model_en2vi.generate(
        **input_ids,
        decoder_start_token_id=tokenizer_en2vi.lang_code_to_id["vi_VN"],
        num_return_sequences=1,
        num_beams=1,
    )

    vi_texts = tokenizer_en2vi.batch_decode(
        output_ids, skip_special_tokens=True)
    return vi_texts[0]
