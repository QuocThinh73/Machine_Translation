import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from bertviz import model_view, head_view
# from bertviz.util import format_attention
import streamlit as st
import streamlit.components.v1 as components


model_name = "phanlehieu27/vinai-en2vi-MedEV"
tokenizer_en2vi = AutoTokenizer.from_pretrained(model_name)
model_en2vi = AutoModelForSeq2SeqLM.from_pretrained(
    model_name, output_attentions=True)
device_en2vi = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_en2vi.to(device_en2vi)


@st.cache_resource
def load_model():
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name, output_attentions=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.to(device_en2vi)
    return tokenizer, model


tokenizer_en2vi, model_en2vi = load_model()


def translate_en2vi(en_texts: str):
    input_ids = tokenizer_en2vi(
        en_texts, padding=True, return_tensors="pt").to(device_en2vi)
    outputs = model_en2vi.generate(
        **input_ids,
        decoder_start_token_id=tokenizer_en2vi.lang_code_to_id["vi_VN"],
        num_return_sequences=1,
        num_beams=1,
        early_stopping=True,
        output_attentions=True,
        return_dict_in_generate=True
    )
    vi_texts = tokenizer_en2vi.batch_decode(
        outputs.sequences, skip_special_tokens=True)

    encoder_attentions = outputs.encoder_attentions
    decoder_attentions = outputs.decoder_attentions
    cross_attentions = outputs.cross_attentions

    encoder_tokens = tokenizer_en2vi.convert_ids_to_tokens(
        input_ids.input_ids[0])
    decoder_tokens = tokenizer_en2vi.convert_ids_to_tokens(
        outputs.sequences[0])
    return vi_texts, encoder_attentions, decoder_attentions, cross_attentions, encoder_tokens, decoder_tokens


# SUI
st.title("BertViz App")
en_texts = st.text_input("Enter Input:")

if en_texts:
    vi_texts, encoder_attentions, decoder_attentions, cross_attentions, encoder_tokens, decoder_tokens = translate_en2vi(
        en_texts)

    st.write("Translate: ", vi_texts[0])

    num_encoder_layers = len(encoder_attentions)
    num_decoder_layers = len(decoder_attentions)
    num_cross_layers = len(cross_attentions)

    st.subheader("Head View")
    head_view_html = head_view(
        encoder_attentions,
        encoder_tokens,
        # display_mode='light',
        html_action='return'
    )
    components.html(head_view_html.data, height=800, scrolling=True)

    st.subheader("Model View")
    model_view_html = model_view(
        encoder_attention=encoder_attentions,
        decoder_attention=decoder_attentions[0],
        cross_attention=cross_attentions[0],
        encoder_tokens=encoder_tokens,
        # decoder_tokens=decoder_tokens,
        decoder_tokens=[encoder_tokens[-1]],
        # display_mode='light',
        html_action='return'
    )
    components.html(model_view_html.data, height=600, scrolling=True)
