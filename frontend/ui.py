import streamlit as st
import requests

st.set_page_config(layout="wide", page_title="Machine Translation")

backend_url = "http://localhost:8000"

try:
    response = requests.get(f"{backend_url}/models")
    if response.status_code == 200:
        model_list = response.json().get("models", [])
    else:
        st.error(
            f"Lỗi khi lấy danh sách mô hình: {response.status_code} - {response.text}")
        model_list = []
except requests.exceptions.RequestException as e:
    st.error(f"Không thể kết nối đến backend để lấy danh sách mô hình: {e}")
    model_list = []

st.markdown("<h1 style='text-align: center; color: blue;'>Ứng dụng dịch văn bản</h1>",
            unsafe_allow_html=True)

st.header("Chọn mô hình dịch")

if model_list:
    model_option = st.selectbox("", model_list)
else:
    st.warning("Không có mô hình nào khả dụng.")
    model_option = None

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Văn bản gốc")
    text_input_1 = st.text_area(
        "", placeholder="Nhập văn bản", height=450, key="test_input_1")

with col2:
    st.markdown("#### Bản dịch")
    translated_placeholder_1 = st.empty()

if st.button("Dịch", use_container_width=True, key="translate_button_1"):
    if not text_input_1.strip():
        st.warning("Bạn chưa nhập văn bản.")
    else:
        try:
            response = requests.post(
                f"{backend_url}/translation", json={"text": text_input_1, "model": model_option})

            if response.status_code == 200:
                translated_text = response.json().get("translated_text", "Không có kết quả")
                translated_placeholder_1.text_area(
                    "", value=translated_text, disabled=True, height=450, key="translated_output_1")
            else:
                st.error(
                    f"Lỗi từ backend: {response.status_code} - {response.text}")
        except requests.exceptions.RequestException as e:
            st.error(f"Không thể kết nối đến backend: {e}")
