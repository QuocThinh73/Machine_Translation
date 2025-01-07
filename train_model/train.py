from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)
from evaluate import load
from transformers import TrainerCallback
import torch
import json
# =======================
# 1. SPLIT DATASET
# =======================


def split_source_target(dataset):
    """
    Tách cột `text` thành `source` và `target` dựa vào số lượng dòng.
    """
    num_rows = len(dataset)
    mid_index = num_rows // 2  # Chia đôi dataset

    # Chia dữ liệu thành hai phần
    source_data = dataset.select(range(0, mid_index))  # Nửa trên: source
    target_data = dataset.select(
        range(mid_index, num_rows))  # Nửa dưới: target

    # Tạo dataset mới với cột `source` và `target`
    new_dataset = Dataset.from_dict({
        "source": source_data["text"],
        "target": target_data["text"]
    })
    return new_dataset


# Tải dataset
ds = load_dataset("nhuvo/MedEV")

# Tách source-target cho các tập train, validation, test
ds["train"] = split_source_target(ds["train"])
ds["validation"] = split_source_target(ds["validation"])
ds["test"] = split_source_target(ds["test"])


# Kiểm tra kết quả
print("Sample source-target (train):", ds["train"][0])
print("Sample source-target (train last):", ds["train"][-1])

# =======================
# 2. TOKENIZER
# =======================
model_name = "vinai/vinai-translate-en2vi-v2"
tokenizer = AutoTokenizer.from_pretrained(
    model_name, src_lang="en_XX", tgt_lang="vi_VN")


def tokenize_function(examples):
    """
    Tokenize source và target.
    """
    # Token hóa source và target
    source_encodings = tokenizer(
        examples["source"], truncation=True, padding="max_length", max_length=128
    )
    target_encodings = tokenizer(
        examples["target"], truncation=True, padding="max_length", max_length=128
    )

    return {
        "input_ids": source_encodings["input_ids"],
        "attention_mask": source_encodings["attention_mask"],
        "labels": target_encodings["input_ids"],
    }


# Áp dụng tokenization
tokenized_datasets = {
    "train": ds["train"].map(tokenize_function, batched=True),
    "validation": ds["validation"].map(tokenize_function, batched=True),
    "test": ds["test"].map(tokenize_function, batched=True),
}

# Kiểm tra tokenization
print("Tokenized sample:", tokenized_datasets["train"][0])

# =======================
# 3. MODEL
# =======================
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
# Thiết lập decoder_start_token_id để sinh văn bản tiếng Việt
model.config.decoder_start_token_id = tokenizer.lang_code_to_id["vi_VN"]
# =======================
# 4. TRAINER SETUP
# =======================
# Tải metrics BLEU và TER
bleu_metric = load("sacrebleu")


def compute_metrics(eval_pred):
    """
    Tính toán BLEU và TER.
    """
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(
        predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    bleu = bleu_metric.compute(predictions=decoded_preds, references=[
                               [l] for l in decoded_labels])

    return {"bleu": bleu["score"]}


# Data collator
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# Callback để log tiến trình


# Biến lưu log
log_data = []

# Sửa callback để ghi log vào file JSON


class LoggingCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            log_entry = {"step": state.global_step}
            # Thêm tất cả giá trị từ `logs` vào log_entry
            log_entry.update(logs)
            log_data.append(log_entry)
            print(f"Step {state.global_step}: {logs}")

            # Ghi log vào file JSON
            with open("./training_logs.json", "w") as f:
                json.dump(log_data, f, indent=4)


# Cấu hình huấn luyện
training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    evaluation_strategy="steps",
    eval_steps=2000,
    save_steps=2000,
    save_total_limit=3,
    learning_rate=5e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    warmup_steps=1250,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=2000,
    predict_with_generate=True,
    generation_max_length=128,
    fp16=True,
    ddp_find_unused_parameters=False,  # Tối ưu DDP
    report_to="none",
)


# Tạo Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[LoggingCallback()],
)

# =======================
# 5. TRAINING
# =======================
if __name__ == "__main__":
    # Huấn luyện mô hình
    trainer.train()

    # Lưu mô hình tốt nhất
    trainer.save_model("./best_model")
    tokenizer.save_pretrained("./best_model")
