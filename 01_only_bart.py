import os
import dotenv
dotenv.load_dotenv()

from transformers import BartTokenizer, BartForConditionalGeneration
from transformers import DataCollatorForSeq2Seq
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
import datasets
import evaluate
import numpy as np

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
file_handler = logging.FileHandler('recent.log')
logger.addHandler(file_handler)


MODEL_ID = "facebook/bart-large"
DEVICE = "cuda"

OUTPUT_DIR="./bart_output",
LOGGING_STEPS=2000,
EVAL_STEPS=2000,
SAVE_STEPS=6000,
EPOCHS=10,
BATCH_SIZE_PER_DEVICE=16,
LEARNING_RATE = 0.00003,
WEIGHT_DECAY=0.001,
ADAM_EPSILION=1e-8,
MAX_GRAD_NORM = 1.0
WARMUP_STEPS = 250
GRADIENT_ACCUMULATION_STEPS=2,
RESUME_FROM_CHECKPOINT = False

tokenizer = BartTokenizer.from_pretrained(MODEL_ID)
model = BartForConditionalGeneration.from_pretrained(MODEL_ID, device_map=DEVICE)

raw_datasets = datasets.load_dataset("google-research-datasets/natural_questions")

# 정답이 없는 데이터를 제외합니다.
new_datasets = raw_datasets.filter(
    lambda batch: [len("".join([t for sa in d["short_answers"] for t in sa["text"] ]) )>0 for d in batch["annotations"]],
    batched=True)

def preprocessing(data):
    """
    short_answers 리스트 내 정답이 있는 것을 찾아 그것을 정답으로 저장합니다.

    validation의 short_answers는 각 원소가 한명의 정답자 대답에 해당하는 원소로 이루어져 있습니다.
    많은 문제가 한명 또는 소수의 정답자만이 질문에 대답을 합니다. 때문에 정답이 있는 한명의 정답자만을 가져옵니다.
    대답 (sa["text"])또한 리스트인데, 각각의 텍스트가 정답이 아닌 (쉼표로 나누어) 모두 대답해야 정답으로 취급합니다.
    """
    sa_list = data["annotations"]["short_answers"]
    answer = ""
    for sa in sa_list:
        if len(sa["text"]) != 0:
            answer = ', '.join(sa["text"])
            break
    data["answer"] = answer.strip()+"</s>"
    return data

new_datasets = new_datasets.map(preprocessing)

def tokenizer_fn(example):
    res = tokenizer([ex["text"] for ex in example["question"]], add_special_tokens=False)
    for k in res:
        example[k] = res[k]
    example["labels"] = tokenizer(example["answer"], add_special_tokens=False)["input_ids"]
    return example

tokenized_datasets = new_datasets.map(
    tokenizer_fn,
    batched=True,
    remove_columns=new_datasets["train"].column_names
)


def compute_metrics(eval_preds):
    """
    exact_match 평가지표를 반환합니다.
    (특수토큰을 제외하고) 텍스트로 변환 시 예측 텍스트가 레이블과 같으면 hit입니다.

    label의 padding 토큰 id는 학습 과정에서 -100으로 치환되며 (torch loss에서 무시하기 위함입니다)
    매트릭이 labels을 반환받을때도 -100으로 변한 상태로 옵니다.
    지금은 0번째 id가 특수토큰이라서 단순 곱셉을 이용했습니다.
    """
    metric = evaluate.load("exact_match")
    outputs, labels = eval_preds
    logits, enc_last_hidden = outputs
    predictions = np.argmax(logits, axis=-1)
    labels = labels * (labels != -100)
    pred_text = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    label_text = tokenizer.batch_decode(labels, skip_special_tokens=True)
    return metric.compute(predictions=pred_text, references=label_text)

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        do_eval=True,
        eval_strategy="steps",
        eval_accumulation_steps=1,
        logging_steps=LOGGING_STEPS,
        eval_steps=EVAL_STEPS,
        save_steps=SAVE_STEPS,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE_PER_DEVICE,
        per_device_eval_batch_size=BATCH_SIZE_PER_DEVICE,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        adam_epsilon=ADAM_EPSILION,
        max_grad_norm=MAX_GRAD_NORM,
        lr_scheduler_type="polynomial",
        warmup_steps=WARMUP_STEPS,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        eval_on_start = True,
)

trainer = Seq2SeqTrainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train(resume_from_checkpoint=RESUME_FROM_CHECKPOINT)