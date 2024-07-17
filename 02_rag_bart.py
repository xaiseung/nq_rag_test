import os
from collections.abc import Mapping
#import dotenv
#dotenv.load_dotenv()

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
file_handler = logging.FileHandler('recent_rag.log')
logger.addHandler(file_handler)


from transformers import BartTokenizer, RagTokenizer, RagRetriever, RagTokenForGeneration, AutoTokenizer
from transformers import DataCollatorForSeq2Seq
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
import datasets
import evaluate
import numpy as np
import torch
import transformers



def nested_detach(tensors):
    """
    Detach `tensors` (even if it's a nested list/tuple/dict of tensors).
    None Type에 대한 지원을 직접 작성했습니다.
    종종 Tensor 대신 NoneType을 반환하는데, 이에 대한 오류 핸들링입니다.
    """
    if isinstance(tensors, type(None)):
        return tensors
    if isinstance(tensors, (list, tuple)):
        return type(tensors)(nested_detach(t) for t in tensors)
    elif isinstance(tensors, Mapping):
        return type(tensors)({k: nested_detach(t) for k, t in tensors.items()})
    return tensors.detach()
transformers.trainer_pt_utils.nested_detach = nested_detach

QUESTION_ENCODER_ID = "facebook/dpr-question_encoder-single-nq-base"
MODEL_ID = "facebook/bart-large"
DEVICE = "cuda"

OUTPUT_DIR="./rag_output",
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
BATCH_EVAL_METRICS = True # 각 배치마다 metric을 계산할지 정합니다. True가 i/o에서 효율적이라 빠릅니다.
RESUME_FROM_CHECKPOINT = False

INPUT_MAXLEN = 128
OUTPUT_MAXLEN = 32


model = RagTokenForGeneration.from_pretrained_question_encoder_generator(
    QUESTION_ENCODER_ID,
    MODEL_ID,
    torch_dtype=torch.float16)

question_encoder_tokenizer = AutoTokenizer.from_pretrained(QUESTION_ENCODER_ID)
generator_tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# RagTokenizer는 두 토크나이저를 기억했다가 분리해서 사용합니다.
# tokenizer.decode시 tokenizer.generator가 불려온다는 것에 유의합시다.
tokenizer = RagTokenizer(question_encoder_tokenizer, generator_tokenizer)

tokenizer.pad = tokenizer.question_encoder.pad

model.config.use_dummy_dataset = False
model.config.index_name = "exact"

# RAG 검색자입니다. model.config에 있는 dataset을 읽어서, 이를 문서로 삼습니다.
retriever = RagRetriever(model.config, question_encoder_tokenizer, generator_tokenizer)
model.set_retriever(retriever)

# 빠른 평가를 위해 일부 텐서를 제외합니다.
model.config.keys_to_ignore_at_inference = ["context_input_ids", "question_encoder_last_hidden_state", "generator_enc_last_hidden_state"]
# 학습 loop에서 스칼라 loss를 받아야 합니다. 이를 제외하면 오류가 발생합니다.
model.config.reduce_loss = True




raw_datasets = datasets.load_dataset("google-research-datasets/natural_questions")

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
    data["answer"] = answer.strip()#+"</s>"
    return data

new_datasets = new_datasets.map(preprocessing)

def tokenizer_fn(example):
    """
    입력은 question_encoder의 토크나이저로 토큰화합니다.
    질문이 generator에 들어갈 때는, RagModel 내에서 알아서
    검색된 문서 + 질문 텍스트를 합쳐서 다시 generator의 토크나이저로 재토큰화합니다.
    """
    res = tokenizer.question_encoder(
        [ex["text"] for ex in example["question"]],
        truncation=True,
        max_length=INPUT_MAXLEN,
        add_special_tokens=True,
    )
    for k in res:
        example[k] = res[k]
    example["labels"] = tokenizer.generator(
        example["answer"],
        truncation=True,
        max_length=OUTPUT_MAXLEN,
        add_special_tokens=True,
    )["input_ids"]
    return example

tokenized_datasets = new_datasets.map(
    tokenizer_fn,
    batched=True,
    remove_columns=new_datasets["train"].column_names
)

trainer = None

def compute_metrics(eval_preds):
    """
    batch_eval_metrics=False일 때 사용하는 함수입니다.

    로그를 저장하고 exact_match 평가지표를 반환합니다.
    (특수토큰을 제외하고) 텍스트로 변환 시 예측 텍스트가 레이블과 같으면 hit입니다.

    RAG는 여러 문서의 확률 분포를 평균내다 보니 eos 토큰 이후로도 텍스트가 나올 수 있어,
    eos 이후 텍스트를 전부 제거합니다.

    label의 padding 토큰 id는 학습 과정에서 -100으로 치환되며 (torch loss에서 무시하기 위함입니다)
    매트릭이 labels을 반환받을때도 -100으로 변한 상태로 옵니다.
    지금은 0번째 id가 특수토큰이라서 단순 곱셉을 이용했습니다.
    """
    global trainer
    metric = evaluate.load("exact_match")
    outputs, labels = eval_preds
    logits, docs_score, *_ = outputs
    docs_prob = torch.softmax(torch.tensor(docs_score), dim=-1).unsqueeze(-1).unsqueeze(-1).numpy()
    marginalized_logits = np.sum(logits.reshape([labels.shape[0], -1, *logits.shape[-2:]])*docs_prob,axis=1)
    predictions = np.argmax(marginalized_logits, axis=-1)
    labels = labels * (labels != -100)
    pred_text = tokenizer.batch_decode(predictions, skip_special_tokens=False)
    for i, txt in enumerate(pred_text):
        eos_idx = txt.find(tokenizer.generator.eos_token)
        if eos_idx != -1:
            txt = txt[:eos_idx]
        pred_text[i] = txt.strip(tokenizer.generator.bos_token)
    label_text = tokenizer.batch_decode(labels, skip_special_tokens=True)
    for l in trainer.state.log_history:
        logger.info(f"{l}")
    logger.info(f"")
    logger.info(f"pred 0:\n{pred_text[0]}")
    logger.info(f"label 0:\n{label_text[0]}")
    return metric.compute(predictions=pred_text, references=label_text)



class ComputeMetricsInBatch:
    """
    batch_eval_metrics=True일 때 사용하는 클래스입니다.

    매트릭 결과를 누적해서 기억하다가, 반환이 필요할 때 (compute_result=True) 계산하여 반환합니다.
    """
    def __init__(self):
        self.accumul_metric = 0
        self.example_num = 0
        self.metric = evaluate.load("exact_match")
    def __call__(self, eval_preds, compute_result=False):
        global trainer
        metric = self.metric
        
        outputs, labels = eval_preds
        logits, docs_score, *_ = outputs
        if isinstance(logits, torch.Tensor):
            logits = logits.detach().to("cpu").numpy()
        if isinstance(docs_score, torch.Tensor):
            docs_score = docs_score.detach().to("cpu")
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().to("cpu").numpy()
        docs_prob = torch.softmax(docs_score, dim=-1).unsqueeze(-1).unsqueeze(-1).numpy()
        marginalized_logits = np.sum(logits.reshape([labels.shape[0], -1, *logits.shape[-2:]])*docs_prob,axis=1)
        predictions = np.argmax(marginalized_logits, axis=-1)
        labels = labels * (labels != -100)
        pred_text = tokenizer.batch_decode(predictions, skip_special_tokens=False)
        for i, txt in enumerate(pred_text):
            eos_idx = txt.find(tokenizer.generator.eos_token)
            if eos_idx != -1:
                txt = txt[:eos_idx]
            pred_text[i] = txt.strip(tokenizer.generator.bos_token)
        label_text = tokenizer.batch_decode(labels, skip_special_tokens=True)
        metric_result = metric.compute(predictions=pred_text, references=label_text)
        self.accumul_metric += metric_result["exact_match"] * len(pred_text)
        self.example_num += len(pred_text)
        if compute_result:
            for l in trainer.state.log_history:
                logger.info(f"{l}")
            logger.info(f"")
            logger.info(f"pred 0:\n{pred_text[0]}")
            logger.info(f"label 0:\n{label_text[0]}")
            res = {"exact_match": self.accumul_metric/self.example_num}
            self.accumul_metric = 0
            self.example_num = 0
            return res
        return metric_result

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer.question_encoder,
                                       padding=True,
                                       label_pad_token_id=tokenizer.generator.pad_token_id,
                                       model=model)

training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        do_eval=True,
        eval_strategy="steps",
        eval_accumulation_steps=1,
        batch_eval_metrics=BATCH_EVAL_METRICS,
        logging_steps=LOGGING_STEPS,
        eval_steps=EVAL_STEPS,
        save_steps=SAVE_STEPS,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE_PER_DEVICE,
        per_device_eval_batch_size=BATCH_SIZE_PER_DEVICE,
        fp16=True,
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
    compute_metrics=(ComputeMetricsInBatch() if BATCH_EVAL_METRICS else compute_metrics),
)

trainer.train(resume_from_checkpoint=RESUME_FROM_CHECKPOINT)