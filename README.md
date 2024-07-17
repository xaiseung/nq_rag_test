# RAG Test for NQ
Test rag and non-rag(BART only) to [natural questions](https://huggingface.co/datasets/google-research-datasets/natural_questions)


HuggingFace, FAISS 라이브러리를 사용하여, Google의 Natural Questions Task에 대해 BART 홀로 학습시키는 것과 RAG+BART를 학습시키는 경우를 비교합니다.

코드 내에서 모델의 학습과 validation set에 대한 평가가 이루어집니다.

*TODO: 추가로, Prompt Tuning을 이용하여 학습 없는 RAG+일반목적LLM도 구현할 예정입니다.*

# Prerequisites

- [cutlass](https://github.com/NVIDIA/cutlass/blob/main/media/docs/quickstart.md)
- numpy
- torch
- [faiss](https://github.com/facebookresearch/faiss/blob/main/INSTALL.md)
- transformers
- datasets
- evaluate
- (optional for 'RAG w/o training'; langchain)
    - langchain
    - langchain-huggingface

# 실행방법

## Non-RAG
```
python 01_only_bart.py
```

## RAG

- [CUTLASS](https://github.com/NVIDIA/cutlass/blob/main/media/docs/quickstart.md)를 설치하고, 환경변수 `$CUTLASS_PATH=/path/to/cutlass`를 설정해야 합니다.
```
python 02_rag_bart.py
```

## RAG w/o training

**TODO**