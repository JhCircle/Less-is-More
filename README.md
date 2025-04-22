<h1 align="center">🧠 Less is More: Enhancing Structured Multi-Agent Reasoning via Quality-Guided Distillation</h1>
<p align="center">
  <img src="./asset/less_is_more.png" width="30%" alt="Less is More: Structured Reasoning Framework"/>

---

</p>
<p align="center">
⭐ If you find this project helpful, please consider giving us a star to support the latest updates.
</p>

---
This repository provides the full implementation of our multi-agent structured reasoning framework, which distills high-quality step-by-step reasoning data into modular role agents. It is designed to help machines better understand, explain, and verify logic-based questions using structured instruction-following.

---

## 🚀 Highlights

- 🎯 **Modular Role Agents**: Specialized LLaMA-3 agents for parsing, statement extraction, evidence retrieval, and verification.
- 🔍 **Similarity-based Few-shot Demo Selection**: Retrieve top-k demos via embedding search (BGE-M3).
- 🏆 **Reward-Guided CoT Selection**: Filter out low-quality reasoning using a trained reward model.
- 🔧 **LoRA Efficient Fine-tuning**: Fast and lightweight training for each role agent.
- 🧱 **Fully Structured Output**: Easily used for further training, prompting, or evaluation.

---

## 📦 Installation

```bash
pip install -r requirements.txt
```

---

## 🗂️ Project Structure

```bash
.
├── data/                               # Raw and processed data
│   ├── train.txt                       # Raw LogicQA-style questions
│   ├── All_Train_With_Scores.jsonl     # CoT scoring results
│   ├── train/{strategy}_filtered.jsonl # Filtered by reward
│   ├── test/test_question_parsing_role.jsonl
│   ├── test/test_cot_parsing_role.jsonl 
│   └── test/test_cot_verify_role_role.jsonl
│
├── utils/
│   ├── prompt.py                      # Prompt templates
│   └── llm_utils.py                   # Inference / pipeline tools
│
│── data_synthesize.py             # Generate CoT + parsing
│── reward_filter.py               # Score CoT quality using reward model
│── extract_train_role.py          # Extract instruction-role data for training
│── extract_test_role.py           # Extract data for evaluation
│── train_qp.sh                    # Shell script for LoRA+ training on Question Parsing
│── train_cp.sh                    # Shell script for LoRA+ training on CoT Parsing
│── train_cv.sh                    # Shell script for LoRA+ training on CoT Verify (Statement+Verification)
│── infer.sh                       # Full structured inference pipeline
│
└── README.md
```

---

## 🛠️ How to Run

### 1️⃣ Step 1: Data Synthesis
Generate high-quality Question Parsing (QP), Chain-of-Thought Parsing (CP), and CoT Verification (CV: including both statement extraction and logical validation) from raw [LogicQA](https://github.com/lgw863/LogiQA-dataset) questions using GPT-4o via Retrieval-Augmented In-Context Learninig.
```bash
python data_synthesize.py \
  --demo_pool demo_pool.json \
  --logiqa_file data/train.txt \
  --output_file data/Train_LogicQA.jsonl \
  --embedding_model BAAI/bge-m3 \
  --tokenizer_name BAAI/bge-m3 \
  --model_id gpt-4o-2024-08-06 \
  --api_key YOUR_API_KEY \
  --base_url YOUR_OPENAI_API
```

---

### 2️⃣ Step 2: Score Quality with Reward Model
Use a reward model to evaluate CoT quality and retain only samples with **reward > 0**.

```bash
python reward_filter.py
```
#### 🎯 Strategy Options

| Strategy             | Description                                  |
|----------------------|----------------------------------------------|
| `with_few_shot`      | Select samples with high reward under few-shot prompting (reward > 0) |
| `without_few_shot`   | Select samples with high reward under zero-shot prompting (reward > 0) |
| `average` _(default)_| Select samples with highest average reward across both settings (reward > 0) |

Generates:

- `data/All_Train_With_Scores.jsonl`
- `data/with_few_shot_filtered.jsonl`
- `data/without_few_shot_filtered.jsonl`
- `data/average_filtered.jsonl`

---

### 3️⃣ Step 3: Extract Role-Based Instruction Data
Convert filtered CoT data into structured instruction formats for each role. Each file is used to train a different role agent (QP / CP / CV).
```bash
python scripts/extract_train_role.py
python scripts/extract_test_role.py
```

Outputs:

```bash
data/train/{strategy}/training_question_parsing_role.jsonl
data/train/{strategy}/training_cot_parsing_role.jsonl
data/train/{strategy}/training_cot_verify_role.jsonl
```

---

### 4️⃣ Step 4: Fine-tune Role Agents via LoRA+
Train each role agent (Question Parsing / CoT Parsing / CoT Verify) using reward-filtered data.
```bash
bash train_qp.sh
bash train_cv.sh
bash train_cs.sh
```

To switch filtering strategy (`with_few_shot`, `without_few_shot`, `average`, `all`), change this line in the `.sh` file:

```bash
strategy="average"
```
## ✅ Summary

| Role Agent   | Input File                                          | Task                        |
|--------------|------------------------------------------------------|-----------------------------|
| QP (Parser)  | `training_question_parsing_role.jsonl`              | Extract constraints/facts  |
| CP (Parser)  | `training_cot_parsing_role.jsonl`                   | Break CoT into statements   |
| CV (Verifier)| `training_cot_verify_role.jsonl`                    | Find evidence + verify logic|

---

### 5️⃣ Step 5: Run Multi-Agent Inference
Use the trained role agents to perform structured reasoning on new questions.

```bash
bash infer.sh

#!/bin/bash

TEST_FILE="test.jsonl"
QP_MODEL_PATH="./Question_Parsing"
CP_MODEL_PATH="./CoT_Parsing"
CV_MODEL_PATH="./CoT_Verify"
EMBEDDING_MODEL="BAAI/bge-m3"

python inference_pipeline.py \
  --test_file "$TEST_FILE" \
  --qp_model_id_or_path "$QP_MODEL_PATH" \
  --cp_model_id_or_path "$CP_MODEL_PATH" \
  --cv_model_id_or_path "$CV_MODEL_PATH" \
  --icl_embedding "$EMBEDDING_MODEL"

```

Produces `results.json` in the following structure:

```json
[
    {
        "question": "Fair use refers to the non-commercial use of works published by others without the permission of the copyright owner, and without having to pay remuneration under the circumstances specified in the law.The \"cases specified in the law\" mainly include: (1) Personal study, research or appreciation, using published works of others; (2) performing published works for free; (3) copying, painting, photography, video recording of artistic works installed or displayed in outdoor public places; (4) Translate published works created in Chinese and written into minority languages and publish works for publication.\nAccording to the above provisions, Which of the following are fair use:\nA.A sang an unpublished song at the class party\nB.B translates an English work into Mongolian work and publishes it\nC.Company C took the sculptures in the public square and made them into pictures.\nD.Ding Wei wrote a paper and copied a paper published by Geng in a journal for reference",
        "question_parsing": [
            "Fair use refers to the non-commercial use of works published by others without the permission of the copyright owner, and without having to pay remuneration under the circumstances specified in the law",
            "The cases specified in the law mainly include: (1) Personal study, research or appreciation, using published works of others",
            "Performing published works for free",
            "Copying, painting, photography, video recording of artistic works installed or displayed in outdoor public places",
            "Translate published works created in Chinese and written into minority languages and publish works for publication"
        ],
        "answer": "d",
        "id": 2021,
        "cot": "e definition of fair use mentions that it is the non-commercial use of works published by others without permission, and the main cases specified in the law include personal study, research, or appreciation, performing published works for free, copying artistic works displayed in outdoor public places, and translating published works into minority languages. Options B and D seem to fit into the category of fair use, as they involve translating and using published works for non-commercial purposes. Option C involves copying artistic works displayed in public places, which is also a specified case of fair use. Option A, however, involves singing an unpublished song, which is not a specified case of fair use.",
        "cot_parsing": [
            {
                "statement": "Options B and D fit into the category of fair use.",
                "evidence": "Options B and D seem to fit into the category of fair use, as they involve translating and using published works for non-commercial purposes.",
                "Verification": "true"
            },
            {
                "statement": "Option C involves fair use.",
                "evidence": "Option C involves copying artistic works displayed in public places, which is a specified case of fair use.",
                "Verification": "true"
            },
            {
                "statement": "Option A does not involve fair use.",
                "evidence": "Singing an unpublished song is not a specified case of fair use.",
                "Verification": "false"
            }
        ]
    }
]
```

---

## 📬 Contact

For any questions, suggestions, or collaborations, feel free to open an issue or start a discussion.
