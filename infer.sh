#!/bin/bash

TEST_FILE="data/test/test_question_parsing_role.jsonl"
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
