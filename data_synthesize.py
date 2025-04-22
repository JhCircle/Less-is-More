import json
import logging
from tqdm import tqdm
from utils.llm_utils import *
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import torch
from utils.prompt import *
import os
from transformers import AutoTokenizer
import simplejson
import re
import argparse


# ========== argparse ==========
def parse_args():
    parser = argparse.ArgumentParser(description="Logic QA Data Processor")

    parser.add_argument('--demo_pool', type=str, default="demo_pool.json",
                        help="Path to training data JSON")
    parser.add_argument('--logiqa_file', type=str, default="data/Train.txt", help="Path to test text file")
    parser.add_argument('--output_file', type=str, default="Train_LogicQA.jsonl", help="Output file path (.jsonl)")
    parser.add_argument('--embedding_model', type=str, default="BAAI/bge-m3", help="Sentence embedding model")
    parser.add_argument('--tokenizer_name', type=str, default="BAAI/bge-m3", help="Tokenizer for counting tokens")
    parser.add_argument('--model_id', type=str, default="gpt-4o-2024-08-06", help="LLM model ID")
    parser.add_argument('--api_key', type=str, required=True, help="Your Openai API Key")
    parser.add_argument('--base_url', type=str, help="Base URL for Openai API")

    return parser.parse_args()


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("process_log.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

def is_valid_json(s):
    try:
        parsed = simplejson.loads(s, strict=False)
        if isinstance(parsed, (list)):
            return True
        else:
            return False
    except ValueError:
        return False
args = parse_args()
processed_questions = set()
if os.path.exists(args.output_file):
    with open(args.output_file, "r", encoding="utf-8") as infile:
        for line_number, line in enumerate(infile, 1):
            try:
                entry = json.loads(line.strip())
                if "question" in entry:
                    processed_questions.add(entry["question"])
                else:
                    print(f"Line {line_number}: 'question' field missing.")
            except json.JSONDecodeError:
                print(f"Line {line_number}: JSON decoding error.")

base_url = args.base_url
pipeline = Pipeline(model_id=args.model_id, base_url=base_url, api_key=args.api_key, prob=True)
tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
model = SentenceTransformer(args.embedding_model).cuda()

with open(args.demo_pool, 'r', encoding='utf-8') as infile:
    data = json.load(infile)
    candidates = [entry["question"] for entry in data]
    candidates_embedding = model.encode(candidates)

with open(args.logiqa_file, "r", encoding="utf-8") as file:
    lines = file.readlines()

with open(args.output_file, "a", encoding="utf-8") as outfile:
    for i in tqdm(range(len(lines))):
        line = lines[i].strip()

        if line.lower() in ["c", "a", "b", "d"]:
            answer = line.lower()
            content = lines[i + 1].strip()
            q = lines[i + 2].strip()
            a = lines[i + 3].strip()
            b = lines[i + 4].strip()
            c = lines[i + 5].strip()
            d = lines[i + 6].strip()

            if q in processed_questions:
                logger.info(f"Skipping already processed question: {question}")
                continue

            question = content + "\n" + q + "\n" + a + "\n" + b + "\n" + c + "\n" + d
            query = question

            tokens = tokenizer.encode(query)
            token_count = len(tokens)
            query_embedding = model.encode([query])[0]
            similarity_scores = model.similarity(query_embedding, candidates_embedding)[0]
            scores, indices = torch.topk(similarity_scores, k=5)
            demo_cot = ""
            demo_question = ""
            demo_cot_parsing = ""
            for idx in indices:
                if data[idx]["question"] != query:
                    demo_cot += f'Example {idx}:\nInput: {data[idx]["question"]} \n The answer is {data[idx]["answer"]}\n Ouput: {data[idx]["cot"]}\n'
                    demo_question += f'Example {idx}:\nInput: {data[idx]["question"]} \n Ouput: {data[idx]["question_parsing"]}\n'
                    demo_cot_parsing += f'Example {idx}:\nInput: {data[idx]["question"]} {data[idx]["cot"]}\nOutput: {{"cot_parsing": "{data[idx]["cot_parsing"]}"}}\n'

            if answer != None:
                query = f"{question} The answer is {answer}.\n"

            cot = pipeline.get_respond(cot_synthesis.format(few_shot_cot=demo_cot, budget=int(token_count * 2.1)), query, max_tokens=int(token_count *3))
            print(cot)
            cot_query = f"{question} {cot}\n"
            cot_parsing_i = pipeline.get_respond(cot_parsing.format(few_shot_cot=demo_cot_parsing), cot_query,
                                               max_tokens=1024,json_format=True)

            print(cot_parsing_i)
            try:
                if not is_valid_json(cot_parsing_i):
                    pattern = r'\[.*?\]'
                    matches = re.findall(pattern, cot_parsing_i, re.DOTALL)
                    cot_parsing_r = simplejson.loads(matches[0], strict=False)
                else:
                    cot_parsing_r = simplejson.loads(cot_parsing_i, strict=False)
            except Exception as e:
                logger.error(f"Error parsing CoT for item {i}: {str(e)}")
                continue

            question_parsing = pipeline.get_respond(question_parsing_prompt.format(few_shot_example=demo_question), question, max_tokens=1024)
            try:
                x = eval(question_parsing)
                if len(x)<=2:
                    continue
            except Exception as e:
                logger.error(f"Error parsing question for item {i}: {str(e)}")
                continue

            alld = {
                "question": question,
                "id": i,
                "question_parsing": x,
                "answer": answer,
                "cot": cot,
                "cot_parsing": cot_parsing_r,
            }
            logger.info(f"Finished item {i}: {question}")

            json.dump(alld, outfile, ensure_ascii=False)
            outfile.write("\n")
