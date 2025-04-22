from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm
import json
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import logging
import os
from utils.llm_utils import *
from utils.prompt import *
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("scoring_log.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

# Prompt templates
cot_synthesis = '''
### Instruction ###
You are an expert in logical and structural reasoning.
Your task is to provide short, clear reasoning for each statement. 
Each statement should logically follow from the previous one and be supported by the provided information. Keep your reasoning concise, using fewer than {budget} tokens.

### Examples ###
{few_shot_cot}
'''
cot_synthesis_instruction = '''
### Instruction ###
You are an expert in logical and structural reasoning.
Your task is to provide short, clear reasoning for each statement. 
Each statement should logically follow from the previous one and be supported by the provided information. Keep your reasoning concise, using fewer than {budget} tokens.
'''

import random
import numpy as np
import torch


def set_seed(seed=42):
    """Set the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
    torch.backends.cudnn.deterministic = True  # Ensures deterministic behavior
    torch.backends.cudnn.benchmark = False  # Disables optimizations that could cause non-determinism
    logger.info(f"Seed set to {seed}")


def load_models():
    """Load the required models."""
    try:
        reward_name = "Ray2333/GRM-Llama3.2-3B-rewardmodel-ft"
        rank_model = AutoModelForSequenceClassification.from_pretrained(reward_name,device_map="cuda", torch_dtype=torch.float16)
        tokenizer = AutoTokenizer.from_pretrained(reward_name)
        emb_tokenizer = AutoTokenizer.from_pretrained(
            'BAAI/bge-m3')
        emb_model = SentenceTransformer('BAAI/bge-m3').cuda()

        logger.info("Models loaded successfully")
        return rank_model, tokenizer, emb_model, emb_tokenizer

    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        raise


def load_data(emb_model):
    """Load training data and compute embeddings for candidates."""
    try:
        with open("Train_LogicQA.jsonl", "r", encoding="utf-8") as file:
            lines = [json.loads(line) for line in file.readlines()]

        with open("demo_pool.json", 'r', encoding='utf-8') as infile:
            data = json.load(infile)
            candidates = [entry["question"] for entry in data]
            candidates_embedding = emb_model.encode(candidates)

        logger.info(f"Loaded {len(lines)} training examples and {len(candidates)} candidates")
        return lines, data, candidates, candidates_embedding

    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise


def get_demo_cot(question, query_embedding, candidates_embedding, data):
    """Get example demonstrations for similar questions."""
    try:
        # Compute similarity and get top k examples
        similarity_scores = cosine_similarity([query_embedding], candidates_embedding)[0]
        scores, indices = torch.topk(torch.tensor(similarity_scores), k=5)

        # Build the demo string
        demo_cot = ""
        i=0
        for idx in indices:
            if data[idx]["question"] != question:  # Avoid using the current question as an example
                demo_cot += f'Example {i}:\nInput: {data[idx]["question"]} \nOutput: {data[idx]["cot"]}\n'
                i+=1

        return demo_cot, similarity_scores

    except Exception as e:
        logger.error(f"Error getting demo COT: {str(e)}")
        raise


def get_scores_both_versions(rank_model, tokenizer, question, answer, demo_cot):
    """Get scores for both versions (with and without few-shot examples)."""
    try:
        tokens = tokenizer.encode(question)
        token_count = len(tokens)

        with_few_shot = cot_synthesis.format(few_shot_cot=demo_cot,
                                             budget=int(token_count * 2.1)) + "\n###User Query###" + question

        message1 = [
            {'role': 'user',
             'content': with_few_shot},
            {'role': 'assistant',
             'content': answer}
        ]
        message_template = tokenizer.apply_chat_template(message1, tokenize=False)
        kwargs = {"padding": 'longest', "truncation": True, "return_tensors": "pt"}
        tokens = tokenizer.encode_plus(message_template, **kwargs)

        with torch.no_grad():
            reward_tensor = rank_model(tokens["input_ids"][0].view(1, -1).to("cuda"),
                                         attention_mask=tokens["attention_mask"][0].view(1, -1).to("cuda"))[0]
            score_with_few_shot = reward_tensor.cpu().detach()

        without_few_shot = cot_synthesis_instruction.format(
            budget=int(token_count * 2.1)) + "\n###User Query###" + question
        message2 = [
            {'role': 'user',
             'content': without_few_shot},
            {'role': 'assistant',
             'content': answer}
        ]
        message_template = tokenizer.apply_chat_template(message2, tokenize=False)
        kwargs = {"padding": 'longest', "truncation": True, "return_tensors": "pt"}
        tokens = tokenizer.encode_plus(message_template, **kwargs)

        with torch.no_grad():
            reward_tensor = rank_model(tokens["input_ids"][0].view(1, -1).to("cuda"),
                                         attention_mask=tokens["attention_mask"][0].view(1, -1).to("cuda"))[0]
            score_without_few_shot = reward_tensor.cpu().detach()
        avg_score = (score_with_few_shot.item() + score_without_few_shot.item()) / 2

        return {
            "with_few_shot": score_with_few_shot.item(),
            "without_few_shot": score_without_few_shot.item(),
            "average": avg_score
        }

    except Exception as e:
        logger.error(f"Error calculating scores: {str(e)}")
        raise


def reward_score():
    set_seed()
    try:
        # Load models and data
        rank_model, tokenizer, emb_model, emb_tokenizer = load_models()
        lines, data, candidates, candidates_embedding = load_data(emb_model)

        # Open the output file to write results in real-time
        with open("All_Train_With_Scores.jsonl", "w", encoding="utf-8") as output_file:
            # Process each question
            for i in tqdm(range(len(lines)), desc="Processing questions"):
                try:
                    line = lines[i]
                    question = line["question"]
                    answer = line["cot"]

                    # Get token count and embedding
                    query_embedding = emb_model.encode([question])[0]
                    # Get demo and similarity scores
                    demo_cot, similarity_scores = get_demo_cot(
                        question,
                        query_embedding,
                        candidates_embedding,
                        data
                    )
                    print(demo_cot)
                    # Get scores for both versions (with and without few-shot)
                    scores = get_scores_both_versions(
                        rank_model,
                        tokenizer,
                        question,
                        answer,
                        demo_cot
                    )
                    line["scores"] = scores

                    # Write the processed line to the output file
                    output_file.write(json.dumps(line, ensure_ascii=False) + "\n")
                    output_file.flush()  # Ensure data is written immediately

                except Exception as e:
                    logger.error(f"Error processing question {i}: {str(e)}")
                    continue

        logger.info("Results saved successfully in real-time.")

    except Exception as e:
        logger.error(f"Fatal error in main: {str(e)}")
        raise

if __name__ == "__main__":
    reward_score()

    import json
    import numpy as np

    with open("All_Train_With_Scores.jsonl", "r", encoding="utf-8") as file:
        lines = [json.loads(line) for line in file.readlines()]

    with_few_shot_filtered = []
    without_few_shot_filtered = []
    average_filtered = []

    for line in lines:
        scores = line["scores"]
        if scores["with_few_shot"] > 0:
            with_few_shot_filtered.append(line)
        if scores["without_few_shot"] > 0:
            without_few_shot_filtered.append(line)
        if scores["average"] > 0:
            average_filtered.append(line)


    def save_jsonl(data, filename):
        with open(filename, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')


    save_jsonl(with_few_shot_filtered, "data/with_few_shot_filtered.jsonl")
    save_jsonl(without_few_shot_filtered, "data/without_few_shot_filtered.jsonl")
    save_jsonl(average_filtered, "data/average_filtered.jsonl")

    print(f"Original: {len(lines)}")
    print(f"with_few_shot > 0: {len(with_few_shot_filtered)}")
    print(f"without_few_shot > 0: {len(without_few_shot_filtered)}")
    print(f"average > 0: {len(average_filtered)}")
