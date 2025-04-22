import os
import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', package,'--no-deps'])
import os
import sys
import subprocess

import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--test_file",
    type=str,
    help="test file path"
)
parser.add_argument(
    "--qp_model_id_or_path",
    type=str,
    help="Question_Parsing Llama3-8B-Instruct model path"
)
parser.add_argument(
    "--cp_model_id_or_path",
    type=str,
    help="CoT_Parsing Llama3-8B-Instruct model path"
)
parser.add_argument(
    "--cv_model_id_or_path",
    type=str,
    help="CoT_Verify Llama3-8B-Instruct model path"
)

parser.add_argument(
    "--icl_embedding",
    type=str,
    default= "BAAI/bge-m3",
    help="embedding model"
)

try:
    import torch
    import simplejson as json

    from swift.llm import (
        get_model_tokenizer, get_template, inference, ModelType, get_default_template_type,
    )
    from swift.utils import seed_everything
    from transformers import AutoTokenizer
    from sentence_transformers import SentenceTransformer

    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:

    if os.path.exists("requirements.txt"):
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    import torch
    import simplejson as json

    from swift.llm import (
        get_model_tokenizer, get_template, inference, ModelType, get_default_template_type,
    )
    from swift.utils import seed_everything
    from transformers import AutoTokenizer
    from sentence_transformers import SentenceTransformer

    from sklearn.metrics.pairwise import cosine_similarity

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

args = parser.parse_args()
test_file = args.test_file
qp_model_id_or_path = args.qp_model_id_or_path
cp_model_id_or_path = args.cp_model_id_or_path
cs_model_id_or_path = args.cs_model_id_or_path
icl_embedding=args.icl_embedding

try:
    with open("demo_pool.json", "r", encoding="utf-8") as file:
        data = json.load(file)
except Exception as e:
    print("loading Demo pools...")

gpu_count = torch.cuda.device_count()
gpu_ids = ','.join(str(i) for i in range(gpu_count))
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids
print("Using GPU: ", os.environ['CUDA_VISIBLE_DEVICES'])

# Load embedding model
tokenizer_emb = AutoTokenizer.from_pretrained(icl_embedding)
emb_model = SentenceTransformer(icl_embedding).cuda()
# Encode candidate questions
candidates = [entry["question"] for entry in data]
candidates_embedding = emb_model.encode(candidates)


model_type = ModelType.llama3_8b_instruct
template_type = get_default_template_type(model_type)
print(f'template_type: {template_type}')

kwargs = {}
qpmodel, qptokenizer = get_model_tokenizer(
    model_type,
    model_id_or_path=qp_model_id_or_path,
    model_kwargs={'device_map': 'auto'},
    **kwargs
)
qpmodel.generation_config.max_new_tokens = 1024
qpmodel.generation_config.temperature=0.0
qpmodel.generation_config.do_sample=False
qpmodel.generation_config.top_p=0.97
print(qpmodel.generation_config)

kwargs = {}
cpmodel, cptokenizer = get_model_tokenizer(
    model_type,
    model_id_or_path=cp_model_id_or_path,
    model_kwargs={'device_map': 'auto'},
    **kwargs
)
cpmodel.generation_config.max_new_tokens = 1024
cpmodel.generation_config.temperature=0.0
cpmodel.generation_config.do_sample=False
cpmodel.generation_config.top_p=0.97
print(cpmodel.generation_config)

kwargs = {}
cv_model_id_or_path=args.cv_model_id_or_path
cvmodel, cvtokenizer = get_model_tokenizer(
    model_type,
    model_id_or_path=cv_model_id_or_path,
    model_kwargs={'device_map': 'auto'},
    **kwargs
)


# Prompts
question_parsing_prompt = '''###Instruction###
Extract the constraints and key details from a problem description, ignoring any specific questions or answer choices. 
Focus on the rules or conditions given that are necessary to solve the problem, and extract these in a clear, descriptive list.
###Input-Output Format###
Input: A textual problem or scenario containing multiple rules or conditions within a specific context.
Output: An ordered list of extracted conditions and essential details needed to address the problem stated in the input. Each extracted condition should be clearly and concisely formatted, capturing only the facts necessary for determining the problem's solution.
###Examples###
{few_shot_example}
'''

cot_parsing_statement_prompt = '''###Instruction###
You are an expert in logical reasoning and structural analysis.
Your task is to identify and extract all distinct statements from the given question conditions and chain-of-thought (CoT) content.
- Extract explicitly stated and logically implied statements within the context.
- Each statement should be independent and clearly structured.
- Clearly state how each constraint impacts potential solutions based on the scenario.
###Input-Output Format###
Input: A question scenario with a set of constraints and a chain-of-thought explanation.
Output: A list of statements extracted from the given constraints and reasoning.
###Examples###
{few_shot_example}
'''

cot_parsing_statement_prompt_budget = '''###Instruction###
You are an expert in logical reasoning and structural analysis.
Your task is to identify and extract all distinct statements from the given question conditions and chain-of-thought (CoT) content.
- Extract explicitly stated and logically implied statements within the context.
- Each statement should be independent and clearly structured.
- Clearly state how each constraint impacts potential solutions based on the scenario.
###Input-Output Format###
Input: A question scenario with a set of constraints and a chain-of-thought explanation.
Output: A list of statements , with at least {budget} elements, extracted from the given constraints and reasoning.
###Examples###
{few_shot_example}
'''
cot_parsing_evidence_prompt = '''###Instruction###
You are an expert in logical analysis and evidence validation.
Your task is to identify and extract specific supporting evidence for each derived statement from the given problem conditions.
- Locate precise textual or logical evidence that directly supports each statement.
- Ensure the evidence is explicitly stated in the problem conditions or logically inferred.
- Maintain clarity, accuracy, and relevance in evidence selection.
###Examples###
{few_shot_example}
'''
cot_parsing_evidence_prompt_withoutfewshot='''###Instruction###
You are an expert in logical analysis and evidence validation.
Your task is to identify and extract specific supporting evidence for each derived statement from the given problem conditions.
- Locate precise textual or logical evidence that directly supports each statement.
- Ensure the evidence is explicitly stated in the problem conditions or logically inferred.
- Maintain clarity, accuracy, and relevance in evidence selection.
'''
cot_parsing_verification_prompt = '''###Instruction###
You are an expert in logical reasoning and verification.
Your task is to verify the logical correctness of each derived statement based on evidence from the problem context.
- Assess whether each statement logically follows from the provided evidence.
- Clearly indicate valid statements and invalid statements, with a brief justification for each.
- Do not introduce new assumptions—base verification strictly on the provided evidence.
###Examples###
{few_shot_example}
'''

cot_parsing_verification_prompt_withoutfewshot = '''###Instruction###
You are an expert in logical reasoning and verification.
Your task is to verify the logical correctness of each derived statement based on evidence from the problem context.
- Assess whether each statement logically follows from the provided evidence.
- Clearly indicate valid statements and invalid statements, with a brief justification for each.
- Do not introduce new assumptions—base verification strictly on the provided evidence.
'''
def is_valid_json(s):
    try:
        parsed = json.loads(s, strict=False)
        if isinstance(parsed, (list)):
            return True
        else:
            return False
    except ValueError:
        return False

seed_everything(42)
with open(test_file, "r", encoding="utf-8") as file:
    data_x = json.load(file)
# Process each question
for d in data_x:
    query = d["question"]
    query_embedding = emb_model.encode([query])[0]

    # Compute similarity
    similarity_scores = cosine_similarity([query_embedding], candidates_embedding)[0]
    scores, indices = torch.topk(torch.tensor(similarity_scores), k=5)


    demo_question = ""
    demo_cot_statement=""
    demo_cot_evidence = ""
    demo_cot_verify = ""
    i=0

    for idx in indices.tolist():
        if data[idx]["question"] != query:
            statements = [entry["statement"] for entry in data[idx]["cot_parsing"]]
            answer_f=data[idx]["answer"]
            cot=data[idx]["cot"]
            cot_p=data[idx]["cot_parsing"]
            demo_cot_statement += f'Example {i}:\n Query: {query}\n The answer is {answer_f}\n Chain-of-Thought analysis: {cot} Output: {json.dumps(statements)}\n'
            demo_question += f'Example {i}:\nInput: {data[idx]["question"]} \n Output: {data[idx]["question_parsing"]}\n'
            i+=1
    j=0
    for idx in indices.tolist():
        if data[idx]["question"] != query:
            statements = [entry["statement"] for entry in data[idx]["cot_parsing"]]
            answer_f=data[idx]["answer"]
            cot=data[idx]["cot"]
            cot_p=data[idx]["cot_parsing"]
            for c in cot_p:
                demo_cot_evidence += f'Example {i}:\n Query: {query}\n The answer is {answer_f}\n Chain-of-Thought analysis: {cot} Statement: {c["statement"]} Output: {c["evidence"]}\n'
                j += 1
        if j==2:
            break
    true_found = False
    false_found = False
    j = 0

    for idx in indices.tolist():
        # if data[idx]["question"] != query:
        statements = [entry["statement"] for entry in data[idx]["cot_parsing"]]
        answer_f = data[idx]["answer"]
        cot = data[idx]["cot"]
        cot_p = data[idx]["cot_parsing"]

        for c in cot_p:
            if c["evidence"] and not true_found:
                true_found = True
            elif not c["evidence"] and not false_found:
                false_found = True
            else:
                continue

            demo_cot_verify += (
                f'Example {i}:\n Query: {query}\n The answer is {answer_f}\n'
                f' Chain-of-Thought analysis: {cot} Statement: {c["statement"]} '
                f'Evidence: {c["evidence"]} Output: {c["Verification"]}\n'
            )
            j += 1
            if true_found and false_found:
                break
        if j == 2:
            break

    # Generate question parsing
    template = get_template(template_type, qptokenizer, default_system=question_parsing_prompt.format(few_shot_example=demo_question))
    question_parsing_q = query
    question_parsing_answer, _ = inference(qpmodel, template, question_parsing_q)
    import re
    try:
        d["question_parsing"] =eval(question_parsing_answer)
    except Exception as e:
        question_parsing_answer=re.sub(r'(?<=\w)"(?=\w)|(?<=:)"(?=\w)|(?<=\s)"(?=\w)|(?<=\w)"(?=\s)|(?<=\w)"(?=,)', '\\"', question_parsing_answer)
        d["question_parsing"] =eval(question_parsing_answer)



    template = get_template(template_type, cptokenizer, default_system=cot_parsing_statement_prompt.format(few_shot_example=demo_cot_statement))
    cot_statement_parsing_q = f'{query}\n The answer is {d["answer"]}\n Chain-of-Thought analysis: {d["cot"]} '
    cot_statement_answer, _ = inference(cpmodel, template, cot_statement_parsing_q)
    import re
    try:
        statements_g =eval(cot_statement_answer)
    except Exception as e:
        statements_answer=re.sub(r'(?<=\w)"(?=\w)|(?<=:)"(?=\w)|(?<=\s)"(?=\w)|(?<=\w)"(?=\s)|(?<=\w)"(?=,)', '\\"', cot_statement_answer)
        statements_g =eval(statements_answer)
    if len(statements_g)<2:
        template = get_template(template_type, cptokenizer,
                                default_system=cot_parsing_statement_prompt_budget.format(budget="two",few_shot_example=demo_cot_statement))
        cot_statement_parsing_q = f'{query}\n The answer is {d["answer"]}\n Chain-of-Thought analysis: {d["cot"]} '
        cot_statement_answer, _ = inference(cpmodel, template, cot_statement_parsing_q)
        import re
        try:
            statements_g = eval(cot_statement_answer)
        except Exception as e:
            statements_answer = re.sub(r'(?<=\w)"(?=\w)|(?<=:)"(?=\w)|(?<=\s)"(?=\w)|(?<=\w)"(?=\s)|(?<=\w)"(?=,)',
                                       '\\"', cot_statement_answer)
            statements_g = eval(statements_answer)
    print(type(statements_g))
    cot_parsing=[]
    for s in statements_g:
        # Generate statement parsing
        try:
            cvtokenizer.model_max_length = 8192

            cvmodel.generation_config.max_new_tokens = 1024
            cvmodel.generation_config.temperature = 0.0
            cvmodel.generation_config.do_sample = False
            cvmodel.generation_config.top_p = 0.97

            # print(cvmodel)
            cvmodel.generation_config.max_length = 8192
            cvmodel.config.seq_length = 8192
            print(cvmodel.generation_config)
            template = get_template(template_type, cvtokenizer,
                                    default_system=cot_parsing_evidence_prompt.format(few_shot_example=demo_cot_evidence))
            cot_evidence_parsing_q = f'Query:{query}\n The answer is {d["answer"]}\n Chain-of-Thought analysis: {d["cot"]} Statement: {s}'
            cot_evidence_answer, _ = inference(cvmodel, template,cot_evidence_parsing_q)
        except Exception as e:
            template = get_template(template_type, cvtokenizer,
                                    default_system=cot_parsing_evidence_prompt_withoutfewshot)
            cot_evidence_parsing_q = f'Query:{query}\n The answer is {d["answer"]}\n Chain-of-Thought analysis: {d["cot"]} Statement: {s}'
            cot_evidence_answer, _ = inference(cvmodel, template,cot_evidence_parsing_q)
        print(cot_evidence_answer)
        try:
            template = get_template(template_type, cvtokenizer,
                                    default_system=cot_parsing_verification_prompt.format(few_shot_example=demo_cot_verify))
            cot_verify_parsing_q = f'Query:{query}\n The answer is {d["answer"]}\n Chain-of-Thought analysis: {d["cot"]} Statement: {s} Evidence: {str(cot_evidence_answer)}'
            cot_verify_answer, _ = inference(cvmodel, template,cot_verify_parsing_q)
        except Exception as e:
            template = get_template(template_type, cvtokenizer,
                                    default_system=cot_parsing_verification_prompt_withoutfewshot)
            cot_verify_parsing_q = f'Query:{query}\n The answer is {d["answer"]}\n Chain-of-Thought analysis: {d["cot"]} Statement: {s} Evidence: {str(cot_evidence_answer)}'
            cot_verify_answer, _ = inference(cvmodel, template,cot_verify_parsing_q)
        cot_parsing.append({
                "statement":str(s),
                "evidence": str(cot_evidence_answer),
                "Verification": str(cot_verify_answer).lower()
        })

    d["cot_parsing"] = cot_parsing

with open("results.json", "w", encoding="utf-8") as outfile:
    json.dump(data_x, outfile, indent=4)
