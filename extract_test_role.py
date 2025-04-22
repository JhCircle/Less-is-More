import json

with open("demo_pool.json", "r", encoding="utf-8") as file:
    data = json.load(file)

question_parsing_prompt = '''###Instruction###
Extract the constraints and key details from a problem description, ignoring any specific questions or answer choices. 
Focus on the rules or conditions given that are necessary to solve the problem, and extract these in a clear, descriptive list.
###Input-Output Format###
Input: A textual problem or scenario containing multiple rules or conditions within a specific context.
Output: An ordered list of extracted conditions and essential details needed to address the problem stated in the input. Each extracted condition should be clearly and concisely formatted, capturing only the facts necessary for determining the problem's solution.
'''
cot_synthesis_prompt = '''###Instruction###
You are an expert in logical and structural reasoning.
Your task is to provide short, clear reasoning for each statement based on statement parsing and statement-evidence pair extraction. 
Each statement should logically follow from the previous one and be supported by the provided information. 
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
'''

cot_parsing_evidence_prompt = '''###Instruction###
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
'''

question_parsing_data = []
cot_statement_data = []
cot_verify_data=[]
cot_parsing_data=[]
# Process each item in the data
for entry in data:
    query = entry["question"]
    answer = entry["question_parsing"]
    cot = entry["cot"]
    cot_parsing_answer = entry["cot_parsing"]
    answer_f=entry["answer"]
    question_parsing = {
        "system": question_parsing_prompt,
        "query": query,
        "answer": json.dumps(answer)
    }
    question_parsing_data.append(question_parsing)

    statements = [entry["statement"] for entry in cot_parsing_answer]
    cot_parsing_statement = {
        "system": cot_parsing_statement_prompt,
        "query": f"Query: {query}\n The answer is {answer_f}\n Chain-of-Thought analysis: {cot}",
        "answer": json.dumps(statements)
    }
    print(statements)
    cot_statement_data.append(cot_parsing_statement)
    cot_parsing_data.append(cot_parsing_statement)
    for idx, entry in enumerate(cot_parsing_answer):
        statement = statements[idx]
        evidence = entry["evidence"]
        verification = entry["Verification"]

        cot_parsing_evidence = {
            "system": cot_parsing_evidence_prompt,
            "query": f"Query: {query}\n The answer is {answer_f}\n Chain-of-Thought analysis: {cot} Statement: {statement}",
            "answer": evidence
        }
        cot_verify_data.append(cot_parsing_evidence)
        cot_parsing_data.append(cot_parsing_evidence)
        cot_parsing_verification = {
            "system": cot_parsing_verification_prompt,
            "query": f"Query: {query}\n The answer is {answer_f}\n Chain-of-Thought analysis: {cot} Statement: {statement} Evidence: {evidence}",
            "answer": verification
        }
        print(cot_parsing_verification)
        cot_verify_data.append(cot_parsing_verification)
        cot_parsing_data.append(cot_parsing_verification)
# Save the processed data to new JSONL files
with open("./data/test/test_question_parsing_role.jsonl", "w", encoding="utf-8") as output_file:
    for entry in question_parsing_data:
        output_file.write(json.dumps(entry, ensure_ascii=False) + "\n")


with open("./data/test/test_cot_verify_role.jsonl", "w", encoding="utf-8") as output_file:
    for entry in cot_verify_data:
        output_file.write(json.dumps(entry, ensure_ascii=False) + "\n")
    for entry in cot_statement_data:
        output_file.write(json.dumps(entry, ensure_ascii=False) + "\n")

with open("./data/test/test_cot_parsing_role.jsonl", "w", encoding="utf-8") as output_file:
    for entry in cot_parsing_data:
        output_file.write(json.dumps(entry, ensure_ascii=False) + "\n")
