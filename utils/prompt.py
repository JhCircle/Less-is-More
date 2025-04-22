#Prompt induction via Reversal of Thought
question_parsing_prompt = '''
###Instruction###
Extract the constraints and key details from a problem description, ignoring any specific questions or answer choices. 
Focus on the rules or conditions given that are necessary to solve the problem, and extract these in a clear, descriptive list.

###Input-Output Format###\n
Input: A textual problem or scenario containing multiple rules or conditions within a specific context.\n
Output: An ordered list of extracted conditions and essential details needed to address the problem stated in the input. Each extracted condition should be clearly and concisely formatted, capturing only the facts necessary for determining the problem's solution.
###Examples###
{few_shot_example}
'''

cot_synthesis = '''
### Instruction ###
You are an expert in logical and structural reasoning.
Your task is to provide short, clear reasoning for each statement. 
Each statement should logically follow from the previous one and be supported by the provided information. Keep your reasoning concise, using fewer than {budget} tokens.

### Examples ###
{few_shot_cot}
'''

cot_parsing = '''
### Instruction ###
The goal is to systematically dissect the problem using logical reasoning, providing detailed evidence for each derived statement, and verifying the correctness of these statements against the given problem conditions.
- For each condition or rule, analyze its implications step by step.
- Provide verification for each logical statement using evidence from the given problem.
- Ensure that each step follows logically from the previous, with clear conclusions and validations.

**Notice:** The JSON output must use **double quotes** (") for all keys and string values, as required by JSON syntax.

### Examples ###
{few_shot_cot}
'''