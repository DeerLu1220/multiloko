import json
import os
from vllm import LLM
from vllm.sampling_params import SamplingParams
import textwrap

MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
# "meta-llama/Meta-Llama-3-8B-Instruct"
# "mistralai/Mistral-7B-Instruct-v0.3"
# MODEL = "sapienzanlp/Minerva-7B-instruct-v1.0"
# MODEL = "sapienzanlp/Minerva-3B-base-v1.0"

sampling_params = SamplingParams(max_tokens=1024)

DATA_DIR = "multiloko_eval/dev.jsonl"

def data_loader(input_file):
    '''read data from input_file, the data inside the file is a multi-line jsonl file, an example line as following:
    {"text": "Storia 2011 ...", "question": "Quando è stato lanciato il TG Norba 24?", "target": "TG ...", "id": "004_tg_norba_24.txt", "targets": ["2010-10-25", "venticinque ottobre duemiladieci", "25 ottobre 2010", "25/10/2010"], "output_type": "una data"}
    save "id", "question", "targets" and "output_type" to a dict, use "id" as the key
    return a dict, the key is "id" and the value is a dict with "question", "targets" and "output_type"
    '''

    data_dict = {}
    with open(input_file, 'r', encoding='utf-8') as file:
        for line in file:
            entry = json.loads(line.strip())
            data_dict[entry['id']] = {
                'question': entry['question'],
                'targets': entry['targets'],
                'output_type': entry['output_type']
            }
    return data_dict

def build_prompt():
    '''build the prompt for the model, the prompt is a string
    the prompt is in Italian, and it is a question with a specific output type
    '''
    fewshot_examples = [
        {"question": "Quando si festeggia Ferragosto?", "answer": "15 agosto", "output_type": "data"},
        {"question": "Quando ha vinto l'Italia per l'ultima volta i mondiali di calcio?", "answer": "2006", "output_type": "data"},
        {"question": "Quale moneta aveva l'Italia prima dell'euro?", "answer": "Lira", "output_type": "nome"},
        {"question": "Come si chiama il primo Presidente della Repubblica italiana?", "answer": "Enrico De Nicola", "output_type": "nome"},
        {"question": "Quando è la Festa della Repubblica in Italia?", "answer": "2 giugno", "output_type": "data"},
        ]
    prompt_parts = []

    for example in fewshot_examples:
        part = textwrap.dedent(f"""
        Rispondi alla seguente domanda in modo chiaro e conciso.
        D: {example["question"]} Produci solo risposte del seguente tipo: {example["output_type"]}.
        R: {example["answer"]}
        """).strip()
        prompt_parts.append(part)
    return prompt_parts

def generate_response(data_dict, output_file, llm):
    '''generate response for the entry, the entry is a dict with "question", "targets" and "output_type"
    return a dict with "id", "question", "response", "targets" and "output_type"
    '''
    with open(output_file, 'w', encoding='utf-8') as file:
        for entry_id, entry in data_dict.items():
            question = entry['question']
            output_type = entry['output_type']
            prompt_parts = build_prompt()
            final_part = textwrap.dedent(f"""
            D: {question} Produci solo risposte del seguente tipo: {output_type}.
            R:""").strip()
            prompt_parts.append(final_part)
            massages = "\n\n".join(prompt_parts)
            # massages = f"Rispondi alla seguente domanda in modo chiaro e conciso: {question} Produci solo risposte del seguente tipo: {output_type}."
            response = llm.generate(massages)
            response_text = response[0].outputs[0].text

            # Save the response to the file
            output_entry = {
                "id": entry_id,
                "question": question,
                "response": response_text,
                "targets": entry['targets'],
                "output_type": output_type,
                "massages": massages,
            }
            file.write(json.dumps(output_entry, ensure_ascii=False) + '\n')

def prepare_eval(path, output_eval_path):
    '''
    read the output file and prepare the evaluation data
    the output file is a jsonl file
    an example of the output file:
    {"id": "1", "question": "What is the capital of France?", "response": "Paris", "targets": ["Paris"], "output_type": "text", "massages": "Rispondi alla seguente domanda in modo chiaro e conciso: What is the capital of France? Produci solo risposte del seguente tipo: text."}
    write the eval data also into a new jsonl file
    an example of the eval data:
    {"language": "language", "id" : "id", "prediction": "prediction"}
    '''
    eval_data = []
    with open(path, 'r', encoding='utf-8') as file:
        for line in file:
            entry = json.loads(line.strip())
            eval_entry = {
                "language": "italian",  # Assuming the language is Italian based on the context
                "id": entry["id"],
                "prediction": entry["response"]
            }
            eval_data.append(eval_entry)
    
    # Write the eval data to a new jsonl file
    with open(output_eval_path, 'w', encoding='utf-8') as eval_file:
        for eval_entry in eval_data:
            eval_file.write(json.dumps(eval_entry, ensure_ascii=False) + '\n')

if __name__ == "__main__":
    model_f = MODEL.split("/")[0]
    model_name = MODEL.split("/")[1]
    output_file = os.path.join(
        "model_output", 
        model_f, 
        model_name,
        "dev_5shot.jsonl"
    )
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_eval_path = os.path.join(
        "model_output", 
        model_f, 
        model_name,
        "evaluate_5shot.jsonl"
    )


    # llm = LLM(model=MODEL,device="cuda")
    model = LLM(
        model=MODEL,
        tensor_parallel_size=1,  # Adjust based on GPU count
        gpu_memory_utilization=0.8,  # Lower if OOM errors
        max_model_len=4096  # Can be adjusted based on needs8192
    )

    data_dict = data_loader(DATA_DIR)
    # Generate responses and save to output file
    generate_response(data_dict, output_file, model)
    
    # Prepare evaluation data and save to output eval path
    prepare_eval(output_file, output_eval_path)
    
    print(f"Responses saved to {output_file}")
    print(f"Evaluation data saved to {output_eval_path}")