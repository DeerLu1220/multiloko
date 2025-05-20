import json
import os
from transformers import AutoProcessor, AutoModel

MODEL = "Qwen/Qwen2.5-Omni-7B"

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

def generate_response(data_dict, output_file, model, tokenizer):
    '''generate response for the entry, the entry is a dict with "question", "targets" and "output_type"
    return a dict with "id", "question", "response", "targets" and "output_type"
    '''
    with open(output_file, 'w', encoding='utf-8') as file:
        for entry_id, entry in data_dict.items():
            question = entry['question']
            output_type = entry['output_type']
            prompt = f"Rispondi alla seguente domanda in modo chiaro e conciso: {question} Produci solo risposte del seguente tipo: {output_type}."

            # Tokenize the input and generate response
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            outputs = model.generate(**inputs, max_new_tokens=1024)
            response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Save the response to the file
            output_entry = {
                "id": entry_id,
                "question": question,
                "response": response_text,
                "targets": entry['targets'],
                "output_type": output_type,
                "prompt": prompt,
            }
            file.write(json.dumps(output_entry, ensure_ascii=False) + '\n')

def prepare_eval(path, output_eval_path):
    '''
    read the output file and prepare the evaluation data
    the output file is a jsonl file
    an example of the output file:
    {"id": "1", "question": "What is the capital of France?", "response": "Paris", "targets": ["Paris"], "output_type": "text", "prompt": "Rispondi alla seguente domanda in modo chiaro e conciso: What is the capital of France? Produci solo risposte del seguente tipo: text."}
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
    output_file = os.path.join(
        "model_output", 
        MODEL.split("/")[1], 
        "dev.jsonl"
        )
    output_eval_path = os.path.join(
        "model_output", 
        MODEL.split("/")[1], 
        "dev_eval.jsonl"
        )

    # Load the model and tokenizer
    print(MODEL)
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B")
    model = AutoModel.from_pretrained("Qwen/Qwen2.5-Omni-7B").cuda()
    # Load data
    data_dict = data_loader(DATA_DIR)

    # Generate responses and save to output file
    generate_response(data_dict, output_file, model, processor.tokenizer)
    
    # Prepare evaluation data and save to output eval path
    prepare_eval(output_file, output_eval_path)
    
    print(f"Responses saved to {output_file}")
    print(f"Evaluation data saved to {output_eval_path}")
