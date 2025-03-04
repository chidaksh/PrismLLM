import json
from tqdm import tqdm

def load_json_data(filepath):
    with open(filepath, 'r') as file:
        data = json.load(file)
    return data

def save_list_to_file(response, filename):
    with open(filename, 'w') as file:
        json.dump(response, file, indent=4)

mmlu_data = load_json_data("./mmlu_answer_mapping.json")
llama_data = load_json_data("./llama_4000.json")
mistral_data = load_json_data("./mistral_4000.json")
gpt_data = load_json_data("./test_responses.json")

leng = len(llama_data)
responses = []
results = []
response = {}

mmlu_dict = {int(item['query_id']): item for item in mmlu_data}
mistral_dict = {int(item['query_id']): item for item in mistral_data}
llama_dict = {int(item['query_id']): item for item in llama_data}
gpt_dict = {int(item['id']): item for item in gpt_data['responses']}
sorted_query_ids = sorted(mmlu_dict.keys())

for key in tqdm(sorted_query_ids, desc="Processing queries", unit="query"):
    response = {
        "query_id": key,
        "query": mmlu_dict[key]["question"],
        "choices": mmlu_dict[key]["choices"],
        "correct_answer": mmlu_dict[key]["mmlu_answer"],
        "mistral_response": mistral_dict.get(key, {}).get("full_response", "N/A"),
        "llama_response": llama_dict.get(key, {}).get("full_response", "N/A"),
        "label": gpt_dict.get(key, {}).get("llm", "N/A"),
    }
    results.append(response)

for output in results:
    responses.append(output)
response_object = {'responses': responses}
save_list_to_file(response_object, "./rearranged_resp.json")