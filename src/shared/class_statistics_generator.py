import json

def add_json_values(json1, json2):
    result = {}

    for key, value1 in json1.items():
        if key in json2:
            value2 = json2[key]

            if isinstance(value1, dict) and isinstance(value2, dict):
                result[key] = add_json_values(value1, value2)
            else:
                result[key] = value1 + value2
        else:
            result[key] = value1

    for key, value2 in json2.items():
        if key not in json1:
            result[key] = value2

    return result

def main():
    # Replace 'file1.json' and 'file2.json' with your actual file paths
    with open('./data/protected/counts_np.json', 'r', encoding='utf-8') as file1, open('./data/protected/counts_p.json', 'r',encoding='utf-8') as file2:
        json1_data = json.load(file1)
        json2_data = json.load(file2)

    result_json = add_json_values(json1_data, json2_data)

    sorted_modifiers = sorted(result_json['modifiers'].items(), key=lambda x: x[1], reverse=True)
    sorted_facts = sorted(result_json['facts'].items(), key=lambda x: x[1], reverse=True)

    result_json['modifiers'] = dict(sorted_modifiers)
    result_json['facts'] = dict(sorted_facts)
    # Replace 'output.json' with your desired output file path
    with open('./data/protected/total_counts.json', 'w',encoding='utf-8') as output_file:
        json.dump(result_json, output_file,ensure_ascii=False, indent=2)

    print("Output JSON file created successfully.")

if __name__ == "__main__":
    main()