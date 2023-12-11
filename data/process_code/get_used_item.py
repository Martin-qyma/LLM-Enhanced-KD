import csv
import json

def filter_json_data(csv_file_path, json_file_path, output_file_path):
    # Initialize a set to store the unique items from the CSV file
    items_set = set()
    
    # Read the CSV file and extract the items from the "item" column
    with open(csv_file_path, mode='r', encoding='utf-8') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            items_set.add(row['item'])

    # Initialize a dictionary to store the filtered JSON data
    filtered_json_data = {}
    
    # Read the JSON file line by line and filter the data
    with open(json_file_path, mode='r', encoding='utf-8') as file:
        for line in file:
            try:
                line_data = json.loads(line.strip())
                for key, value in line_data.items():
                    if key in items_set:
                        filtered_json_data[key] = value
            except json.JSONDecodeError:
                pass
    
    # Write the filtered JSON data to the output file
    with open(output_file_path, mode='w', encoding='utf-8') as file:
        for key, value in filtered_json_data.items():
            json.dump({key: value}, file, ensure_ascii=False)
            file.write('\n')
    print(f"Filtered JSON data has been written to {output_file_path}.")

if __name__ == "__main__":
    csv_file_path = './data/total.csv'
    json_file_path = './data/mapped_item_content.json'
    output_file_path = './data/used_item.json'
    
    filter_json_data(csv_file_path, json_file_path, output_file_path)