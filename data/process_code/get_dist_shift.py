import json
import csv

csv_file_path = "../total.csv"
# Initialize a set to store the unique items from the CSV file
items_set = set()
# Read the CSV file and extract the items from the "item" column
with open(csv_file_path, mode="r", encoding="utf-8") as file:
    csv_reader = csv.DictReader(file)
    for row in csv_reader:
        items_set.add(row["item"])
print("items_set loaded")

history_items = set()
genre_fiction_items = set()
used_file_path = "../used_item.json"
# Read the used_item.json file line by line and extract the required information
with open(used_file_path, "r") as file:
    for line in file:
        # Parse each line as a JSON object
        data = json.loads(line.strip())
        values_list = list(data.values())
        # Create a new dictionary with indices as keys
        new_dict = values_list[0]
        # Get history items
        if "History" in new_dict["category"]:
            history_items.add(list(data.keys())[0])
        # Get genre_fiction items
        if "Genre Fiction" in new_dict["category"]:
            genre_fiction_items.add(list(data.keys())[0])

history_output_file_path = (
    "../distribution_shift/cold/history.csv"
)
genre_fiction_output_file_path = (
    "../distribution_shift/warm/genre_fiction.csv"
)
with open(csv_file_path, mode="r", encoding="utf-8") as input_file, open(
    history_output_file_path, "w", newline=""
) as output_file1, open(
    genre_fiction_output_file_path, "w", newline=""
) as output_file2:
    writer1 = csv.DictWriter(output_file1, fieldnames=["user", "item"])
    writer1.writeheader()
    writer2 = csv.DictWriter(output_file2, fieldnames=["user", "item"])
    writer2.writeheader()
    csv_reader = csv.DictReader(input_file)
    for row in csv_reader:
        if row["item"] in history_items:
            writer1.writerow(row)
        if row["item"] in genre_fiction_items:
            writer2.writerow(row)
print("Processing completed.")
print(f"History item is saved to {history_output_file_path}")
print(f"Genre Fiction item is saved to {genre_fiction_output_file_path}")
