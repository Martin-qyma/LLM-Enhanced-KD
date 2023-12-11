import json
from collections import defaultdict

# Define file paths
cold_file_path = "./data/source/metadata.json"
output_content_file_path = "./data/mapped_item_content.json"

# Initialize item_content_mapping dictionary
item_content_mapping = {}
asin_mapping = defaultdict(lambda: len(asin_mapping) + 100000000)

# Read the metadata.json file line by line and extract the required information
try:
    with open(cold_file_path, "r") as file, open(
        output_content_file_path, "w"
    ) as output_file:
        for line in file:
            try:
                # Parse each line as a JSON object
                entry = json.loads(line)
                if entry.get("description") and entry.get("category"):
                    asin = entry.get("asin", None)
                    if asin:
                        # Map "asin" to a number consistently with the former mapping
                        mapped_asin = asin_mapping[asin]

                        # Extract the required content and add "main_cat"
                        content = {
                            "title": entry.get("title", ""),
                            "description": entry.get("description", ""),
                            "brand": entry.get("brand", ""),
                            "price": entry.get("price", ""),
                            "category": entry.get("category", ""),
                            "date": entry.get("date", ""),
                            "rank": entry.get("rank", ""),
                            "main_cat": entry.get("main_cat", ""),
                        }

                        # Map the item to its content
                        item_content_mapping[mapped_asin] = content

                        # Write the mapping to the output file
                        output_file.write(json.dumps({mapped_asin: content}) + "\n")
            except json.JSONDecodeError as e:
                # Log any parsing error and the line causing it
                print(f"Error: {e}")
                print(line)
except FileNotFoundError:
    print(f"{cold_file_path} not found. Please ensure the file path is correct.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")

print(
    f"Processing completed. Mapped item content is saved to {output_content_file_path}"
)

asin_mapping_file_path = "./data/asin_mapping.json"
with open(asin_mapping_file_path, "w") as f:
    json.dump(dict(asin_mapping), f)

print(f"Processing completed. asin_mapping is saved to {asin_mapping_file_path}")
