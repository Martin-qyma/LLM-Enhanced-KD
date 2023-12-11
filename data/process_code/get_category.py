import json
from collections import Counter

categories = []  # List to store categories

with open('used_item.json', 'r') as file:
    for line in file:
        # Parse the JSON data from the line
        data = json.loads(line.strip())

        # Check if the "category" key is in this line of data
        # Convert dict_values to a list
        values_list = list(data.values())

        # Create a new dictionary
        new_dict = values_list[0]
        if "category" in new_dict:
            categories.append(new_dict["category"])


'''
get the occurrence of each book's category
'''
categories_tuples = [tuple(sublist) for sublist in categories]
# Count occurrences of each category
category_counts = Counter(categories_tuples)
ranked_categories = category_counts.most_common()

for category, count in ranked_categories:
    print(f"{category}: {count}")
    

'''
get the occurrence of each category
'''
# flat_categories = [item for sublist in categories for item in sublist]

# # Count occurrences of each category
# category_counts = Counter(flat_categories)
# ranked_categories = category_counts.most_common()

# for category, count in ranked_categories:
#     print(f"{category}: {count}")