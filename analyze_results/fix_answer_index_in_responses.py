import re
from collections import defaultdict
import json
from collections import OrderedDict
import os



def update_dictionary_keys(input_dict):
    updated_dict = {}
    for key, value in input_dict.items():
        # Split the key to get the parts: "completeness", "factuality", "usefulness", and the indices
        parts = key.split('_')
        query_idx = int(parts[1])  # Extract the query index
        answer_idx = int(parts[2])  # The incorrect answer index

        # Calculate the correct answer index based on its position
        # Each answer has 3 ratings (completeness, factuality, usefulness)
        new_answer_idx = (answer_idx - 1) // 10

        # Create a new key with the corrected answer index
        new_key = f"{parts[0]}_{query_idx}_{new_answer_idx}"
        
        # Update the dictionary with the new key
        updated_dict[new_key] = value

    return updated_dict

dir = ("../answers")
new_dir = ("../updated_answers")

for file in os.listdir(dir):
    filepath = os.path.join (dir, file)
    
    with open (filepath, "r") as fp:
        data = json.load(fp)
        
        data["answers"] = update_dictionary_keys(data["answers"])
    
    with open (os.path.join(new_dir, file), "w") as fp:
        json.dump (data, fp, ensure_ascii=False, indent=4)
