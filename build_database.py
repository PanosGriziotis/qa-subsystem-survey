import json
import pandas as pd
import os
import argparse

import json
import argparse
import pandas as pd

output_file_dir = "static/js"

parser = argparse.ArgumentParser()
parser.add_argument("--rag")
parser.add_argument("--ex")
parser.add_argument("--dataset")
args = parser.parse_args()

with open(args.rag, "r") as fp:
    rag_data = json.load(fp)
with open(args.ex, "r") as fp:
    ex_data = json.load(fp)

df = pd.read_csv(args.dataset)

query_intent_pairs = zip(df["query"], df["intent_name"])
out_of_scope_queries = []
faq_queries = []

# Separate queries into out-of-scope and FAQ
for query, intent in query_intent_pairs:
    
    if "out_of_scope" in intent:
        out_of_scope_queries.append(query)
    else:
        faq_queries.append(query)

# Initialize dictionaries for out-of-scope and FAQ queries
out_of_scope_dicts = []
faq_dicts = []

non_answered_queries = []

# Create dictionaries for out-of-scope queries
for query in out_of_scope_queries:
    # Initialize the dictionary with None values for extractive and RAG answers
    query_dict = {"query": query, "answers": [None, None]}  

    # Check if any answer matches the non-answer string
    exclude_query = False

    # Find the extractive answer for out-of-scope queries
    for d in ex_data:
        if d["query"] == query:
            answer = d["answers"][0]["answer"]
            if d["answers"][0]["meta"]["context_relevance"] < 0.17:
                exclude_query = True
                break  # Stop checking further answers for this query
            else:
                query_dict["answers"][0] = answer  # index 0 for extractive answer

    # Skip adding this query if it has a non-answer
    if exclude_query:
        continue

    # Find the RAG answer for out-of-scope queries
    for d in rag_data:
        if d["query"] == query:
            answer = d["answers"][0]["answer"]
            if d["answers"][0]["meta"]["context_relevance"] < 0.17:
                exclude_query = True
                break  # Stop checking further answers for this query
            else:
                query_dict["answers"][1] = answer  # index 1 for RAG answer

    # Skip adding this query if it has a non-answer
    if exclude_query:
        continue

    # If no non-answer found, add to the out_of_scope_dicts
    out_of_scope_dicts.append(query_dict)

# Create dictionaries for FAQ queries
for query in faq_queries:
    # Initialize the dictionary with None values for extractive, RAG, and Theano answers
    query_dict = {"query": query, "answers": [None, None, None]}  

    # Check if any answer matches the non-answer string
    exclude_query = False

    # Find the extractive answer for FAQ queries
    for d in ex_data:
        if d["query"] == query:
            answer = d["answers"][0]["answer"]
            if d["answers"][0]["meta"]["context_relevance"] < 0.17:
                exclude_query = True
                break  # Stop checking further answers for this query
            else:
                query_dict["answers"][0] = answer  # index 0 for extractive answer

    # Skip adding this query if it has a non-answer
    if exclude_query:
        continue

    # Find the RAG answer for FAQ queries
    for d in rag_data:
        if d["query"] == query:
            answer = d["answers"][0]["answer"]
            if d["answers"][0]["meta"]["context_relevance"] < 0.17:
                exclude_query = True
                break  # Stop checking further answers for this query
            else:
                query_dict["answers"][1] = answer  # index 1 for RAG answer

    # Skip adding this query if it has a non-answer
    if exclude_query:
        continue

    # Find the Theano answer for FAQ queries
    theano_answer = df[df["query"] == query]["response"].values
    if len(theano_answer) > 0:
        query_dict["answers"][2] = theano_answer[0]  # index 2 for Theano answer

    # If no non-answer found, add to the faq_dicts
    faq_dicts.append(query_dict)
    
# Save the dictionaries to files (optional)
with open(f"{output_file_dir}/test_out_of_scope.json", "w") as outfile:
    json.dump(out_of_scope_dicts, outfile, ensure_ascii=False, indent=4)

with open(f"{output_file_dir}/test_faq.json", "w") as outfile:
    json.dump(faq_dicts, outfile, ensure_ascii=False, indent=4)