import json
import pandas as pd
import matplotlib.pyplot as plt

def load_data(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def get_scores(query_idx, answer_idx, ratings):
    return {
        "completeness": ratings.get(f"completeness_{query_idx}_{answer_idx}", None),
        "factuality": ratings.get(f"factuality_{query_idx}_{answer_idx}", None),
        "usefulness": ratings.get(f"usefulness_{query_idx}_{answer_idx}", None)
    }

def process_data(filepath):    
    
    with open(filepath, 'r') as file:
        data= json.load(file)


    results = {"query": [], "intent_category": [], "answer_1": [], "answer_2": [], "answer_3": [], "completeness_1": [], "factuality_1": [], "usefulness_1": [], "completeness_2": [], "factuality_2": [], "usefulness_2": [], "completeness_3": [], "factuality_3": [], "usefulness_3": [] }
    
    for query_idx, query_data in enumerate (data["answersWithText"]):
        
        query = query_data["query"]
        answers = query_data["answers"]
        results["query"].append(query)
        if len(answers) == 3:

            results["intent_category"].append("faq")
        else:
            results["intent_category"].append("out_of_scope")

        # Process each answer
        for i, answer in enumerate(answers, 1):
            answer_idx = answer["answerIndex"]
            scores = get_scores(query_idx, answer_idx, data["answers"])

            # Append the answer and its scores to the results
            results[f"answer_{i}"].append(answer["answerText"])
            results[f"completeness_{i}"].append(scores["completeness"])
            results[f"factuality_{i}"].append(scores["factuality"])
            results[f"usefulness_{i}"].append(scores["usefulness"])

        # Handle the case where there are less than 3 answers (fill with None)
        for i in range(len(answers) + 1, 4):
            results[f"answer_{i}"].append(None)
            results[f"completeness_{i}"].append(None)
            results[f"factuality_{i}"].append(None)
            results[f"usefulness_{i}"].append(None)

    return pd.DataFrame(results)

# Function to create side-by-side box plots for comparison
def create_comparative_box_plot(data1, data2):
    df1 = pd.DataFrame(data1)
    df2 = pd.DataFrame(data2)

    # Combine both datasets and group them
    combined_df = pd.concat([df1, df2])
    
    print (combined_df)
    # List of unique broad intents
    intent_categories = combined_df['intent_category'].unique()
    print (intent_categories)
    
    # Plot side-by-side box plots for each broad intent
    plt.figure(figsize=(12, 6))
    
    for i, intent in enumerate(intent_categories):
        plt.subplot(1, len(intent_categories), i+1)
        
        # Extract data for the specific intent
        intent_data = combined_df[combined_df['intent_category'] == intent]
        
        # Group data by bot for that specific intent
        bot_groups = [intent_data[intent_data['type'] == bot]['avg_relevance_score'] 
                      for bot in ['Extractive', 'Generative']]
        
        box_colors = ['skyblue', 'orange']
        box = plt.boxplot(bot_groups, patch_artist=True, labels=['Extractive', 'Generative'])

        for patch, color in zip(box['boxes'], box_colors):
            patch.set_facecolor(color)

        plt.title(intent)
        plt.ylim(0, 1)  # Assuming relevance score ranges between 0 and 1
    
        if i == 0:
            plt.ylabel('Answer Relevance')
    # General plot settings
    plt.savefig("./box_plots.png")
    plt.close()

import os
dir = "../clean_answers"

reports = []
for file in os.listdir (dir):
    filepath = os.path.join(dir, file)

    df = process_data(filepath=filepath)
    reports.append (df)

print (len(reports))
overall_report = pd.concat(reports, axis=0)
overall_report.to_csv("./results_reports.csv", index=False)