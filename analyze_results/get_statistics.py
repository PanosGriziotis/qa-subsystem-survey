import os
import json

def get_demographics_report(dir):

    total = {"age": {"under_20":0, "20_30":0, "30_40":0, "40_plus":0}, "gender": {"male" :0, "female": 0, "other": 0} } 
    
    ai_familiarity = []

    for file in os.listdir(dir):
        with open (os.path.join(dir, file),"r") as fp:
            data = json.load(fp)
        demo = data["demographics"]
        
        for key, value in demo.items():
            if key != "ai_familiarity":
                total[key][value] +=1
            else:
                ai_familiarity.append(value)
    
    avg_ai_fam = sum(ai_familiarity)/len(ai_familiarity)
    total["ai_familiarity"] = avg_ai_fam

    with open ("demo_statistics.json", 'w') as fp:
        json.dump(total, fp, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    import sys
    
    get_demographics_report(sys.argv[1])