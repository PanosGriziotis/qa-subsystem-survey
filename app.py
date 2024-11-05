from flask import Flask, render_template, request, jsonify
import json
import os
from datetime import datetime

app = Flask(__name__, static_url_path='/static')

@app.route("/")
def index():

    # serve the main html document
    return render_template("index.html")


@app.route('/postmethod', methods=['POST'])
def postmethod():
    print("Incoming data from client... ")
    # Receive post request from client. 
    # Message data in json format
    data = request.get_json()
    print(data)
    
    # Store data to local fodler in server e.g. in './answers/'
    #filename = "./answers/*.json"
    answers_dir = "./answers"
    if not os.path.exists(answers_dir):
        os.makedirs(answers_dir)

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    filename = os.path.join(answers_dir, f"response_{timestamp}.json")

    with open(filename, 'w', encoding='utf-8') as f:
             json.dump(data, f, ensure_ascii=False, indent=4)

    # Return ack to client
    return jsonify("DONE")


if __name__ == "__main__":
    # localhost in port 1988 e.g. http://0.0.0.0:1988
    app.run(host='0.0.0.0', port=1988, debug=True)