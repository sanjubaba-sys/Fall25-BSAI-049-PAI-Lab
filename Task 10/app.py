from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Simple chatbot logic (rule-based)
def get_response(user_input):
    user_input = user_input.lower()

    if "admission" in user_input or "apply" in user_input:
        return "Admissions are open from June to August every year."

    elif "requirements" in user_input or "eligibility" in user_input:
        return "You need at least 50% marks in intermediate (FA/FSc/A-Levels)."

    elif "deadline" in user_input:
        return "The last date to apply is 31st August."

    elif "program" in user_input or "courses" in user_input:
        return "We offer CS, Engineering, Business, and Medical programs."

    elif "fee" in user_input:
        return "Fee varies by program. Average is 50,000 to 120,000 per semester."

    else:
        return "Sorry, I can only help with admission info, requirements, deadlines, and programs."

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json["message"]
    response = get_response(user_message)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)