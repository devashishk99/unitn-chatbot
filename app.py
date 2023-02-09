from flask import Flask, render_template, request, jsonify 
from chat import get_response

#initializes the flask app
app = Flask(__name__)

#redirects to home
@app.get("/")
def index_get():
    return render_template("base.html")

#redirects to predict page
@app.post("/predict")
def predict():
    text = request.get_json().get("message")
    # TODO: check if the text is valid
    response = get_response(text)
    message = {"answer": response}
    return jsonify(message)

#main function
if __name__ == "__main__":
    app.run(debug=True)