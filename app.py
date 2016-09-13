from flask import Flask
# import csvprocessor
from coffee_name_recognizer import *
app = Flask(__name__)


@app.route("/")
def main():
    return "Welcome!"

@app.route("/getCoffeeName/<spec>")
def get_category(spec):
    return str(test_model(spec))

if __name__ == "__main__":
    app.run(host= '0.0.0.0')
