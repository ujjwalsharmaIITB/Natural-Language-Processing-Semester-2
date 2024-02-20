
import neural_network_class

from flask import Flask, request, jsonify, render_template

app = Flask(__name__)



@app.get("/")
def returnHomePage():
    return render_template("index.html")


@app.get("/model_5/<example>")
def return_pred_5(example):
    example = example.strip()
    example = [ x for x in example]
    return jsonify({
        'prediction' : neural_network_class.predict_output_5(example)
    })


@app.get("/model_4/<example>")
def return_pred_4(example):
    example = example.strip()
    example = [ x for x in example]
    return jsonify({
        'prediction' : neural_network_class.predict_output_4(example)
    })


@app.get("/model_3/<example>")
def return_pred_3(example):
    example = example.strip()
    example = [ x for x in example]
    return jsonify({
        'prediction' : neural_network_class.predict_output_3(example)
    })


@app.get("/model_2/<example>")
def return_pred_2(example):
    example = example.strip()
    example = [ x for x in example]
    return jsonify({
        'prediction' : neural_network_class.predict_output_2(example)
    })
