from flask import Flask, render_template, request, redirect
import ML
import pandas as pd
import os
import glob

app = Flask(__name__)

#Routing or mapping, URL to python function
@app.route('/', methods=['GET', 'POST']) #homepage of website
def index():
    if request.method == "POST":
        url_data = request.form["url"]
        plot = ML.plot_whole_dataframe(ML.dataframe_to_list('Al-Jazeera'),ML.dataframe_to_list('BBC'),ML.dataframe_to_list('Breitbart'),ML.dataframe_to_list('CNN'),ML.dataframe_to_list('FOX'),ML.dataframe_to_list('HuffPo'),ML.dataframe_to_list('NBCNews'),ML.dataframe_to_list('NPR'),ML.dataframe_to_list('NYT'),ML.dataframe_to_list('WashPo'), url_data)
        return render_template("results.html")
    return render_template("index.html")

@app.route('/about')
def about():
    return render_template("about1.html")

@app.route('/mlfindings')
def ml():
    return render_template("mlfindings.html")

@app.route('/resources')
def resources():
    return render_template ("resources.html")



#@app.route('/results')
#def resources():


if __name__ == "__main__":
    app.run(debug=False, use_reloader=False)
