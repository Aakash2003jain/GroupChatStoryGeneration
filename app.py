from flask import Flask, render_template, request
import tensorflow as tf

from transformers import  PegasusTokenizer
import torch

app = Flask(__name__)
model_dir = r"..\Text-summarization\pegasus-samsum-model" #Load the saved model after training
tokenizer = PegasusTokenizer.from_pretrained(r"..\Text-summarization\tokenizer") # Load the saved tokenizer after training
print("Tokenizer loaded")

device = "cuda" if torch.cuda.is_available() else "cpu"

from transformers import pipeline
gen_kwargs = {"length_penalty": 0.8, "num_beams":8, "max_length": 128}

pipe = pipeline("summarization", model=model_dir,tokenizer=tokenizer)
print("Done")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/text-summarization', methods=["POST"])
def summarize():
    if request.method == "POST":
        input_text = request.form["inputtext_"]

        # Use the summarization pipeline to generate the summary
        summary = pipe(input_text, **gen_kwargs)[0]['summary_text']

    return render_template("output.html", data={"summary": summary})

if __name__ == '__main__': # It Allows You to Execute Code When the File Runs as a Script
    app.run()
