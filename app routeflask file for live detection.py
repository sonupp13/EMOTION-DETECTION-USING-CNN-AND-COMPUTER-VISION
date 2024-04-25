from flask import Flask,render_template
import subprocess

app = Flask(__name__,static_folder='static')

@app.route('/')
def index():
    return render_template("live.html")

@app.route('/run_script', methods=['POST'])
def run_script():
    #run your python script here
    subprocess.run(['python','detect.py'])     #'python' syntax
    return render_template("live.html")

if(__name__ == '__main__'):
    app.run(debug=True)