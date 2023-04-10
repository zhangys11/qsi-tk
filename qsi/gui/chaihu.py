'''
This modulde is used to run the GUI of the CLA package.

NOTE: The Flask dev server is not designed to be particularly secure, stable, or efficient. 
By default it runs on localhost (127.0.0.1), change it to app.run(host="0.0.0.0") to run on all your machine's IP addresses.
0.0.0.0 is a special value that you can't use in the browser directly, you'll need to navigate to the actual IP address of the machine on the network. You may also need to adjust your firewall to allow external access to the port.
The Flask quickstart docs explain this in the "Externally Visible Server" section:
    If you run the server you will notice that the server is only accessible from your own computer, not from any other in the network. This is the default because in debugging mode a user of the application can execute arbitrary Python code on your computer.
    If you have the debugger disabled or trust the users on your network, you can make the server publicly available simply by adding --host=0.0.0.0 to the command line.
'''


from threading import Timer
import webbrowser
import os
import sys
import uuid
from flask import Flask, render_template, request
from flaskwebgui import FlaskUI

if __package__:
    from . import chaihu_models
else:
    ROOT_DIR = os.path.dirname (os.path.dirname(__file__))
    if ROOT_DIR not in sys.path:
        sys.path.append(ROOT_DIR)
    import chaihu_models

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # limit to 5MB


def analyze(csv, save_local=False):

    # store html result into a local html file

    if save_local:

        fn = os.path.dirname(os.path.realpath(__file__)) + \
            "/" + str(uuid.uuid4()) + ".html"

        with open(fn, 'w') as f:
            f.write(chaihu_models.analyze_file(csv))

        # fn is the local save path

    return chaihu_models.analyze_file(csv)  # return the html content

# routes
def analyze_probs(csv, save_local=False):

    # store html result into a local html file

    if save_local:

        fn = os.path.dirname(os.path.realpath(__file__)) + \
            "/" + str(uuid.uuid4()) + ".html"

        with open(fn, 'w') as f:
            f.write(chaihu_models.analyze_file(csv))

        # fn is the local save path

    return chaihu_models.analyze_probs(csv)

@app.route("/", methods=['GET', 'POST'])
def index():
    return render_template("home.html")


@app.route("/about")
def about_page():
    return "Created by Dr. Zhang (oo@zju.edu.cn)"


@app.route("/submit", methods=['GET', 'POST'])
def run_cla():
    if request.method == 'POST':

        f = request.files['dataFile']
        csv = os.path.dirname(os.path.realpath(
            __file__)) + "/" + str(uuid.uuid4()) + ".csv"
        f.save(csv)
        r = analyze(csv)
        probs = analyze_probs(csv)

    # render_template("home.html", use_sample = use_sample, d = d, nobs = n, cla_result = r)
    return {'message': 'success', 'html': r, 'probs': probs}


def open_browser():
    webbrowser.open_new('http://localhost:5005/')


if __name__ == '__main__':
    # use netstat -ano|findstr 5005 to check port use
    Timer(3, open_browser).start()
    app.run(host="0.0.0.0", port=5005, debug=False)
    # FlaskUI(app=app, server="flask", port=5005).run()
