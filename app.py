from flask import Flask, render_template, request

from lang_detection import test_language


app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/detect_language', methods=['POST'])
def detect_language():
    paragraph = request.form['paragraph']

    with open('test.txt', 'w') as file:
        file.write(paragraph)

    language = test_language('test.txt')
    return render_template('result.html', language=language)


if __name__ == "__main__":
    app.run()
