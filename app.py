from flask import Flask, render_template, request, send_file, url_for
import os
from detect import run_inference_and_generate_report
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
OUTPUT_FOLDER = 'outputs'
REPORT_FOLDER = 'reports'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(REPORT_FOLDER, exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        img_file = request.files['image']
        if img_file:
            filename = secure_filename(img_file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            img_file.save(filepath)

            output_path, report_path = run_inference_and_generate_report(filepath)
            output_filename = os.path.basename(output_path)
            report_filename = os.path.basename(report_path)

            return render_template('index.html', 
                                   output_image=output_filename,
                                   report_file=report_filename)

    return render_template('index.html')





@app.route('/download_report/<filename>')
def download_report(filename):
    report_path = os.path.join(REPORT_FOLDER, filename)
    return send_file(report_path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
