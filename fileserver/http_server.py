from flask import Flask, request, send_from_directory, jsonify, send_file
import os

app = Flask(__name__)

absolute_dir_path = os.path.dirname(os.path.realpath(__file__))
UPLOAD_FOLDER = os.path.join(absolute_dir_path, "uploads")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file part"})

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"})
    filename = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    print("HI", flush=True)
    print(filename, flush=True)
    file.save(filename)

    return jsonify({"message": "File uploaded successfully"})


@app.route("/download/<filename>", methods=["GET"])
def download_file(filename):
    return send_file(
        os.path.join(app.config["UPLOAD_FOLDER"], filename), as_attachment=False
    )


@app.route("/")
def root():
    return "Welcome to the File Server!"


if __name__ == "__main__":
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(host="0.0.0.0", port=8080, debug=True)
