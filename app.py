import os

from dotenv import load_dotenv
from flask import Flask, request, jsonify
import librosa
import laion_clap


load_dotenv()

CLAP_CKPT_PATH = os.getenv("CLAP_CKPT_PATH")
UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER")

model = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-base")
model.load_ckpt(CLAP_CKPT_PATH)

app = Flask(__name__)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


@app.route("/", methods=["GET"])
def hello_world():
    return jsonify({"status": "ok"})


@app.route("/embed-text", methods=["POST"])
def embed_text():
    data = request.get_json()
    if not isinstance(data, list) or not all(isinstance(item, str) for item in data):
        return (
            jsonify({"message": "Invalid input, expecting an array of strings."}),
            400,
        )

    text_embed = model.get_text_embedding(data, use_tensor=False)
    return jsonify(text_embed.tolist())


@app.route("/embed-audio", methods=["POST"])
def embed_audio():
    if "file" not in request.files:
        return jsonify({"message": "No file part in the request"}), 400

    file = request.files["file"]
    if not file or file.filename == "":
        return jsonify({"message": "Missing file name"}), 400

    filename = file.filename
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(file_path)

    audio_data, _ = librosa.load(file_path, sr=48000)  # sample rate should be 48000
    audio_data = audio_data.reshape(1, -1)  # Make it (1,T) or (N,T)
    audio_embed = model.get_audio_embedding_from_data(x=audio_data, use_tensor=False)

    return jsonify(audio_embed.tolist())


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=4000)
