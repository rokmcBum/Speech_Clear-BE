import base64
import io

import librosa
import matplotlib
from matplotlib import pyplot as plt

matplotlib.use("Agg")


def draw(tmp_path):
    y, sr = librosa.load(tmp_path, sr=16000)
    plt.figure(figsize=(10, 3))
    librosa.display.waveshow(y, sr=sr, alpha=0.7, color="royalblue")

    plt.xlabel("Time (seconds)", fontsize=10, color="gray")
    plt.ylabel("Amplitude (dB)", fontsize=10, color="gray")

    plt.grid(True, linestyle="--", linewidth=0.4, alpha=0.6)
    plt.tight_layout(pad=0.5)

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close()
    buf.seek(0)

    # Base64 인코딩
    waveform_base64 = base64.b64encode(buf.read()).decode("utf-8")

    return f"data:image/png;base64,{waveform_base64}"
