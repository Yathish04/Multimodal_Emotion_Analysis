# realtime_emotion_fixed.py
"""
Real-time voice emotion detection (fixed for Windows)
- Uses SpeechBrain pre-trained emotion model.
- Writes temp WAVs robustly (avoids Windows file-lock issues).
- Converts temp path to POSIX style before calling SpeechBrain (fixes libsndfile path bug).
- Handles tensor/ndarray probability outputs safely.
"""

import os
import time
import tempfile
import traceback

import numpy as np
import torch
import sounddevice as sd
import soundfile as sf
import librosa

from speechbrain.inference import foreign_class

# ----------------- CONFIG -----------------
CHUNK_SECONDS = 3.0      # seconds per recorded chunk
SAMPLE_RATE = 16000      # model expects 16kHz
CHANNELS = 1             # mono
MODEL_SOURCE = "speechbrain/emotion-recognition-wav2vec2-IEMOCAP"
PYSOURCE_FILE = "custom_interface.py"
CLASSNAME = "CustomEncoderWav2vec2Classifier"
# ------------------------------------------


def record_chunk(duration=CHUNK_SECONDS, sr=SAMPLE_RATE, channels=CHANNELS):
    """Record audio chunk from default microphone as float32 mono numpy array."""
    print(f"Recording {duration:.2f}s ... (samplerate request: {sr} Hz)")
    recording = sd.rec(int(duration * sr), samplerate=sr, channels=channels, dtype="float32")
    sd.wait()
    # recording shape: (frames, channels) or (frames,) if channels==1
    if recording.ndim > 1 and recording.shape[1] > 1:
        recording = np.mean(recording, axis=1)
    else:
        recording = recording.flatten()
    return recording.astype("float32"), sr


def ensure_16k_mono_array(arr, orig_sr):
    """Resample to SAMPLE_RATE and ensure float32 mono."""
    if orig_sr != SAMPLE_RATE:
        arr = librosa.resample(arr, orig_sr=orig_sr, target_sr=SAMPLE_RATE)
        orig_sr = SAMPLE_RATE
    # already mono (we ensured in record_chunk)
    return arr.astype("float32"), orig_sr


def save_temp_wav(arr, sr):
    """
    Save numpy array to a temporary WAV file and return absolute path.
    Uses NamedTemporaryFile(delete=False) and closes it before writing - robust on Windows.
    """
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_path = tmp.name
    tmp.close()  # release handle so other libraries can open it
    # Write with soundfile (libsndfile) as 16-bit PCM
    sf.write(tmp_path, arr, sr, subtype="PCM_16")
    return os.path.abspath(tmp_path)


def load_classifier():
    """Load SpeechBrain foreign_class classifier (downloads model on first run)."""
    print("Loading classifier (this may download model files on first run)...")
    classifier = foreign_class(
        source=MODEL_SOURCE,
        pymodule_file=PYSOURCE_FILE,
        classname=CLASSNAME
    )
    return classifier


def get_topk_probs(probs, labels=None, k=4):
    """
    Return top-k (label, prob) pairs sorted by descending probability.
    Accepts torch.Tensor or numpy array. Handles shapes like (1,N) and (N,).
    """
    if probs is None:
        return []

    # Convert torch tensor -> numpy
    if isinstance(probs, torch.Tensor):
        probs = probs.detach().cpu().numpy()

    probs = np.array(probs)

    # Squeeze if there's a batch dimension e.g., (1, N)
    if probs.ndim > 1:
        probs = probs.squeeze()

    if probs.size == 0:
        return []

    probs = probs.ravel()
    order = np.argsort(probs)[::-1]  # indices descending
    topk = []
    for idx in order[:min(k, probs.size)]:
        label = labels[idx] if (labels is not None and idx < len(labels)) else f"idx_{idx}"
        topk.append((label, float(probs[idx])))
    return topk


def pretty_print_results(out_prob, score, index, text_lab):
    """Nicely print classifier outputs (robust to different tensor/array types)."""
    if text_lab is not None:
        print("Predicted label:", text_lab)
    if out_prob is not None:
        topk = get_topk_probs(out_prob, k=4)
        if topk:
            print("Top probabilities:")
            for lbl, p in topk:
                print(f"  {lbl:20s} : {p:.4f}")
        else:
            print("No probabilities available (empty).")
    else:
        print("No probability output from classifier.")

    if score is not None:
        try:
            s_val = float(score.detach().cpu().item()) if isinstance(score, torch.Tensor) else float(score)
            print(f"Model score: {s_val:.4f}")
        except Exception:
            print("Model score: (unavailable)")


def main():
    classifier = load_classifier()
    print(f"Ready. Recording in chunks of {CHUNK_SECONDS}s. Press Ctrl+C to stop.\n")

    chunk_count = 0
    try:
        while True:
            chunk_count += 1
            t0 = time.time()

            # 1) Record
            arr, sr = record_chunk(duration=CHUNK_SECONDS, sr=SAMPLE_RATE, channels=CHANNELS)
            arr, sr = ensure_16k_mono_array(arr, orig_sr=sr)

            # 2) Save robust temp wav
            tmp_wav = save_temp_wav(arr, sr)
            print(f"Chunk #{chunk_count}: saved temp wav -> {tmp_wav}")

            # 3) Convert to POSIX path to avoid libsndfile Windows path bug
            wav_for_classify = os.path.abspath(tmp_wav).replace("\\", "/")
            print("Passing POSIX path to classifier:", wav_for_classify)

            # 4) Run inference (catch exceptions and print debug info)
            try:
                out_prob, score, index, text_lab = classifier.classify_file(wav_for_classify)
            except Exception as e:
                print("Inference error:", type(e).__name__, e)
                traceback.print_exc()
                # debug info about file
                try:
                    exists = os.path.exists(tmp_wav)
                    size = os.path.getsize(tmp_wav) if exists else None
                    print("Temp exists?", exists, "size:", size)
                except Exception as ex2:
                    print("Could not stat temp file:", ex2)
                out_prob, score, index, text_lab = None, None, None, None

            elapsed = time.time() - t0
            print(f"Chunk #{chunk_count}  (record + infer = {elapsed:.2f}s)")

            # 5) Print results safely
            pretty_print_results(out_prob, score, index, text_lab)

            # 6) Cleanup the temp file
            try:
                os.remove(tmp_wav)
            except Exception:
                pass

            print("-" * 40)

    except KeyboardInterrupt:
        print("\nStopped by user. Goodbye!")


if __name__ == "__main__":
    main()
