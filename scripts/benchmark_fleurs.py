"""Vergleicht lokale Diktiermodelle auf Deutsch (FLEURS de_de, Testteil) mit sherpa-onnx auf dem Prozessor, so wie
Pegasus Desktop sie lädt (src/main/dictation/local-stt.ts: nemo_transducer, greedy_search, 4 Threads).

Gemessen: Wortfehlerrate (WER, klein geschrieben, ohne Satzzeichen) und Rechenzeit je Sekunde Audio (RTF).
Gedacht für den Wechsel Parakeet TDT 0.6B v3 → Nemotron-Labs-Unified-ASR 0.6B (export_nemotron_unified.py).

    pip install sherpa-onnx==1.13.8 soundfile huggingface_hub numpy   (gleiche Version wie die App)
    python benchmark_fleurs.py --model parakeet=<ordner> --model nemotron-de=<ordner> [--model nemotron-auto=<ordner>]
Jeder Ordner enthält encoder.int8.onnx, decoder.int8.onnx, joiner.int8.onnx, tokens.txt. Die ersten 150
Sätze (--n) laden einmalig rund 0,5 GB von Hugging Face (google/fleurs). Ergebnis als Tabelle und
fleurs-<datum>.json mit allen Einzelsätzen im aktuellen Ordner.
"""
import argparse
import csv
import datetime
import io
import json
import re
import tarfile
import time
from pathlib import Path

import numpy as np
import sherpa_onnx
import soundfile as sf
from huggingface_hub import hf_hub_download


def normalize(text: str) -> list[str]:
    text = text.lower().replace("-", " ")
    text = re.sub(r"[^\w\s]", " ", text)
    return text.split()


def edit_distance(ref: list[str], hyp: list[str]) -> int:
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h))
        prev = cur
    return prev[-1]


def load_fleurs(n: int) -> list[tuple[str, np.ndarray, str]]:
    tsv = hf_hub_download("google/fleurs", "data/de_de/test.tsv", repo_type="dataset")
    rows, seen = [], set()
    with open(tsv, encoding="utf-8") as f:
        for row in csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            sid, file_name, raw = row[0], row[1], row[2]
            if sid in seen:
                continue
            seen.add(sid)
            rows.append((file_name, raw))
            if len(rows) == n:
                break
    wanted = {name for name, _ in rows}
    audio = {}
    tar_path = hf_hub_download("google/fleurs", "data/de_de/audio/test.tar.gz", repo_type="dataset")
    with tarfile.open(tar_path) as tar:
        for member in tar:
            name = Path(member.name).name
            if name in wanted:
                data, rate = sf.read(io.BytesIO(tar.extractfile(member).read()), dtype="float32")
                if rate != 16000:
                    raise SystemExit(f"{name}: {rate} Hz statt 16000")
                audio[name] = data
    return [(name, audio[name], raw) for name, raw in rows if name in audio]


# Wie die App (local-stt.ts PAD_SECONDS): Stille am Ende, sonst fehlen letzte Wörter
PAD = np.zeros(int(0.6 * 16000), dtype=np.float32)


def recognizer(folder: Path, threads: int):
    return sherpa_onnx.OfflineRecognizer.from_transducer(
        encoder=str(folder / "encoder.int8.onnx"), decoder=str(folder / "decoder.int8.onnx"),
        joiner=str(folder / "joiner.int8.onnx"), tokens=str(folder / "tokens.txt"), num_threads=threads,
        sample_rate=16000, feature_dim=80, decoding_method="greedy_search", model_type="nemo_transducer")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="append", required=True, help="name=ordner")
    parser.add_argument("--n", type=int, default=150)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    samples = load_fleurs(args.n)
    print(f"{len(samples)} Sätze, {sum(len(a) for _, a, _ in samples) / 16000 / 60:.1f} min Audio, sherpa-onnx {sherpa_onnx.__version__}")
    first = samples[0][1]
    print(f"Kontrolle Audio: dtype {first.dtype}, Form {first.shape}, {len(first) / 16000:.1f} s, "
          f"Spitze {float(np.abs(first).max()):.3f}, RMS {float(np.sqrt(np.mean(first ** 2))):.4f}")
    report = {"datum": datetime.date.today().isoformat(), "saetze": len(samples), "modelle": {}}

    for spec in args.model:
        name, folder = spec.split("=", 1)
        start = time.perf_counter()
        rec = recognizer(Path(folder), args.threads)
        load_s = time.perf_counter() - start
        errors = words = 0
        audio_s = decode_s = 0.0
        details = []
        for file_name, data, raw in samples:
            stream = rec.create_stream()
            stream.accept_waveform(16000, np.ascontiguousarray(np.concatenate([data, PAD]), dtype=np.float32))
            t = time.perf_counter()
            rec.decode_stream(stream)
            decode_s += time.perf_counter() - t
            audio_s += len(data) / 16000
            ref, hyp = normalize(raw), normalize(stream.result.text)
            e = edit_distance(ref, hyp)
            errors += e
            words += len(ref)
            details.append({"datei": file_name, "referenz": raw, "erkannt": stream.result.text, "fehler": e})
        wer = errors / max(words, 1) * 100
        rtf = decode_s / audio_s
        report["modelle"][name] = {"ordner": folder, "wer": round(wer, 2), "rtf": round(rtf, 4),
                                   "laden_s": round(load_s, 1), "saetze": details}
        print(f"{name:20s} WER {wer:5.2f} %   RTF {rtf:.3f}   Laden {load_s:.1f} s")
        print(f"  Beispiel: {details[0]['erkannt'][:100]!r}")

    out = Path(f"fleurs-{report['datum']}.json")
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Einzelergebnisse: {out.resolve()}")


if __name__ == "__main__":
    main()
