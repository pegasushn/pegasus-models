"""Exportiert NVIDIA Nemotron-Labs-Unified-ASR 0.6B als lokales Diktiermodell für sherpa-onnx (Nachfolger von
Parakeet TDT 0.6B v3 in src/main/dictation/local-model.ts).

Das Modell (https://huggingface.co/nvidia/nemotron-labs-unified-asr-0.6b, Lizenz OpenMDW 1.1) ist ein NeMo-RNNT
(Unified FastConformer, 25 europäische Sprachen) mit Sprach-Prompt: Nach dem Encoder hängt NeMo einen One-Hot-Vektor
der Sprache an jeden Frame und schickt beides durch `prompt_kernel` (Linear → ReLU → Linear). sherpa-onnx kennt den
Prompt im Offline-Erkenner nicht. Weil die Sprache je Diktat fest ist, baut dieses Skript den Prompt fest in den
Encoder ein (eine Variante je Sprache). Danach ist es für sherpa-onnx ein gewöhnlicher NeMo-Transducer
(`modelType: 'nemo_transducer'`, RNNT statt TDT, 128 Mel-Kanäle), wie parakeet-unified-en-0.6b.

Vorlagen: sherpa-onnx scripts/nemo/parakeet-unified-en-0.6b/export_onnx.py (Offline-Export, Metadaten) und
scripts/nemo/nemotron-3.5-asr-streaming-0.6b/export_onnx.py (Prompt hinter dem Encoder).
Quantisierung wie quantize_parakeet.py: int8 nur für MatMul-Gewichte, pro Kanal (der übliche Export mit QUInt8
für den ganzen Encoder kostete bei Parakeet auf Deutsch ein Drittel Genauigkeit).

Ablauf (Python 3.11/3.12, rund 8 GB Arbeitsspeicher, 2,5 GB Download, ohne Grafikkarte möglich):
    pip install "nemo-toolkit[asr] @ git+https://github.com/NVIDIA-NeMo/Speech.git@main" onnx onnxruntime sherpa-onnx soundfile
    python export_nemotron_unified.py <arbeitsordner> --lang de --lang auto
Ergebnis je Sprache: <arbeitsordner>/nemotron-unified-asr-0.6b-<sprache>-q8pc/ mit encoder.int8.onnx,
decoder.int8.onnx, joiner.int8.onnx, tokens.txt, manifest.json, NOTICE.txt; Größen und SHA-256 für LOCAL_MODEL. Mit --wav <datei>
transkribiert das Skript die Datei zur Kontrolle mit NeMo und mit dem fertigen int8-Modell (sherpa-onnx).
Danach messen: benchmark_fleurs.py.
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

import nemo.collections.asr as nemo_asr
import onnx
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic

MODEL_NAME = "nvidia/nemotron-labs-unified-asr-0.6b"
FILES = ("encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx", "tokens.txt")


def add_meta_data(path: Path, meta: dict, external: bool) -> None:
    model = onnx.load(str(path), load_external_data=True)
    while len(model.metadata_props):
        model.metadata_props.pop()
    for key, value in meta.items():
        prop = model.metadata_props.add()
        prop.key = key
        prop.value = str(value)
    if external:
        onnx.save(model, str(path), save_as_external_data=True, all_tensors_to_one_file=True,
                  location=path.stem + ".weights")
    else:
        onnx.save(model, str(path))


def with_fixed_prompt(asr_model, prompt_index: int) -> None:
    """Hängt den Sprach-Prompt fest an `encoder.forward_for_export` (den Weg, den NeMos export() nimmt)."""
    encoder = asr_model.encoder
    # Ursprüngliche Klasse merken: beim zweiten Aufruf (andere Sprache) sonst Prompt doppelt
    base = encoder.__dict__.get("_pegasus_base", type(encoder))
    encoder.__dict__["_pegasus_base"] = base
    one_hot = torch.zeros(int(asr_model.num_prompts))
    one_hot[prompt_index] = 1.0

    class PromptedEncoder(base):  # type: ignore[misc, valid-type]
        def forward_for_export(self, audio_signal, length, cache_last_channel=None, cache_last_time=None,
                               cache_last_channel_len=None):
            encoded, encoded_len = base.forward_for_export(self, audio_signal, length)
            x = encoded.transpose(1, 2)  # (B, D, T) -> (B, T, D)
            prompt = self.pegasus_prompt.to(x.dtype).view(1, 1, -1).expand(x.shape[0], x.shape[1], -1)
            x = self.pegasus_prompt_kernel(torch.cat([x, prompt], dim=-1)).to(encoded.dtype)
            return x.transpose(1, 2), encoded_len

    encoder.__class__ = PromptedEncoder
    encoder.pegasus_prompt_kernel = asr_model.prompt_kernel
    if hasattr(encoder, "pegasus_prompt"):
        encoder.pegasus_prompt.copy_(one_hot)
    else:
        encoder.register_buffer("pegasus_prompt", one_hot, persistent=False)


def write_manifest(target: Path, lang: str) -> list[dict]:
    files = []
    for name in FILES:
        data = (target / name).read_bytes()
        files.append({"name": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
    manifest = {
        "id": target.name,
        "model": f"NVIDIA Nemotron-Labs-Unified-ASR 0.6B ({MODEL_NAME})",
        "language_prompt": lang,
        "format": "ONNX für sherpa-onnx (NeMo-Transducer, RNNT, Sprach-Prompt fest im Encoder), int8 pro Kanal "
                  "quantisiert von PEGASUS (export_nemotron_unified.py)",
        "license": "OpenMDW-1.1",
        "licenseUrl": "https://github.com/OpenMDW/OpenMDW/blob/main/1.1/LICENSE.OpenMDW-1.1",
        "source": f"https://huggingface.co/{MODEL_NAME}",
        "hardware": "Läuft auf dem Prozessor (ONNX Runtime CPU), keine Grafikkarte nötig",
        "totalBytes": sum(f["bytes"] for f in files),
        "files": files,
    }
    (target / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (target / "NOTICE.txt").write_text(
        f"{target.name}\n\n"
        f"Dieses Modell basiert auf NVIDIA Nemotron-Labs-Unified-ASR 0.6B (https://huggingface.co/{MODEL_NAME}),\n"
        "Copyright NVIDIA Corporation, lizenziert unter der OpenMDW License Agreement, Version 1.1\n"
        "(https://github.com/OpenMDW/OpenMDW/blob/main/1.1/LICENSE.OpenMDW-1.1).\n\n"
        "Änderungen durch PEGASUS: Export nach ONNX für sherpa-onnx, Sprach-Prompt "
        f"„{lang}“ fest im Encoder, int8-Quantisierung pro Kanal.\n"
        "NVIDIA unterstützt oder empfiehlt PEGASUS nicht.\n", encoding="utf-8")
    return files


def quantize(src: Path, dst: Path) -> None:
    quantize_dynamic(str(src), str(dst), per_channel=True, weight_type=QuantType.QInt8,
                     op_types_to_quantize=["MatMul"])


def sherpa_transcribe(folder: Path, wav: Path) -> str:
    import sherpa_onnx
    import soundfile as sf

    rec = sherpa_onnx.OfflineRecognizer.from_transducer(
        encoder=str(folder / "encoder.int8.onnx"), decoder=str(folder / "decoder.int8.onnx"),
        joiner=str(folder / "joiner.int8.onnx"), tokens=str(folder / "tokens.txt"), num_threads=4,
        sample_rate=16000, feature_dim=128, decoding_method="greedy_search", model_type="nemo_transducer")
    samples, rate = sf.read(str(wav), dtype="float32", always_2d=False)
    stream = rec.create_stream()
    stream.accept_waveform(rate, samples)
    rec.decode_stream(stream)
    return stream.result.text


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("workdir", type=Path)
    parser.add_argument("--lang", action="append", help="Sprache des festen Prompts (z. B. de, auto); mehrfach möglich")
    parser.add_argument("--nemo", type=Path, help="lokale .nemo-Datei statt Download")
    parser.add_argument("--wav", type=Path, help="Kontroll-Audio (16 kHz mono)")
    args = parser.parse_args()
    langs = args.lang or ["de", "auto"]
    work = args.workdir.resolve()
    work.mkdir(parents=True, exist_ok=True)

    if args.nemo:
        asr_model = nemo_asr.models.ASRModel.restore_from(restore_path=str(args.nemo), map_location="cpu")
    else:
        asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name=MODEL_NAME, map_location="cpu")
    asr_model.eval()
    for attr in ("prompt_kernel", "num_prompts", "resolve_prompt_language"):
        if not hasattr(asr_model, attr):
            raise SystemExit(f"{type(asr_model).__name__} hat kein '{attr}': NeMo von git main installieren")
    print("Klasse", type(asr_model).__name__, "num_prompts", asr_model.num_prompts)
    print("prompt_dictionary", dict(asr_model.cfg.model_defaults.prompt_dictionary))

    # Offline: volle Aufmerksamkeit (der Unified-Encoder kann auch Streaming mit begrenztem Kontext). Er führt den
    # Kontext mit drei Werten ([-1, -1, -1] = voll, Standard); nur umstellen, wenn etwas anderes eingestellt ist.
    current = list(getattr(asr_model.encoder, "att_context_size", None) or [])
    if current and any(v != -1 for v in current):
        asr_model.encoder.set_default_att_context_size([-1] * len(current))
    print("att_context_size", getattr(asr_model.encoder, "att_context_size", None))

    if args.wav:
        for lang in langs:
            out = asr_model.transcribe([str(args.wav)], target_lang=lang, verbose=False)
            print(f"NeMo ({lang}):", out[0].text if hasattr(out[0], "text") else out[0])

    normalize_type = asr_model.cfg.preprocessor.normalize
    meta = {
        "vocab_size": asr_model.decoder.vocab_size,
        "normalize_type": "" if normalize_type == "NA" else normalize_type,
        "pred_rnn_layers": asr_model.decoder.pred_rnn_layers,
        "pred_hidden": asr_model.decoder.pred_hidden,
        "subsampling_factor": asr_model.encoder.subsampling_factor,
        "model_type": "EncDecRNNTBPEModel",
        "version": "2",
        "model_author": "NeMo",
        "url": f"https://huggingface.co/{MODEL_NAME}",
        "feat_dim": asr_model.cfg.preprocessor.features,
    }

    fp32 = work / "fp32"
    fp32.mkdir(exist_ok=True)
    with open(fp32 / "tokens.txt", "w", encoding="utf-8") as f:
        vocab = list(asr_model.joint.vocabulary)
        for i, s in enumerate(vocab):
            f.write(f"{s} {i}\n")
        f.write(f"<blk> {len(vocab)}\n")
    asr_model.decoder.export(str(fp32 / "decoder.onnx"))
    asr_model.joint.export(str(fp32 / "joiner.onnx"))
    quantize(fp32 / "decoder.onnx", fp32 / "decoder.int8.onnx")
    quantize(fp32 / "joiner.onnx", fp32 / "joiner.int8.onnx")

    for lang in langs:
        index = asr_model.resolve_prompt_language(lang)
        print(f"Sprache {lang}: Prompt {index}")
        with_fixed_prompt(asr_model, index)
        enc_dir = work / f"fp32-encoder-{lang}"
        enc_dir.mkdir(exist_ok=True)
        enc = enc_dir / "encoder.onnx"
        asr_model.encoder.export(str(enc))
        add_meta_data(enc, {**meta, "comment": f"Sprach-Prompt fest: {lang} (Index {index}), nur Offline"}, external=True)

        target = work / f"nemotron-unified-asr-0.6b-{lang}-q8pc"
        target.mkdir(exist_ok=True)
        quantize(enc, target / "encoder.int8.onnx")
        add_meta_data(target / "encoder.int8.onnx", {**meta, "comment": f"Sprach-Prompt fest: {lang} (Index {index}), "
                                                     "nur Offline, int8 MatMul pro Kanal (Pegasus)"}, external=False)
        for name in ("decoder.int8.onnx", "joiner.int8.onnx", "tokens.txt"):
            shutil.copy(fp32 / name, target / name)

        shutil.rmtree(enc_dir, ignore_errors=True)  # fp32-Encoder (2,4 GB) wird nicht mehr gebraucht
        print(f"\n{target.name}")
        for f in write_manifest(target, lang):
            print(f"  {f['name']}  {f['bytes']}  {f['sha256']}")
        if args.wav:
            try:
                print(f"  sherpa-onnx int8 ({lang}):", sherpa_transcribe(target, args.wav))
            except ImportError:
                print("  (sherpa-onnx nicht installiert, Kontrolle übersprungen)")


if __name__ == "__main__":
    main()
