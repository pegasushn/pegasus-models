# PEGASUS Modelle

Modelle für die lokale Diktierfunktion von Pegasus Suite Desktop. Die App lädt sie selbst herunter und prüft jede Datei per SHA-256 gegen die in der App hinterlegten Werte.

## parakeet-tdt-0.6b-v3-q8pc

- Grundlage: [NVIDIA Parakeet TDT 0.6B v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3), Lizenz [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/)
- Änderungen durch PEGASUS: Export nach ONNX für sherpa-onnx, int8-Quantisierung pro Kanal
- Läuft auf dem Prozessor, keine Grafikkarte nötig
- Dateien und Prüfsummen: siehe Release `parakeet-tdt-0.6b-v3-q8pc` (`manifest.json`, `NOTICE.txt`)

## nemotron-unified-asr-0.6b-de-q8pc / -auto-q8pc (Test)

- Grundlage: [NVIDIA Nemotron-Labs-Unified-ASR 0.6B](https://huggingface.co/nvidia/nemotron-labs-unified-asr-0.6b), Lizenz [OpenMDW 1.1](https://github.com/OpenMDW/OpenMDW/blob/main/1.1/LICENSE.OpenMDW-1.1)
- Änderungen durch PEGASUS: Export nach ONNX für sherpa-onnx, Sprach-Prompt fest im Encoder (`de` bzw. `auto`), int8-Quantisierung pro Kanal
- Erzeugt vom Workflow „Nemotron Unified-ASR umwandeln“ (`.github/workflows/nemotron-unified.yml`, Skripte in `scripts/`), samt Messung gegen Parakeet (`benchmark-fleurs-de.json` im Release)
- Läuft auf dem Prozessor, keine Grafikkarte nötig

NVIDIA unterstützt oder empfiehlt PEGASUS nicht.
