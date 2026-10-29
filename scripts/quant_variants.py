"""Verkleinert ein fp32-Modell (encoder.onnx, decoder.onnx, joiner.onnx, tokens.txt) in mehreren int8-Varianten, um
die Genauigkeit auf Prozessoren mit und ohne VNNI zu vergleichen (benchmark_fleurs.py).

Hintergrund: Dynamische int8-Quantisierung rechnet Aktivierungen als uint8 mal Gewichte als int8 (U8S8). Auf
x86-Prozessoren ohne VNNI (z. B. Intel Core 6. bis 10. Generation, AMD Zen 2/3) nutzt ONNX Runtime dafür einen Befehl,
dessen Zwischensumme übersteuern kann; dann wird die Erkennung unbrauchbar. Mit VNNI (AMD Zen 4, Intel ab 11./12.
Generation) tritt das nicht auf. `reduce_range` (7-Bit-Gewichte) oder uint8-Gewichte (U8U8) vermeiden das.

    python quant_variants.py <fp32-ordner> <zielordner> [--variant s8 --variant s8rr --variant u8]
"""
import argparse
import shutil
from pathlib import Path

from onnxruntime.quantization import QuantType, quantize_dynamic

VARIANTS = {
    "s8": dict(weight_type=QuantType.QInt8, reduce_range=False),     # bisher (Parakeet und Nemotron im Release)
    "s8rr": dict(weight_type=QuantType.QInt8, reduce_range=True),    # 7-Bit-Gewichte, kein Übersteuern
    "u8": dict(weight_type=QuantType.QUInt8, reduce_range=False),    # uint8-Gewichte (U8U8)
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("src", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--variant", action="append", choices=list(VARIANTS))
    args = parser.parse_args()
    for name in args.variant or list(VARIANTS):
        target = args.out / name
        target.mkdir(parents=True, exist_ok=True)
        for part in ("encoder", "decoder", "joiner"):
            quantize_dynamic(str(args.src / f"{part}.onnx"), str(target / f"{part}.int8.onnx"), per_channel=True,
                             op_types_to_quantize=["MatMul"], **VARIANTS[name])
        shutil.copy(args.src / "tokens.txt", target / "tokens.txt")
        print("fertig", target, sum(f.stat().st_size for f in target.iterdir()))


if __name__ == "__main__":
    main()
