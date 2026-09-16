#!/usr/bin/env python3
"""
scripts/contraste_controle.py
=============================
Contrasta cada braco contra o CONTROLE, seed a seed, usando so as seeds que os
dois lados tem. GERADO, nunca transcrito a mao.

Por que pareado por seed: a seed fixa inicializacao e ordem dos dados nos dois
bracos, entao a diferenca por seed remove a variacao que vem so do sorteio. E o
mesmo motivo pelo qual o `consolida_reteste.py` se recusa a comparar bracos com
conjuntos de seeds diferentes: comparar media de n=8 contra media de n=10, com
seeds distintas, mistura efeito com sorteio.

Uso:
    python3 scripts/contraste_controle.py <pasta_com_os_bracos> [--controle B0_berhu]
"""
import argparse
import json
import statistics
from pathlib import Path

METRICAS = [
    ("boundary_fscore", "F-score de borda (limiar fixo)", True),
    # fmax e f_auc entram porque o fscore de limiar fixo e sensivel a CALIBRACAO
    # da distribuicao de gradiente, nao so a qualidade da borda.
    ("boundary_fmax", "F-score de borda no melhor limiar", True),
    ("boundary_f_auc", "area sob a varredura de limiar", True),
    ("abs_rel", "AbsRel", False),
    ("d1", "delta1", True),
    ("rmse", "RMSE", False),
]


def faixa(txt):
    """'0-5' -> {0..5};  '0,3,7' -> {0,3,7};  None -> None (tudo)."""
    if not txt:
        return None
    vals = set()
    for parte in txt.split(","):
        parte = parte.strip()
        if "-" in parte:
            a, b = parte.split("-")
            vals.update(range(int(a), int(b) + 1))
        else:
            vals.add(int(parte))
    return vals


def le_braco(pasta: Path, seeds_ok=None):
    out = {}
    for q in sorted(pasta.glob("seed_*/test_metrics.json")):
        n = int(q.parent.name.split("_")[1])
        if seeds_ok is not None and n not in seeds_ok:
            continue
        out[n] = json.loads(q.read_text())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bracos")
    ap.add_argument("--controle", default="B0_berhu")
    ap.add_argument("--seeds", default=None,
                    help="recorta a faixa de seeds, ex.: 0-5. Sem isso entra tudo.")
    args = ap.parse_args()

    seeds_ok = faixa(args.seeds)
    raiz = Path(args.bracos)
    todos = {d.name: le_braco(d, seeds_ok) for d in sorted(raiz.iterdir())
             if d.is_dir() and le_braco(d, seeds_ok)}
    if args.controle not in todos:
        raise SystemExit(f"controle {args.controle} nao encontrado em {raiz}")
    ctrl = todos[args.controle]

    print(f"# Cada braco contra o controle `{args.controle}`, pareado por seed\n")
    print("Gerado por `scripts/contraste_controle.py`. So entram as seeds que os "
          "DOIS lados tem; o `n` de cada linha e esse conjunto comum.\n")
    print("`delta` negativo em AbsRel/RMSE e positivo em F-score/delta1 favorece "
          "o braco. `seeds a favor` conta em quantas seeds do conjunto comum o "
          "braco bateu o controle.\n")

    for chave, rotulo, maior_melhor in METRICAS:
        seta = "maior melhor" if maior_melhor else "menor melhor"
        print(f"\n## {rotulo} ({seta})\n")
        print("| braco | n comum | seeds | delta medio | desvio | amplitude do delta | seeds a favor |")
        print("|---|---|---|---|---|---|---|")
        for nome, seeds in todos.items():
            if nome == args.controle:
                continue
            comuns = sorted(set(seeds) & set(ctrl))
            if not comuns:
                print(f"| {nome} | 0 | - | - | - | - | - |")
                continue
            dif = [seeds[s][chave] - ctrl[s][chave] for s in comuns]
            media = statistics.mean(dif)
            desvio = statistics.stdev(dif) if len(dif) > 1 else float("nan")
            favor = sum(1 for d in dif if (d > 0) == maior_melhor)
            lista = ",".join(str(s) for s in comuns)
            dp = f"{desvio:.4f}" if len(dif) > 1 else "n/d"
            print(f"| {nome} | {len(comuns)} | {lista} | {media:+.4f} | {dp} | "
                  f"{min(dif):+.4f} a {max(dif):+.4f} | {favor}/{len(dif)} |")


if __name__ == "__main__":
    main()
