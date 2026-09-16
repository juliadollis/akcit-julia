#!/usr/bin/env python3
"""
scripts/compara_zeroshot.py
===========================
Monta a comparacao dos bracos treinados contra o DepthPro zero-shot, lendo os
`test_metrics.json` do disco. GERADO, nunca transcrito a mao.

Uso:
    python3 scripts/compara_zeroshot.py <pasta_com_os_bracos> \
        --zero-shot <pasta_do_zero_shot> [--seeds 0-2] [--titulo "..."]

A pasta dos bracos e um espelho do runs_riemann: <braco>/seed_<n>/test_metrics.json.

`--seeds` recorta a faixa (ex.: `0-2` para as tres seeds do envio original,
`3-9` para as que fecharam depois). Sem ele, entra tudo o que existir.

Cada braco pode ter um numero DIFERENTE de seeds concluidas, entao a tabela
reporta o n de cada um em vez de fingir simetria. Alem da media sai a amplitude
entre seeds e a PIOR seed do braco: se ate a pior bate o zero-shot, a conclusao
nao depende de media nenhuma.
"""
import argparse
import json
from pathlib import Path

# (chave, rotulo, maior_e_melhor)
METRICAS = [
    ("boundary_fscore", "F-score de borda (limiar fixo)", True),
    # fmax e f_auc entram porque o fscore de limiar fixo e sensivel a CALIBRACAO da
    # distribuicao de gradiente, nao so a qualidade da borda. O fmax e o F no melhor
    # limiar de cada modelo, e o f_auc integra a varredura. Sem os tres lado a lado
    # nao da para saber se um ganho e borda melhor ou limiar mais bem posicionado.
    ("boundary_fmax", "F-score de borda no melhor limiar", True),
    ("boundary_f_auc", "area sob a varredura de limiar", True),
    ("abs_rel", "AbsRel", False),
    ("d1", "delta1", True),
    ("rmse", "RMSE", False),
]


def faixa(txt):
    """'0-2' -> {0,1,2};  '0,3,7' -> {0,3,7};  None -> None (tudo)."""
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


def le_braco(pasta: Path, seeds_ok):
    seeds = {}
    for q in sorted(pasta.glob("seed_*/test_metrics.json"),
                    key=lambda p: int(p.parent.name.split("_")[1])):
        n = int(q.parent.name.split("_")[1])
        if seeds_ok is not None and n not in seeds_ok:
            continue
        seeds[n] = json.loads(q.read_text())
    return seeds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bracos")
    ap.add_argument("--zero-shot", required=True)
    ap.add_argument("--seeds", default=None)
    ap.add_argument("--titulo", default="Bracos treinados contra o DepthPro zero-shot (Spring, teste)")
    ap.add_argument("--nome-zero-shot", default="ZERO_SHOT_spring")
    args = ap.parse_args()

    seeds_ok = faixa(args.seeds)
    zs = json.loads((Path(args.zero_shot) / "test_metrics.json").read_text())
    raiz = Path(args.bracos)
    bracos = {}
    for d in sorted(raiz.iterdir()):
        if not d.is_dir() or d.name == args.nome_zero_shot:
            continue
        s = le_braco(d, seeds_ok)
        if s:
            bracos[d.name] = s

    print(f"# {args.titulo}\n")
    print("Gerado por `scripts/compara_zeroshot.py`. Zero-shot = DepthPro de "
          "prateleira, sem fine-tune, medido pelo mesmo `Trainer.validate()` que "
          "gerou o `test_metrics.json` de cada seed treinada.\n")
    if seeds_ok:
        print(f"Recorte de seeds: `{args.seeds}`.\n")
    print("`pior seed` e a seed menos favoravel do braco naquela metrica.\n")

    for chave, rotulo, maior_melhor in METRICAS:
        seta = "maior melhor" if maior_melhor else "menor melhor"
        print(f"\n## {rotulo} ({seta})\n")
        print(f"Zero-shot: **{zs[chave]:.4f}**\n")
        print("| braco | n | seeds | media | amplitude | pior seed | delta vs zero-shot | bate o zero-shot |")
        print("|---|---|---|---|---|---|---|---|")
        for nome, seeds in bracos.items():
            vals = [m[chave] for m in seeds.values()]
            media = sum(vals) / len(vals)
            pior = min(vals) if maior_melhor else max(vals)
            delta = media - zs[chave]
            ganho = delta > 0 if maior_melhor else delta < 0
            pior_ganha = (pior > zs[chave]) if maior_melhor else (pior < zs[chave])
            veredito = "sim" if ganho else "**nao**"
            if ganho and pior_ganha:
                veredito = "sim, ate a pior seed"
            lista = ",".join(str(s) for s in sorted(seeds))
            print(f"| {nome} | {len(vals)} | {lista} | {media:.4f} | "
                  f"{min(vals):.4f} a {max(vals):.4f} | {pior:.4f} | "
                  f"{delta:+.4f} | {veredito} |")


if __name__ == "__main__":
    main()
