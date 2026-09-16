"""Gravação de amostras em disco.

Duas garantias que o writer antigo não dava:

1. **Nada é normalizado na gravação.** O defeito D1 nasceu de um `dm / dm.max()` que
   vivia num bloco de visualização e virou o dado, apagando o K algebricamente. Aqui
   o writer não faz aritmética sobre o sinal — só codifica o que recebe.

2. **Campo ausente é erro, não omissão.** O writer antigo filtrava valores `None` do
   JSON, então a diferença entre "não medido" e "medido como zero" desaparecia.
   `validate_metadata` roda antes de cada gravação.

O mapa de defocus **não é gravado**. Ele é derivado no dataloader pela mesma função
da geração — ver `dataio.sample` para o porquê.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from dataio.sample import Sample, validate_metadata
from dataio.split import SceneSplit


def _require_png_writer():
    """PNG uint16 sem cv2, que quebra o container (GLIBC 2.38 contra 2.35)."""
    try:
        from PIL import Image
    except ImportError as exc:                       # sem fallback
        raise ImportError(
            "Pillow é necessário para gravar PNG. Instale com "
            "`pip install --no-deps --target <projeto>/.pydeps pillow`."
        ) from exc
    return Image


@dataclass
class WriteStats:
    samples: int = 0
    bytes_written: int = 0

    def summary(self) -> str:
        if self.samples == 0:
            return "[writer] nada gravado."
        mb = self.bytes_written / 1e6
        return (f"[writer] {self.samples} amostras · {mb:.1f} MB · "
                f"{mb / self.samples:.2f} MB por amostra")


class FileSampleWriter:
    """Um diretório por release, arquivos planos por amostra.

    Layout, com `<id>` = `sample_id`:

        <out>/depth/<id>.png        disparidade uint16 (ver `dataio.encoding`)
        <out>/mask/<id>.png         máscara final de foco, uint8 — na rota C é a
                                    REGIÃO REFINADA, a mesma que produziu
                                    `focus_disparity`; `mask_source` diz qual é
        <out>/meta/<id>.json        TUDO que não é pixel
        <out>/generated/<id>_*.jpg  só o que ESTA rota gera (AIF em B, bokeh em A)
        <out>/manifest.jsonl        uma linha por amostra, para varredura rápida
    """

    def __init__(self, output_dir: str | Path, *, split: SceneSplit,
                 jpeg_quality: int = 95):
        self.root = Path(output_dir)
        self.jpeg_quality = int(jpeg_quality)
        for sub in ("depth", "mask", "meta", "generated"):
            (self.root / sub).mkdir(parents=True, exist_ok=True)

        # O split é MATERIALIZADO no release, não deixado para o config do treino.
        # Config é editável, some no rsync e diverge entre runs; um arquivo no release
        # não. Antes `SceneSplit.save()` existia e ninguém o chamava daqui.
        self.split = split
        self.split_path = split.save(self.root / "split.json")

        self.manifest_path = self.root / "manifest.jsonl"
        self._manifest = self.manifest_path.open("a", encoding="utf-8")
        self._seen: set[str] = self.completed_ids()
        self.stats = WriteStats()

    # -- gravação --------------------------------------------------------------

    def write(self, sample: Sample) -> dict:
        Image = _require_png_writer()
        meta = sample.metadata()
        validate_metadata(meta)                      # antes de tocar no disco

        # Colisão de id gravava 2 linhas de manifesto e 1 arquivo `meta/`, com a
        # segunda sobrescrevendo a primeira em silêncio — a contagem publicada
        # inflava sem que nada denunciasse.
        if sample.sample_id in self._seen:
            raise ValueError(
                f"sample_id repetido: {sample.sample_id!r}. Já gravado neste release."
            )

        # O split da amostra vem do split MATERIALIZADO, por cena. Se a cena não está
        # nele, é erro de manifesto — não caso a resolver em runtime.
        split_side = self.split.of(sample.refs.scene_id)

        written = 0
        depth_path = self.root / "depth" / f"{sample.sample_id}.png"
        Image.fromarray(sample.depth.disparity_u16, mode="I;16").save(depth_path, optimize=True)
        written += depth_path.stat().st_size

        mask_path = self.root / "mask" / f"{sample.sample_id}.png"
        mask_u8 = (np.asarray(sample.mask) > 0.5).astype(np.uint8) * 255
        Image.fromarray(mask_u8, mode="L").save(mask_path, optimize=True)
        written += mask_path.stat().st_size

        # Ordem de canal é DECLARADA, não assumida. A inversão cega `[..., ::-1]`
        # transformava um RGB [200,0,0] em [0,0,200] sem que nada registrasse a
        # convenção — e na rota B a AIF gerada pela DeblurNet é exatamente uma imagem
        # de 3 canais.
        for name, image in sample.generated_images.items():
            path = self.root / "generated" / f"{sample.sample_id}_{name}.jpg"
            array = np.asarray(image)
            if array.ndim == 3 and array.shape[2] == 3:
                if sample.channel_order not in ("bgr", "rgb"):
                    raise ValueError(f"channel_order desconhecido: {sample.channel_order!r}")
                if sample.channel_order == "bgr":
                    array = array[..., ::-1]
            Image.fromarray(np.clip(array, 0, 255).astype(np.uint8)).save(
                path, quality=self.jpeg_quality, subsampling=0)
            written += path.stat().st_size

        meta_path = self.root / "meta" / f"{sample.sample_id}.json"
        meta_path.write_text(sample.metadata_json(), encoding="utf-8")
        written += meta_path.stat().st_size

        # A linha do manifesto é a superfície de varredura rápida — então ela carrega
        # justamente os campos que denunciariam um fallback: `split`, `mask_source`,
        # `depth_backend`, `control_version` e `max_coc`. Antes ela omitia os cinco.
        self._manifest.write(json.dumps({
            "sample_id": sample.sample_id, "route": sample.route,
            "scene_id": sample.refs.scene_id, "split": split_side,
            "source_dataset": sample.refs.source_dataset,
            "source_sample_id": sample.refs.source_sample_id,
            "k_value": meta["k_value"], "k_source": meta["k_source"],
            "focus_disparity": meta["focus_disparity"],
            # K é um número EM PIXEL. Sem a resolução ao lado dele, a linha do
            # manifesto não diz o que o K significa, e quem for reamostrar ou
            # comparar entre rotas não tem como normalizar. É a regra do CLAUDE.md:
            # toda quantidade em pixel carrega a resolução em que foi medida.
            "image_h": meta["image_h"], "image_w": meta["image_w"],
            "depth_h": meta["depth_h"], "depth_w": meta["depth_w"],
            "max_coc": meta["max_coc"],
            "is_k_censored": meta["is_k_censored"],
            "is_valid_for_control": meta["is_valid_for_control"],
            "calibration_ssim": meta["calibration_ssim"],
            "mask_source": meta["mask_source"],
            "depth_backend": meta["depth_backend"],
            "control_version": meta["control_version"],
            # A marcação do refinamento da região em foco viaja no manifesto porque é
            # por ela que se monta o treino COM e SEM as amostras refinadas — e
            # filtrar 22.990 amostras não pode exigir abrir 22.990 JSONs. Se o número
            # de `retention_only` for alto, ele tem que estar visível numa varredura
            # de uma linha, não escondido no metadado.
            "focus_source": meta["focus_source"],
            "focus_was_refined": meta["focus_was_refined"],
            "focus_agreement": meta["focus_agreement"],
            "focus_retention_in_region": meta["focus_retention_in_region"],
            "focus_region_area_ratio": meta["focus_region_area_ratio"],
            # `focus_region_area_ratio` é fração, mas foi medida numa grade específica,
            # e `focus_retention_window_px` é pixel puro. Mesma regra do `image_h`
            # acima: quantidade em pixel viaja com a resolução em que foi medida.
            "focus_retention_h": meta["focus_retention_h"],
            "focus_retention_w": meta["focus_retention_w"],
        }, ensure_ascii=False) + "\n")
        self._manifest.flush()

        self._seen.add(sample.sample_id)
        self.stats.samples += 1
        self.stats.bytes_written += written
        return meta

    # -- retomada --------------------------------------------------------------

    def completed_ids(self) -> set[str]:
        """IDs já gravados, lidos do manifesto. Base da retomada."""
        done: set[str] = set()
        if not self.manifest_path.exists():
            return done
        with self.manifest_path.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    done.add(json.loads(line)["sample_id"])
                except (json.JSONDecodeError, KeyError):
                    continue
        return done

    def close(self) -> None:
        self._manifest.close()

    def __enter__(self) -> "FileSampleWriter":
        return self

    def __exit__(self, *exc_info) -> None:
        print(self.stats.summary())
        self.close()


def read_metadata(output_dir: str | Path, sample_id: str) -> dict:
    return json.loads((Path(output_dir) / "meta" / f"{sample_id}.json").read_text(encoding="utf-8"))


def iter_manifest(output_dir: str | Path):
    """Itera o manifesto sem carregar tudo em memória."""
    path = Path(output_dir) / "manifest.jsonl"
    if not path.exists():
        return
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def estimate_disk_budget(
    n_samples: int, depth_long_side: int, *,
    generates_image: bool, image_megapixels: float,
) -> dict:
    """Orçamento de disco ANTES de rodar. A cota é 500 GB soft, 600 GB hard.

    Constantes medidas: PNG uint16 de profundidade suave ~1,2 B/px; JPEG q95 ~0,63 B/px.
    Existe porque guardar as imagens de origem de novo custaria 365 GB contra 115 GB
    de folga — e isso precisa aparecer antes do run, não na hora 9.
    """
    depth_px = depth_long_side * depth_long_side * 0.75
    control_gb = n_samples * (depth_px * 1.2 + depth_px * 0.04) / 1e9
    image_gb = n_samples * image_megapixels * 1e6 * 0.63 / 1e9 if generates_image else 0.0
    return {
        "n_samples": n_samples,
        "control_gb": round(control_gb, 1),
        "generated_image_gb": round(image_gb, 1),
        "total_gb": round(control_gb + image_gb, 1),
    }
