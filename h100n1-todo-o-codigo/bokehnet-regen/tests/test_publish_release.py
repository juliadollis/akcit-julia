"""Testes do validador de release.

Um validador que nunca reprova é pior que validador nenhum — foi exatamente o caso do
`check_no_leak` antigo, cujo teste chamado "detecta vazamento" afirmava `clean is True`.
Então aqui **cada** checagem tem um teste que a faz REPROVAR, e só depois um que a faz
passar.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

_RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_RAIZ / "src"))
sys.path.insert(0, str(_RAIZ / "scripts"))

import numpy as np                                                    # noqa: E402
from PIL import Image                                                 # noqa: E402

from dataio.split import SceneSplit                                   # noqa: E402
from publish_release import Problema, escreve_card, valida            # noqa: E402


def _meta(sample_id: str, **over) -> dict:
    base = {
        "sample_id": sample_id, "route": "c", "scene_id": "0001",
        "k_value": 16.6, "k_source": "eq5_ssim_sweep", "focus_disparity": 0.5,
        "max_coc": 100.0, "is_k_censored": False, "is_valid_for_control": True,
        "calibration_ssim": 0.91, "k_analytic": 15.2, "k_effective_factor": 0.9873,
        "mask_source": "birefnet", "depth_backend": "depth_pro",
        # Marcação da região em foco (§3.2(c)). `birefnet` = M confiável, não refinada.
        "focus_source": "birefnet", "focus_was_refined": False,
        "focus_agreement": 0.78, "focus_retention_in_region": 0.88,
        "focus_region_area_ratio": 0.21,
        "focus_retention_h": 384, "focus_retention_w": 512,
        "focus_retention_window_px": 33,
        "focus_initial_mask_was_empty": False,
        "focus_initial_mask_area_ratio": 0.21,
        "focus_disparity_from_initial_mask": 0.5,
        "control_version": "metric_disparity_official_v1",
        "image_h": 1500, "image_w": 2000, "depth_h": 576, "depth_w": 768,
        "depth_encoding": "disparity_uint16",
        "disparity_min": 0.05, "disparity_max": 2.5,
        "source_dataset": "akcit-pixel/RealBokeh", "source_sample_id": f"src_{sample_id}",
        "provenance": {"pipeline_commit": "abc123", "depth_model_sha256": "d" * 64,
                       "mask_model_sha256": "m" * 64, "image_hw": [1500, 2000],
                       "image_h": 1500, "image_w": 2000},
    }
    base.update(over)
    return base


class _Release:
    """Monta um release mínimo e válido no disco, para então quebrá-lo de um jeito."""

    def __init__(self, root: Path, *, n: int = 4, val_scenes: int = 1):
        self.root = root
        for sub in ("depth", "mask", "meta"):
            (root / sub).mkdir(parents=True, exist_ok=True)

        self.ids, linhas, ledger, assignment = [], [], [], {}
        for i in range(n):
            cena = f"{i // 2:04d}"
            lado = "val" if i // 2 < val_scenes else "train"
            assignment[cena] = lado
            sid = f"c_realbokeh_{cena}_l{i % 2 + 1}"
            self.ids.append(sid)

            Image.fromarray(np.zeros((8, 8), dtype=np.uint16), mode="I;16").save(
                root / "depth" / f"{sid}.png")
            Image.fromarray(np.zeros((8, 8), dtype=np.uint8), mode="L").save(
                root / "mask" / f"{sid}.png")
            meta = _meta(sid, scene_id=cena)
            (root / "meta" / f"{sid}.json").write_text(json.dumps(meta), encoding="utf-8")

            linhas.append({"sample_id": sid, "route": "c", "scene_id": cena,
                           "split": lado, "source_dataset": meta["source_dataset"],
                           "source_sample_id": meta["source_sample_id"],
                           "k_value": meta["k_value"], "k_source": meta["k_source"],
                           "focus_disparity": 0.5, "max_coc": 100.0,
                           "is_k_censored": False, "is_valid_for_control": True,
                           "calibration_ssim": 0.91, "mask_source": "birefnet",
                           "focus_source": "birefnet", "focus_was_refined": False,
                           "focus_agreement": 0.78,
                           "focus_retention_in_region": 0.88,
                           "focus_region_area_ratio": 0.21,
                           "focus_retention_h": 384, "focus_retention_w": 512,
                           "depth_backend": "depth_pro",
                           "control_version": meta["control_version"]})
            ledger.append({"sample_id": sid, "source_sample_id": meta["source_sample_id"],
                           "shard": "s00.parquet", "row": i,
                           "aif_sha256": "a" * 64, "bokeh_sha256": "b" * 64})

        self.escreve_manifesto(linhas)
        (root / "source_images.jsonl").write_text(
            "\n".join(json.dumps(x) for x in ledger) + "\n", encoding="utf-8")
        SceneSplit(assignment, "teste", val_scenes / max(len(assignment), 1)).save(
            root / "split.json")
        (root / "run_config.json").write_text(json.dumps(
            {"provenance_base": {"pipeline_commit": "abc123",
                                 "depth_backend": "depth_pro",
                                 "source_dataset": "akcit-pixel/RealBokeh"}}),
            encoding="utf-8")

    def escreve_manifesto(self, linhas: list[dict]) -> None:
        (self.root / "manifest.jsonl").write_text(
            "\n".join(json.dumps(x) for x in linhas) + "\n", encoding="utf-8")

    def manifesto(self) -> list[dict]:
        return [json.loads(l) for l in
                (self.root / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if l]


class _Base(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        self.root = Path(self._tmp)
        self.rel = _Release(self.root)

    def tearDown(self):
        shutil.rmtree(self._tmp, ignore_errors=True)

    def assertReprova(self, trecho: str):
        with self.assertRaises(Problema) as ctx:
            valida(self.root)
        self.assertIn(trecho, str(ctx.exception))


class TestPassaQuandoEstaCerto(_Base):

    def test_release_valido_passa(self):
        resumo = valida(self.root)
        self.assertEqual(resumo["amostras"], 4)
        self.assertEqual(resumo["cenas"], 2)
        self.assertEqual(resumo["por_split"], {"val": 2, "train": 2})
        self.assertEqual(resumo["control_version"], "metric_disparity_official_v1")
        self.assertFalse(resumo["autocontido"])

    def test_avisa_que_nao_e_autocontido(self):
        self.assertTrue(any("NÃO autocontido" in a for a in valida(self.root)["avisos"]))

    def test_com_source_vira_autocontido_e_some_o_aviso(self):
        (self.root / "source").mkdir()
        (self.root / "source" / "x_aif.jpg").write_bytes(b"\xff\xd8fake")
        resumo = valida(self.root)
        self.assertTrue(resumo["autocontido"])
        self.assertFalse(any("autocontido" in a for a in resumo["avisos"]))


class TestReprovaQuandoEstaErrado(_Base):

    def test_sem_manifesto(self):
        (self.root / "manifest.jsonl").unlink()
        self.assertReprova("não há release")

    def test_sem_split_materializado(self):
        (self.root / "split.json").unlink()
        self.assertReprova("Split materializado é requisito")

    def test_sample_id_repetido(self):
        linhas = self.rel.manifesto()
        linhas.append(dict(linhas[0]))
        self.rel.escreve_manifesto(linhas)
        self.assertReprova("repetidos no manifesto")

    def test_arquivo_de_depth_faltando(self):
        (self.root / "depth" / f"{self.rel.ids[0]}.png").unlink()
        self.assertReprova("sem arquivo em depth/")

    def test_metadado_invalido(self):
        """`max_coc` fora de 100 é o normalizador escondido que quebrou o treino."""
        sid = self.rel.ids[0]
        meta = _meta(sid, scene_id="0000", max_coc=10.510746)
        (self.root / "meta" / f"{sid}.json").write_text(json.dumps(meta), encoding="utf-8")
        self.assertReprova("metadados inválidos")

    def test_control_version_divergente(self):
        sid = self.rel.ids[0]
        meta = _meta(sid, scene_id="0000", control_version="kfix_v0")
        (self.root / "meta" / f"{sid}.json").write_text(json.dumps(meta), encoding="utf-8")
        self.assertReprova("control_version divergente")

    def test_depth_backend_divergente(self):
        sid = self.rel.ids[0]
        meta = _meta(sid, scene_id="0000", depth_backend="depth_anything")
        (self.root / "meta" / f"{sid}.json").write_text(json.dumps(meta), encoding="utf-8")
        self.assertReprova("depth_backend divergente")

    def test_amostra_sem_sha256_de_origem(self):
        (self.root / "source_images.jsonl").unlink()
        self.assertReprova("sem linha em source_images.jsonl")

    def test_cena_fora_do_split(self):
        linhas = self.rel.manifesto()
        linhas[0]["scene_id"] = "9999"
        self.rel.escreve_manifesto(linhas)
        self.assertReprova("fora do split.json")

    def test_vazamento_de_split(self):
        """A mesma cena marcada dos dois lados no manifesto."""
        linhas = self.rel.manifesto()
        linhas[0]["split"] = "train" if linhas[0]["split"] == "val" else "val"
        self.rel.escreve_manifesto(linhas)
        self.assertReprova("vazamento de split")

    def test_censura_acima_de_20_por_cento_reprova(self):
        """47% era o número do release antigo. Acima de 20% o teto está errado."""
        for sid in self.rel.ids[:2]:
            cena = sid.split("_")[2]
            meta = _meta(sid, scene_id=cena, is_k_censored=True)
            (self.root / "meta" / f"{sid}.json").write_text(json.dumps(meta),
                                                            encoding="utf-8")
        self.assertReprova("K censurado no teto")

    def test_censura_baixa_e_so_aviso(self):
        """1 em 20 = 5%: passa, mas o número aparece. Reprovar 5% descartaria dado bom;
        não dizer nada é como 47% passou batido no release anterior."""
        rel = _Release(Path(tempfile.mkdtemp(dir=self._tmp)), n=20, val_scenes=3)
        sid = rel.ids[0]
        meta = _meta(sid, scene_id=sid.split("_")[2], is_k_censored=True)
        (rel.root / "meta" / f"{sid}.json").write_text(json.dumps(meta), encoding="utf-8")
        self.assertTrue(any("censurado" in a for a in valida(rel.root)["avisos"]))

    def test_validacao_vazia_e_aviso(self):
        rel = _Release(Path(tempfile.mkdtemp(dir=self._tmp)), n=4, val_scenes=0)
        self.assertTrue(any("Validação vazia" in a for a in valida(rel.root)["avisos"]))


class TestCard(_Base):

    def test_card_diz_o_essencial(self):
        resumo = valida(self.root)
        card = escreve_card(self.root, resumo, "akcit-pixel/teste").read_text(
            encoding="utf-8")
        for exigido in ("metric_disparity_official_v1", "max_coc", "cenas",
                        "rejections.jsonl", "split.json", "licença da origem",
                        "abc123"):
            with self.subTest(exigido=exigido):
                self.assertIn(exigido, card)

    def test_card_diz_que_os_pixels_estao_fora_quando_estao(self):
        card = escreve_card(self.root, valida(self.root), "x/y").read_text(
            encoding="utf-8")
        self.assertIn("não** estão aqui", card)

    def test_card_muda_quando_o_release_e_autocontido(self):
        (self.root / "source").mkdir()
        (self.root / "source" / "x_aif.jpg").write_bytes(b"\xff\xd8fake")
        card = escreve_card(self.root, valida(self.root), "x/y").read_text(
            encoding="utf-8")
        self.assertIn("bytes originais", card)

    def test_card_diz_de_onde_a_regiao_em_foco_veio(self):
        """Quem baixa o release precisa poder montar o treino COM e SEM as amostras
        refinadas — e saber quantas são antes de baixar 31 GB."""
        resumo = valida(self.root)
        self.assertEqual(resumo["focus_sources"], {"birefnet": len(self.rel.ids)})
        card = escreve_card(self.root, resumo, "x/y").read_text(encoding="utf-8")
        self.assertIn("focus_source", card)
        self.assertIn("35,2%", card)
        self.assertIn("REGIÃO FINAL de foco", card)


class TestMarcacaoDoRefinamento(_Base):

    def _com_fonte(self, fonte: str, mask_source: str):
        for sid in self.rel.ids:
            path = self.root / "meta" / f"{sid}.json"
            meta = json.loads(path.read_text(encoding="utf-8"))
            meta.update({"focus_source": fonte, "focus_was_refined": True,
                         "mask_source": mask_source})
            path.write_text(json.dumps(meta), encoding="utf-8")

    def test_lote_dominado_pelo_refinamento_AVISA(self):
        """Não reprova — o refinamento é o comportamento certo pelo §3.2(c) —, mas um
        release em que ele domina não pode virar treino sem o laudo de validação."""
        self._com_fonte("retention_only", "retention_only")
        resumo = valida(self.root)
        self.assertTrue(any("REFINADA" in a for a in resumo["avisos"]),
                        resumo["avisos"])

    def test_mascara_gravada_que_MENTE_reprova_o_release(self):
        """`focus_source=retention_only` com `mask_source=birefnet` afirma que o arquivo
        em `mask/<id>.png` é a máscara do segmentador. Quem auditasse o rótulo olharia a
        máscara errada."""
        self._com_fonte("retention_only", "birefnet")
        with self.assertRaises(Problema) as ctx:
            valida(self.root)
        self.assertIn("metadados inválidos", str(ctx.exception))

    def test_release_sem_a_marcacao_reprova(self):
        for sid in self.rel.ids:
            path = self.root / "meta" / f"{sid}.json"
            meta = json.loads(path.read_text(encoding="utf-8"))
            del meta["focus_source"]
            path.write_text(json.dumps(meta), encoding="utf-8")
        with self.assertRaises(Problema):
            valida(self.root)


if __name__ == "__main__":
    unittest.main()
