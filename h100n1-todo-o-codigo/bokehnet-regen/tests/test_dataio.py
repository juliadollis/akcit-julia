"""Testes do contrato de dados: codificação, metadados e split por cena."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from control.contract import MAX_COC, signed_coc_px                    # noqa: E402
from dataio.encoding import (                                          # noqa: E402
    encode_depth, decode_depth_m, decode_disparity,
    quantization_coc_error_px, resize_depth_nearest,
)
from dataio.sample import (                                            # noqa: E402
    FOCUS_SOURCE_TO_MASK_SOURCE, ControlLabel, FocusRegionRecord, KSource,
    MaskSource, Sample, SampleProvenance, SampleRefs, validate_metadata,
)
from qc.focus_region import FocusSource, RefinedFocusRegion             # noqa: E402
from dataio.split import (                                             # noqa: E402
    build_scene_split, check_no_leak, split_from_source,
)
from dataio.writer import FileSampleWriter, estimate_disk_budget       # noqa: E402
from control.contract import SampleRejected                            # noqa: E402


def _depth(h=64, w=96, near=1.0, far=60.0):
    return np.linspace(near, far, h * w, dtype=np.float32).reshape(h, w)


_PROV = SampleProvenance(pipeline_commit="abc123", depth_model_sha256="d" * 64,
                         mask_model_sha256="m" * 64, image_hw=(64, 96), seed=7)


def _focus(source=FocusSource.BIREFNET, **kw):
    """A marcação da região em foco, com a origem da máscara batendo com ela.

    Default `BIREFNET`: o caminho do paper, em que M é confiável e não se mexe.
    """
    base = dict(agreement=0.82, precision=0.76, iou=0.64,
                retention_in_region=0.91, area_ratio=0.24,
                retention_hw=(64, 96), retention_window_px=33,
                initial_mask_was_empty=False, initial_mask_area_ratio=0.24,
                disparity_from_initial_mask=1.0 / 3.0)
    base.update(kw)
    return FocusRegionRecord(source=source, **base)


def _sample(sample_id="c_0001", scene="1038", k=18.0, focus=None, **kw):
    focus = _focus() if focus is None else focus
    return Sample(
        sample_id=sample_id, route="c",
        refs=SampleRefs("akcit-pixel/RealBokeh", f"{scene}_level_3", scene,
                        aif_ref="image_focus", bokeh_ref="image_blur", source_split="train"),
        control=ControlLabel(k_value=k, k_source=KSource.EQ5_SSIM_SWEEP,
                             focus_disparity=1.0 / 3.0, is_k_censored=False,
                             depth_backend="depth_pro", calibration_ssim=0.91),
        depth=encode_depth(_depth(), image_hw=(64, 96), long_side=768),
        mask=(np.arange(64 * 96).reshape(64, 96) % 7 == 0),
        # A `mask_source` sai do `focus_source` pela ponte única: o arquivo gravado é a
        # região que produziu `focus_disparity`.
        mask_source=focus.mask_source,
        focus=focus,
        provenance=_PROV, **kw)


def _split_de(*cenas):
    return build_scene_split(list(cenas) or ["1038"], val_fraction=0.1)


# ==============================================================================
# Codificação — quantizar em DISPARIDADE, não em profundidade
# ==============================================================================

class Encoding(unittest.TestCase):
    def test_roundtrip_preserva_o_coc(self):
        """O erro que importa não é o de profundidade, é o de CoC."""
        z = _depth(200, 300, 1.0, 100.0)
        enc = encode_depth(z, image_hw=z.shape[:2], long_side=768)
        z_back = decode_depth_m(enc)
        focus_disp, k = 1.0 / 3.0, 50.0
        coc_orig = np.abs(signed_coc_px(z, focus_disp, k))
        coc_back = np.abs(signed_coc_px(z_back, focus_disp, k))
        self.assertLess(float(np.abs(coc_orig - coc_back).max()), 0.01)

    def test_erro_declarado_bate_com_o_medido(self):
        enc = encode_depth(_depth(120, 160, 1.0, 100.0), image_hw=(120, 160), long_side=768)
        k = 50.0
        declarado = quantization_coc_error_px(enc, k)
        disp_erro = float(np.abs(1.0 / _depth(120, 160, 1.0, 100.0) - decode_disparity(enc)).max())
        self.assertLessEqual(disp_erro * k, declarado * 2.5)
        self.assertLess(declarado, 0.01)

    def test_disparidade_ganha_em_cena_NATURAL(self):
        """A razão de existir do módulo, com a condição explícita.

        Cena natural = conteúdo perto, céu no teto do Depth Pro. É o caso que deixou
        24,7% da rota B com a cena útil em menos de 256 níveis de 65535, mediana de 23
        no subgrupo `z_max >= 1000 m`.

        A vantagem NÃO é universal: numa cena linear em `z` — densa no fundo distante —
        a ordem se inverte. Nossas cenas não são assim, e o teste abaixo trava a
        distinção para ninguém generalizar demais.
        """
        z = np.concatenate([np.linspace(1.0, 50.0, 3600),
                            np.full(496, 10000.0)]).astype(np.float32).reshape(64, 64)
        enc = encode_depth(z, image_hw=z.shape[:2], long_side=768)
        niveis_disp = len(np.unique(enc.disparity_u16))

        z01 = (z - z.min()) / (z.max() - z.min())
        niveis_depth = len(np.unique(np.rint(z01 * 65535).astype(np.uint16)))

        self.assertGreater(niveis_disp, 2500)
        self.assertLess(niveis_depth, 400)
        self.assertGreater(niveis_disp, 5 * niveis_depth)

    def test_em_cena_linear_em_z_a_ordem_se_inverte(self):
        """Documenta o limite da escolha, em vez de fingir que ela é universal."""
        z = np.linspace(1.0, 10000.0, 4096, dtype=np.float32).reshape(64, 64)
        niveis_disp = len(np.unique(encode_depth(z, image_hw=z.shape[:2], long_side=768).disparity_u16))
        z01 = (z - z.min()) / (z.max() - z.min())
        niveis_depth = len(np.unique(np.rint(z01 * 65535).astype(np.uint16)))
        self.assertGreater(niveis_depth, niveis_disp)

    def test_resize_usa_vizinho_mais_proximo(self):
        """Interpolar profundidade atravessa descontinuidade e inventa plano
        intermediário: entre 1 m e 20 m, a média é 10,5 m — superfície fantasma."""
        z = np.full((100, 100), 1.0, dtype=np.float32)
        z[:, 50:] = 20.0
        pequeno = resize_depth_nearest(z, 50)
        self.assertEqual(set(np.unique(pequeno).tolist()), {1.0, 20.0})

    def test_grava_a_resolucao_da_IMAGEM_e_a_do_depth(self):
        """`image_hw` é o shape da IMAGEM, não do array de profundidade. Antes o campo
        se chamava `source_hw` e guardava o shape recebido — nome de uma coisa, valor
        de outra —, e `k_at_resolution` calculava o fator errado em silêncio. Medido:
        Depth Pro a 1152x1536 sobre uma foto 3024x4032 dava erro de 2,63x no K."""
        enc = encode_depth(_depth(384, 512), image_hw=(3024, 4032), long_side=768)
        meta = enc.to_metadata()
        self.assertEqual((meta["image_h"], meta["image_w"]), (3024, 4032))
        self.assertEqual((meta["depth_h"], meta["depth_w"]), (384, 512))

    def test_span_de_disparidade_vem_da_resolucao_CHEIA(self):
        """O vizinho mais próximo descarta o pixel mais perto e encolhe o span — e o
        span alimenta `k_for_bokehme`. Medido: objeto de 3x3 px a 0,30 m some na
        decimação e o K do renderer sai 3,4x menor."""
        z = np.full((512, 512), 20.0, dtype=np.float32)
        z[100:103, 100:103] = 0.30                     # objeto pequeno e muito perto
        enc = encode_depth(z, image_hw=(512, 512), long_side=64)
        self.assertAlmostEqual(enc.disparity_max, 1.0 / 0.30, places=4)
        self.assertAlmostEqual(enc.z_min_m, 0.30, places=5)

    def test_faixa_degenerada_rejeita_com_slug(self):
        """`reject`, não `ValueError`: sem slug a falha escapa do histograma de
        motivos, que é o instrumento que calibra todo limiar."""
        with self.assertRaises(SampleRejected):
            encode_depth(np.full((16, 16), 5.0, dtype=np.float32), image_hw=(16, 16))


# ==============================================================================
# Metadados — ausência é erro, não omissão
# ==============================================================================

class Metadata(unittest.TestCase):
    def test_amostra_completa_valida(self):
        validate_metadata(_sample().metadata())

    def test_campo_faltando_levanta(self):
        meta = _sample().metadata()
        del meta["focus_disparity"]
        with self.assertRaises(ValueError) as ctx:
            validate_metadata(meta)
        self.assertIn("focus_disparity", str(ctx.exception))

    def test_k_invalido_levanta(self):
        meta = _sample().metadata()
        meta["k_value"] = 0.0
        with self.assertRaises(ValueError):
            validate_metadata(meta)

    def test_focus_depth_e_derivado_e_consistente(self):
        meta = _sample().metadata()
        self.assertAlmostEqual(1.0 / meta["focus_depth_m"], meta["focus_disparity"], places=9)

    def test_json_serializa_tipos_numpy(self):
        json.loads(_sample().metadata_json())


# ==============================================================================
# Marcação da região em foco — o requisito de poder treinar COM e SEM as refinadas
# ==============================================================================

class MarcacaoDaRegiaoEmFoco(unittest.TestCase):

    def test_sem_a_marcacao_o_metadado_e_REPROVADO(self):
        """O campo novo é obrigatório, e a ausência é erro, não omissão.

        `focus_disparity` é a base do rótulo inteiro. Sem dizer de onde a região veio,
        ele é um número sem proveniência — e medimos que a máscara do BiRefNet acerta o
        plano de foco em 35,2% das amostras.
        """
        meta = _sample().metadata()
        del meta["focus_source"]
        with self.assertRaises(ValueError) as ctx:
            validate_metadata(meta)
        self.assertIn("focus_source", str(ctx.exception))

    def test_cada_campo_novo_e_exigido_individualmente(self):
        """Um a um: um `frozenset` que perdesse uma entrada passaria em silêncio."""
        for campo in ("focus_source", "focus_was_refined", "focus_agreement",
                      "focus_retention_in_region", "focus_region_area_ratio",
                      "focus_retention_h", "focus_retention_w",
                      "focus_retention_window_px"):
            with self.subTest(campo=campo):
                meta = _sample().metadata()
                del meta[campo]
                with self.assertRaises(ValueError) as ctx:
                    validate_metadata(meta)
                self.assertIn(campo, str(ctx.exception))

    def test_amostra_SEM_objeto_focus_reprova(self):
        """`Sample.focus=None` não grava os campos, e o writer barra antes do disco."""
        amostra = _sample()
        amostra.focus = None
        with self.assertRaises(ValueError):
            validate_metadata(amostra.metadata())

    def test_focus_source_fora_do_enum_reprova(self):
        meta = _sample().metadata()
        meta["focus_source"] = "grabcut"
        with self.assertRaises(ValueError):
            validate_metadata(meta)

    def test_was_refined_que_MENTE_sobre_a_origem_reprova(self):
        """É por este booleano que se monta o treino com e sem as refinadas. Se ele
        puder discordar da origem, o filtro seleciona outro conjunto que não o que diz
        selecionar."""
        meta = _sample(focus=_focus(FocusSource.RETENTION_ONLY)).metadata()
        meta["focus_was_refined"] = False
        with self.assertRaises(SampleRejected) as ctx:
            validate_metadata(meta)
        self.assertEqual(ctx.exception.reason, "focus_provenance_inconsistent")

    def test_mask_source_que_nao_e_a_mascara_do_rotulo_reprova(self):
        """A máscara gravada tem que ser a que produziu `focus_disparity`.

        Dizer `retention_only` no foco e `birefnet` na máscara afirma que o arquivo em
        `mask/<id>.png` é a máscara do segmentador — e quem auditasse o rótulo olharia
        a máscara errada, sem nada denunciar.
        """
        meta = _sample(focus=_focus(FocusSource.RETENTION_ONLY)).metadata()
        meta["mask_source"] = MaskSource.BIREFNET.value
        with self.assertRaises(SampleRejected) as ctx:
            validate_metadata(meta)
        self.assertEqual(ctx.exception.reason, "focus_provenance_inconsistent")

    def test_mask_source_refinada_com_foco_do_birefnet_tambem_reprova(self):
        """A incoerência vale nos dois sentidos."""
        meta = _sample().metadata()
        meta["mask_source"] = MaskSource.BIREFNET_REFINED.value
        with self.assertRaises(SampleRejected):
            validate_metadata(meta)

    def test_area_da_regiao_ZERO_reprova(self):
        """Região vazia não define plano de foco: a Eq. 4 não teria pixel."""
        meta = _sample().metadata()
        meta["focus_region_area_ratio"] = 0.0
        with self.assertRaises(ValueError):
            validate_metadata(meta)

    def test_acordo_fora_de_zero_um_reprova(self):
        for valor in (-0.1, 1.5, float("nan")):
            with self.subTest(valor=valor):
                meta = _sample().metadata()
                meta["focus_agreement"] = valor
                with self.assertRaises(ValueError):
                    validate_metadata(meta)

    def test_resolucao_da_retencao_invalida_reprova(self):
        """Inclusive `None`: a chave existir faz o teste de campo obrigatório passar, e a
        resolução da medida continua não existindo."""
        for campo in ("focus_retention_h", "focus_retention_w",
                      "focus_retention_window_px"):
            for valor in (0, -1, None):
                with self.subTest(campo=campo, valor=valor):
                    meta = _sample().metadata()
                    meta[campo] = valor
                    with self.assertRaises(ValueError):
                        validate_metadata(meta)

    def test_retencao_NaN_na_regiao_e_aceita(self):
        """NaN aqui significa "não havia detalhe medível na região", que é diferente de
        "reteve zero". Colapsar as duas afirmaria borrão onde não houve medida."""
        meta = _sample(focus=_focus(retention_in_region=float("nan"))).metadata()
        validate_metadata(meta)

    def test_toda_fonte_de_foco_tem_fonte_de_mascara(self):
        """A ponte é total. Um `FocusSource` novo sem entrada levanta `KeyError` — não
        escolhe um default, que é como `mask_source="automatic"` nasceu."""
        self.assertEqual(set(FOCUS_SOURCE_TO_MASK_SOURCE), set(FocusSource))
        for source in FocusSource:
            self.assertIsInstance(FOCUS_SOURCE_TO_MASK_SOURCE[source], MaskSource)

    def test_was_refined_bate_com_o_do_modulo_de_refino(self):
        """As duas definições não podem divergir — é a regra "cópias divergem"."""
        for source in FocusSource:
            regiao = RefinedFocusRegion(
                mask=np.ones((4, 4), dtype=bool), source=source, agreement=0.5, precision=0.4, iou=0.3,
                retention_in_region=0.8, area_ratio=0.1, initial_mask_was_empty=False)
            registro = FocusRegionRecord.from_region(
                regiao, retention_hw=(4, 4), retention_window_px=3,
                initial_mask_area_ratio=0.1)
            self.assertEqual(registro.was_refined, regiao.was_refined, source.value)

    def test_from_region_nao_segura_a_mascara(self):
        """Um registro de metadado não deve manter um array de 3 megapixels vivo."""
        regiao = RefinedFocusRegion(
            mask=np.ones((8, 8), dtype=bool), source=FocusSource.RETENTION_ONLY,
            agreement=0.0, precision=0.0, iou=0.0, retention_in_region=0.7, area_ratio=0.05,
            initial_mask_was_empty=True)
        registro = FocusRegionRecord.from_region(
            regiao, retention_hw=(8, 8), retention_window_px=3,
            initial_mask_area_ratio=0.0)
        self.assertNotIn("mask", registro.to_metadata())
        self.assertFalse(any(isinstance(v, np.ndarray)
                             for v in registro.to_metadata().values()))


# ==============================================================================
# Writer — não faz aritmética sobre o sinal
# ==============================================================================

class Writer(unittest.TestCase):
    def test_grava_e_reabre(self):
        with tempfile.TemporaryDirectory() as tmp:
            with FileSampleWriter(tmp, split=_split_de("1038", "s1", "s2")) as w:
                w.write(_sample())
            root = Path(tmp)
            self.assertTrue((root / "depth" / "c_0001.png").exists())
            self.assertTrue((root / "mask" / "c_0001.png").exists())
            meta = json.loads((root / "meta" / "c_0001.json").read_text())
            self.assertEqual(meta["k_value"], 18.0)
            self.assertEqual(meta["scene_id"], "1038")

    def test_nao_grava_defocus_map(self):
        """O mapa é derivado no dataloader pela MESMA função da geração. Gravá-lo
        criaria uma segunda fonte de verdade — que é o defeito D1."""
        with tempfile.TemporaryDirectory() as tmp:
            with FileSampleWriter(tmp, split=_split_de("1038", "s1", "s2")) as w:
                w.write(_sample())
            self.assertEqual(list((Path(tmp)).glob("**/*defocus*")), [])

    def test_k_diferente_sobrevive_a_gravacao(self):
        """Regressão direta do D1: o writer não pode normalizar nada."""
        with tempfile.TemporaryDirectory() as tmp:
            with FileSampleWriter(tmp, split=_split_de("1038", "s1", "s2")) as w:
                w.write(_sample("a", "s1", k=5.0))
                w.write(_sample("b", "s2", k=90.0))
            ks = [json.loads(l)["k_value"] for l in
                  (Path(tmp) / "manifest.jsonl").read_text().strip().splitlines()]
            self.assertEqual(sorted(ks), [5.0, 90.0])

    def test_manifesto_carrega_a_resolucao_do_K(self):
        """K é um número EM PIXEL: sem a resolução na mesma linha, o manifesto não diz
        o que o K significa, e comparar K entre rotas vira comparação sem escala."""
        with tempfile.TemporaryDirectory() as tmp:
            with FileSampleWriter(tmp, split=_split_de("1038", "s1")) as w:
                w.write(_sample("a", "s1"))
            linha = json.loads(
                (Path(tmp) / "manifest.jsonl").read_text().strip().splitlines()[0])
            for campo in ("image_h", "image_w", "depth_h", "depth_w"):
                with self.subTest(campo=campo):
                    self.assertIn(campo, linha)
            self.assertEqual((linha["image_h"], linha["image_w"]), (64, 96))

    def test_manifesto_carrega_a_marcacao_do_refinamento(self):
        """Sem isto, montar o treino sem as amostras refinadas exigiria abrir 22.990
        JSONs — e o manifesto existe justamente para ser a varredura rápida."""
        with tempfile.TemporaryDirectory() as tmp:
            with FileSampleWriter(tmp, split=_split_de("1038", "s1")) as w:
                w.write(_sample("a", "s1", focus=_focus(FocusSource.RETENTION_ONLY)))
            linha = json.loads(
                (Path(tmp) / "manifest.jsonl").read_text().strip().splitlines()[0])
        self.assertEqual(linha["focus_source"], "retention_only")
        self.assertTrue(linha["focus_was_refined"])
        self.assertEqual(linha["mask_source"], "retention_only")
        for campo in ("focus_agreement", "focus_retention_in_region",
                      "focus_region_area_ratio", "focus_retention_h",
                      "focus_retention_w"):
            with self.subTest(campo=campo):
                self.assertIn(campo, linha)

    def test_retomada_le_o_manifesto(self):
        with tempfile.TemporaryDirectory() as tmp:
            with FileSampleWriter(tmp, split=_split_de("1038", "s1", "s2")) as w:
                w.write(_sample("c_0001"))
            self.assertEqual(FileSampleWriter(tmp, split=_split_de("1038")).completed_ids(), {"c_0001"})

    def test_metadado_incompleto_nao_chega_ao_disco(self):
        with tempfile.TemporaryDirectory() as tmp:
            s = _sample()
            object.__setattr__(s.control, "depth_backend", "depth_anything")
            with self.assertRaises(SampleRejected):
                FileSampleWriter(tmp, split=_split_de("1038")).write(s)
            self.assertEqual(list((Path(tmp) / "meta").glob("*.json")), [])

    def test_orcamento_de_disco(self):
        b = estimate_disk_budget(20554, 768, generates_image=False, image_megapixels=3.0)
        self.assertEqual(b["generated_image_gb"], 0.0)
        self.assertLess(b["total_gb"], 20.0)


# ==============================================================================
# Split por cena — o risco que cresceu de tamanho
# ==============================================================================

class Split(unittest.TestCase):
    def test_deterministico_entre_processos(self):
        """`hash()` do Python é randomizado por PYTHONHASHSEED: usá-lo faria a
        validação de ontem virar treino hoje."""
        cenas = [f"cena_{i}" for i in range(500)]
        a = build_scene_split(cenas, val_fraction=0.1)
        b = build_scene_split(list(reversed(cenas)), val_fraction=0.1)
        self.assertEqual(a.assignment, b.assignment)

    def test_fracao_aproximada(self):
        split = build_scene_split([f"c{i}" for i in range(2000)], val_fraction=0.05)
        frac = split.counts().get("val", 0) / 2000
        self.assertAlmostEqual(frac, 0.05, delta=0.02)

    def test_detecta_vazamento_por_cena(self):
        """21 aberturas da mesma cena: split por imagem colocaria a mesma cena nos
        dois lados. Por cena, não."""
        split = build_scene_split(["cena_A", "cena_B"], val_fraction=0.4)
        rows = ([{"sample_id": f"A_{i}", "scene_id": "cena_A", "split": split.of("cena_A")}
                 for i in range(21)]
                + [{"sample_id": f"B_{i}", "scene_id": "cena_B", "split": split.of("cena_B")}
                   for i in range(5)])
        rep = check_no_leak(rows, split)
        self.assertTrue(rep.clean)
        self.assertEqual(rep.samples_per_scene["cena_A"], 21)
        self.assertIn("21", rep.summary())

    def test_amostra_sem_cena_e_denunciada(self):
        split = build_scene_split(["x"], val_fraction=0.1)
        rep = check_no_leak([{"sample_id": "s1"}], split)
        self.assertFalse(rep.clean)
        self.assertEqual(rep.samples_without_scene, ["s1"])

    def test_cena_fora_do_split_e_denunciada(self):
        split = build_scene_split(["x"], val_fraction=0.1)
        rep = check_no_leak([{"sample_id": "s1", "scene_id": "desconhecida", "split": "train"}], split)
        self.assertFalse(rep.clean)
        self.assertEqual(rep.scenes_missing_from_split, ["desconhecida"])

    def test_gate_de_vazamento_PODE_reprovar(self):
        """A versão anterior era tautológica: `split.of()` é função pura, então
        `len(v) > 1` era impossível por construção e o ramo de detecção era código
        morto. Agora a linha carrega o split com que foi GRAVADA."""
        split = build_scene_split(["cena_A", "cena_B"], val_fraction=0.4)
        rows = [{"sample_id": "a1", "scene_id": "cena_A", "split": "train"},
                {"sample_id": "a2", "scene_id": "cena_A", "split": "val"}]
        rep = check_no_leak(rows, split)
        self.assertFalse(rep.clean)
        self.assertEqual(rep.scenes_in_both, ["cena_A"])
        self.assertIn("VAZAMENTO", rep.summary())

    def test_split_gravado_divergindo_do_materializado_e_pego(self):
        split = build_scene_split(["c1"], val_fraction=0.1)
        errado = "val" if split.of("c1") == "train" else "train"
        rep = check_no_leak([{"sample_id": "s", "scene_id": "c1", "split": errado}], split)
        self.assertFalse(rep.clean)
        self.assertEqual(rep.scenes_disagreeing_with_split, ["c1"])

    def test_origem_sem_split_NAO_vai_para_treino_em_silencio(self):
        """`(source or "")` mandava dado ausente para TREINO — o lado que infla a
        métrica — e o gate de vazamento não pegava, porque a cena ESTAVA no split."""
        for ausente in (None, "", "desconhecido"):
            with self.assertRaises(ValueError):
                split_from_source({"c1": "train", "c2": ausente})

    def test_respeita_o_split_da_origem(self):
        """A RealBokeh_3MP já separa train/test/validation por cena — reusar é melhor
        que sortear."""
        split = split_from_source({"a": "train", "b": "test", "c": "validation", "d": "train"})
        self.assertEqual(split.of("a"), "train")
        self.assertEqual(split.of("b"), "val")
        self.assertEqual(split.of("c"), "val")
        self.assertEqual(split.counts(), {"train": 2, "val": 2})

    def test_salva_e_recarrega(self):
        with tempfile.TemporaryDirectory() as tmp:
            split = build_scene_split([f"c{i}" for i in range(50)], val_fraction=0.2)
            path = split.save(Path(tmp) / "split.json")
            from dataio.split import SceneSplit
            self.assertEqual(SceneSplit.load(path).assignment, split.assignment)


if __name__ == "__main__":
    unittest.main(verbosity=2)
