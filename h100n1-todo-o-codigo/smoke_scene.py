"""Exercita o caminho top_k_mode=scene, que nunca rodou. Sem GPU."""
import sys, time
sys.path.insert(0, "/workspace/retreinar-deblur")
from genfocus_train.config import load_config, stage_config
from genfocus_train.data import build_dataset, DatasetRuntimeConfig

for cfgp in ["configs/ab15k_controle.yaml", "configs/ab15k_scene.yaml"]:
    c = load_config(f"/workspace/retreinar-deblur/{cfgp}")
    s = stage_config(c, "deblur")
    print(f"\n### {cfgp}  top_k_mode={s.top_k_mode}")
    t0 = time.time()
    ds = build_dataset("deblur", s, DatasetRuntimeConfig(
        image_size=s.image_size, train=s.augment, scale_mode=s.scale_mode,
        top_k_mode=s.top_k_mode, scene_key=s.scene_key))
    print(f"  len(dataset) = {len(ds)}   ({time.time()-t0:.0f}s)")
    it = ds[0]
    print(f"  chaves: {sorted(it.keys())}")
    print(f"  blurry {tuple(it['blurry_image'].shape)} {it['blurry_image'].dtype}"
          f"  range [{it['blurry_image'].min():.2f}, {it['blurry_image'].max():.2f}]")
    print(f"  full_seq_len = {it['full_seq_len'].item()}")
    # duas leituras do mesmo indice: em scene, a linha sorteada deve poder variar
    ids = {ds[0]["id"] for _ in range(6)}
    print(f"  id(s) vistos em 6 leituras do indice 0: {len(ids)} distinto(s) -> {list(ids)[:3]}")
