import time, sys, torch
from genfocus_train.config import load_config
from genfocus_train.backbone import create_backbone
from genfocus_train.models import BokehNet, flow_matching_loss
from genfocus_train.data import build_dataset, build_dataloader, DatasetRuntimeConfig
from genfocus_train.trainer import _make_train_batch

cfg = load_config(sys.argv[1]); sc = cfg.data.bokeh
bb = create_backbone(cfg.model, "bokeh"); bb.load(torch.bfloat16, torch.device("cuda"))
te = bb.precompute_text("an excellent photo with a large aperture",
                        torch.device("cuda"), torch.bfloat16)
bb.release_text_encoders()
m = BokehNet(bb, te)
# CRITICO: o transformer_forward do Genfocus checa `self.training and
# self.gradient_checkpointing` (flux.py:426 e 445). Sem o .train() o
# checkpointing NAO age, e a medida de VRAM sai muito acima do treino real.
# O trainer faz isso em trainer.py:604; a bancada tem de fazer igual.
bb.transformer.train()
rt = DatasetRuntimeConfig(
    image_size=sc.image_size, train=True, defocus_source=sc.defocus_source,
    max_coc=cfg.model.max_coc, min_calibration_ssim=sc.min_calibration_ssim,
    geo_condition=sc.geo_condition, geo_escalares=sc.geo_escalares,
    geo_constantes=sc.geo_constantes, geo_field=sc.geo_field,
    geo_sem_escalares="pula")
ds = build_dataset("bokeh", sc, rt)
dl = build_dataloader(ds, batch_size=1, num_workers=2, pin_memory=True, shuffle=True)
it = iter(dl); torch.cuda.reset_peak_memory_stats(); ts = []
for i in range(6):
    b = next(it)
    b = {k: (v.cuda() if isinstance(v, torch.Tensor) else v) for k, v in b.items()}
    torch.cuda.synchronize(); t0 = time.time()
    o = _make_train_batch(model=m, stage="bokeh", batch=b,
                          occlusion_lambda=cfg.runtime.occlusion_lambda,
                          occlusion_pool=cfg.runtime.occlusion_pool,
                          geo_branches=cfg.runtime.geo_branches)
    flow_matching_loss(o.prediction, o.target, o.weight).backward()
    torch.cuda.synchronize(); ts.append(time.time() - t0)
med = sum(ts[3:]) / 3
print("RESULT tokens=%s s_micro=%.3f s_step_accum16=%.1f vram_gb=%.1f branches=%s lambda=%s"
      % (tuple(o.prediction.shape), med, 16 * med,
         torch.cuda.max_memory_allocated() / 1e9,
         cfg.runtime.geo_branches, cfg.runtime.occlusion_lambda))
