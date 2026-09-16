import numpy as np, glob, os, cv2
D="/workspace/vp/temp_depth_maps"
for dp in sorted(glob.glob(D+"/timseiz*_depth.npy")):
    base=dp[:-10]
    mk=base+"_mask.npy"
    depth=np.load(dp).astype(np.float32)
    # o laco roda a 512 no lado maior
    H,W=depth.shape; s=512.0/max(H,W); w=(int(W*s)//16)*16; h=(int(H*s)//16)*16
    depth=cv2.resize(depth,(w,h),interpolation=cv2.INTER_LINEAR)
    disp=1.0/np.where(depth>0,depth,np.finfo(np.float32).max)
    m=np.load(mk).astype(np.float32); m=cv2.resize(m,(w,h),interpolation=cv2.INTER_LINEAR)
    frac=float(np.mean(m>127))
    fc_mask=float(np.median(disp[m>127])) if (m>127).any() else float("nan")
    fc_centro=float(disp[h//2,w//2])
    print(os.path.basename(base))
    print("   depth m: min=%.2f med=%.2f max=%.2f | disp: min=%.4f med=%.4f max=%.4f" % (
        depth.min(), np.median(depth), depth.max(), disp.min(), np.median(disp), disp.max()))
    print("   mascara: %.1f%% dos pixels | disp_focus MASCARA=%.4f  CENTRO=%.4f" % (frac*100, fc_mask, fc_centro))
    for nome,fc in [("mascara",fc_mask),("centro",fc_centro)]:
        d=np.abs(disp-fc)
        for K in [1,10,50,100,300]:
            dm=np.clip(K*d/100.0,0,1)
            print("      %s K=%-4d mapa: media=%.3f max=%.3f frac>0.5=%.3f" % (nome,K,dm.mean(),dm.max(),np.mean(dm>0.5)))
