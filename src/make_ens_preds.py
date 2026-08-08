"""fv2 greedy 가중 앙상블의 검증 예측 파일 생성 (캘리브레이션 백테스트용)."""
import numpy as np

a = np.load("out/cat_fv2_val_preds.npz")
s3 = np.load("out/cat_fv2s3_val_preds.npz")
s2 = np.load("out/cat_fv2s2_val_preds.npz")
ens = 0.5 * a["pred"] + (1 / 3) * s3["pred"] + (1 / 6) * s2["pred"]
np.savez_compressed("out/cat_fv2ens_val_preds.npz", y=a["y"], pred=ens)
print("saved out/cat_fv2ens_val_preds.npz")
