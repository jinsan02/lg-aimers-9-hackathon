"""체크포인트에서 numpy 스칼라 제거 (weights_only=True 호환화). 일회용."""
import sys

import torch

p = sys.argv[1]
ckpt = torch.load(p, map_location="cpu", weights_only=False)
ckpt["val_bss"] = float(ckpt["val_bss"])
ckpt["epoch"] = int(ckpt["epoch"])
torch.save(ckpt, p)
print("cleaned:", ckpt["val_bss"], ckpt["epoch"])
