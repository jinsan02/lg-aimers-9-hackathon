"""환경 검증 — 어느 머신에서든 실행해 라이브러리/GPU 상태 확인."""
import importlib

for m in ["numpy", "pandas", "sklearn", "joblib", "lightgbm", "xgboost",
          "catboost", "torch"]:
    try:
        mod = importlib.import_module(m)
        extra = ""
        if m == "torch":
            extra = f" cuda={mod.cuda.is_available()}"
        print(f"  {m:12s} {getattr(mod, '__version__', '?')}{extra}")
    except ImportError:
        print(f"  {m:12s} (없음)")

try:
    import xgboost as xgb
    import numpy as np
    X = np.random.rand(500, 5)
    y = (np.random.rand(500) > 0.5).astype(int)
    xgb.train({"device": "cuda", "objective": "binary:logistic",
               "verbosity": 0}, xgb.DMatrix(X, y), num_boost_round=3)
    print("  xgboost GPU  OK")
except Exception as e:
    print(f"  xgboost GPU  실패: {type(e).__name__}: {e}")
