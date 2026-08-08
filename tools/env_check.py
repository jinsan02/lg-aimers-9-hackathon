"""머신마다 같은 pkl + 같은 test.csv 가 **같은 예측**을 내는가 (E154).

A100 은 pandas 2.3.3 / numpy 2.2.6 / sklearn 1.7.2 인데 평가 서버와 노트북은
pandas 2.0.3 / numpy 1.26.4 / sklearn 1.8.0 이다. 학습은 A100 에서, 추론은
평가 서버에서 일어나므로 **피처 계산이 두 환경에서 갈리면** 조용히 점수가 샌다.

v14/v15 가 예상대로 나왔으니 큰 문제는 없다는 방증이지만, 방증은 측정이 아니다.
여기서 실제로 같은 값이 나오는지 확인한다.

실행: python tools/env_check.py [pkl경로]
"""

import sys

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, "src")


def main():
    pkl = sys.argv[1] if len(sys.argv) > 1 else "model/cat_v14f_s42.pkl"
    import fpipe
    import sklearn
    print(f"pandas {pd.__version__} numpy {np.__version__} "
          f"sklearn {sklearn.__version__}")
    t = pd.read_csv("data/test.csv", encoding="utf-8-sig")
    p = fpipe.predict(joblib.load(pkl), t)
    print("PRED " + " ".join(f"{v:.12f}" for v in p))
    return 0


if __name__ == "__main__":
    sys.exit(main())
