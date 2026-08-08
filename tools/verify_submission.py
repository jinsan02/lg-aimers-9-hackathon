"""제출 zip 종합 검증 - 올리기 전 반드시 통과시킨다.

08-06 사고 3건에서 나온 체크리스트:
  1) argparse help 줄바꿈 -> SyntaxError 로 배치가 통째로 죽었는데 grep이 가림
  2) 원격에 ./model 없어서 저장 실패, 역시 grep이 가림
  3) --te-dev 가드가 잘못된 컬럼명을 봐서 **기능이 통째로 무효**였는데 아무도 몰랐음

검사 항목:
  A. zip 구조 - script.py가 루트에 있고, 상위 폴더가 없고, 필요한 모듈이 다 들어있나
  B. 모델 무결성 - 전부 로드되나, 학습 피처를 추론에서 전부 재현하나
  C. 실제 추론 - 5행 샘플로 완주, 예측이 [0,1] 범위이고 상수가 아닌가
  D. 규모 추론 - 245,789행에서 6스레드 기준 소요 시간 (한도 600초)
  E. 제출 규격 - 파일명 30자, 용량, row_id 순서/컬럼

사용: python tools/verify_submission.py submissions/blendv6_XXXX.zip
"""

import os
import subprocess
import sys
import tempfile
import time
import zipfile

import numpy as np
import pandas as pd

LIMIT_SEC = 600
N_EVAL = 245789


def fail(msg):
    print(f"  [FAIL] {msg}")
    return 1


def ok(msg):
    print(f"  [ok] {msg}")
    return 0


def main(zip_path):
    bad = 0
    name = os.path.basename(zip_path)
    print(f"=== A. zip 구조 : {name} ===")
    bad += fail(f"파일명 {len(name)}자 (30자 초과)") if len(name) > 30 \
        else ok(f"파일명 {len(name)}자")
    size = os.path.getsize(zip_path) / 1e6
    bad += ok(f"용량 {size:.1f} MB")
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
    bad += ok("script.py 루트에 존재") if "script.py" in names \
        else fail("script.py가 zip 루트에 없음")
    tops = {n.split("/")[0] for n in names}
    extra = tops - {"script.py", "requirements.txt", "model",
                    "features.py", "target_enc.py", "season_std.py", "skill.py"}
    bad += fail(f"예상 못한 최상위 항목: {extra}") if extra else ok("최상위 구성 정상")
    n_model = sum(1 for n in names if n.startswith("model/"))
    bad += ok(f"모델 {n_model}개 동봉")

    print("\n=== B~E. 압축 해제 후 실제 실행 ===")
    with tempfile.TemporaryDirectory() as td:
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(td)
        os.makedirs(f"{td}/data", exist_ok=True)
        for f in ("test.csv", "sample_submission.csv"):
            pd.read_csv(f"./data/{f}", encoding="utf-8-sig").to_csv(
                f"{td}/data/{f}", index=False, encoding="utf-8-sig")

        py = os.path.abspath(".venv/Scripts/python.exe")
        env = dict(os.environ, PYTHONIOENCODING="utf-8", OMP_NUM_THREADS="6")
        t = time.time()
        r = subprocess.run([py, "script.py"], cwd=td, env=env,
                           capture_output=True, text=True, encoding="utf-8")
        if r.returncode != 0:
            print(r.stdout[-2000:] if r.stdout else "")
            print(r.stderr[-2000:] if r.stderr else "")
            return fail(f"script.py 실행 실패 (exit {r.returncode})")
        bad += ok(f"script.py 완주 ({time.time() - t:.1f}s, 5행)")

        out = f"{td}/output/submission.csv"
        bad += ok("output/submission.csv 생성") if os.path.exists(out) \
            else fail("output/submission.csv 없음")
        sub = pd.read_csv(out)
        ref = pd.read_csv("./data/sample_submission.csv", encoding="utf-8-sig")
        bad += ok("컬럼/순서 일치") if list(sub.columns) == list(ref.columns) \
            and sub.row_id.tolist() == ref.row_id.tolist() \
            else fail("row_id 순서 또는 컬럼 불일치")
        p = sub["control_success"].to_numpy()
        bad += fail(f"예측이 [0,1] 밖: {p.min():.4f}~{p.max():.4f}") \
            if p.min() < 0 or p.max() > 1 else ok(f"예측 범위 {p.min():.4f}~{p.max():.4f}")
        bad += fail("예측이 상수 (모델이 동작하지 않았을 수 있음)") \
            if np.allclose(p, p[0]) else ok("예측이 상수 아님")

        # D. 규모 추론
        print("\n=== D. 평가 규모 추론 시간 (6스레드) ===")
        cols = pd.read_csv("./data/test.csv", encoding="utf-8-sig", nrows=0).columns
        big = pd.read_csv("./data/train.csv", encoding="utf-8-sig",
                          usecols=[c for c in cols if c != "row_id"])
        big = big[big.season == 2024].sample(N_EVAL, replace=True, random_state=0)
        big = big.reset_index(drop=True)
        big["season"] = 2025
        big.insert(0, "row_id", np.arange(N_EVAL))
        big.to_csv(f"{td}/data/test.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame({"row_id": np.arange(N_EVAL),
                      "control_success": 0.5}).to_csv(
            f"{td}/data/sample_submission.csv", index=False, encoding="utf-8-sig")
        t = time.time()
        r = subprocess.run([py, "script.py"], cwd=td, env=env,
                           capture_output=True, text=True, encoding="utf-8")
        el = time.time() - t
        if r.returncode != 0:
            print(r.stderr[-1500:] if r.stderr else "")
            return fail("대규모 추론 실패")
        bad += fail(f"추론 {el:.0f}초 — 한도 {LIMIT_SEC}초 초과") if el > LIMIT_SEC \
            else ok(f"추론 {el:.0f}초 (한도 {LIMIT_SEC}초의 {el / LIMIT_SEC * 100:.1f}%)")
        q = pd.read_csv(f"{td}/output/submission.csv")["control_success"].to_numpy()
        bad += ok(f"{N_EVAL:,}행 예측 평균 {q.mean():.4f} 표준편차 {q.std():.4f}")

    print(f"\n{'=' * 46}\n{'통과 - 제출 가능' if bad == 0 else f'실패 {bad}건 - 제출 금지'}")
    return bad


if __name__ == "__main__":
    sys.exit(1 if main(sys.argv[1]) else 0)
