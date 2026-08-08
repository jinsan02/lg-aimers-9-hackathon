"""KBO 도메인 기전이 데이터에 실재하는지 일괄 검정 (E111).

컬럼명만 보고 세운 야구 가설들을 **효과 크기 순으로** 줄 세운다.
피처를 만들기 전에 여기서 걸러야 E101/E103(트랙맨) 같은 헛수고를 안 한다.

읽는 법 - 두 숫자를 본다.
  raw   : 그 상황의 성공률 - 전체 성공률.  (투수 구성 차이가 섞여 있다)
  adj   : 투수 고정효과를 뺀 뒤의 차이. **이게 진짜 상황 효과**다.
          같은 투수가 그 상황에 들어갔을 때 얼마나 달라지는가.
  n     : 표본. adj 의 표준오차는 대략 0.5/sqrt(n).

adj 가 0.003 미만이면 Brier 로 환산해 의미 없는 크기다
  (BSS 이득 상한 ~= n_share * adj^2 / (r(1-r)) * 1e5).

실행: python tools/domain_probe.py
"""

import sys

import numpy as np
import pandas as pd

DATA = "./data"


def main():
    use = ["season", "game_type", "pitcher_id", "balls_before", "strikes_before",
           "outs_before", "inning", "game_month", "game_dayofweek", "top_bottom",
           "runner_on_1b", "runner_on_2b", "runner_on_3b", "num_runners_on",
           "score_diff_pitcher_team", "li", "pitcher_hand", "batter_hand",
           "asof_pitcher_n", "asof_batter_n", "asof_batter_success_rate",
           "control_success"]
    df = pd.read_csv(f"{DATA}/train.csv", usecols=use)
    df = df[(df.game_type == "R") & (df.season >= 2022)].copy()
    y = df["control_success"].to_numpy(np.float64)
    r = y.mean()
    base = r * (1 - r)
    print(f"R리그 2022~2024 {len(df):,}행 | 성공률 {r:.4f}\n")

    # 투수 고정효과 제거: 각 행에서 그 투수의 평균을 뺀 잔차
    pm = df.groupby("pitcher_id")["control_success"].transform("mean").to_numpy()
    resid = y - pm

    b1 = df.runner_on_1b.to_numpy() > 0
    b2 = df.runner_on_2b.to_numpy() > 0
    b3 = df.runner_on_3b.to_numpy() > 0
    ou = df.outs_before.to_numpy()
    inn = df.inning.to_numpy()
    sd = np.abs(df.score_diff_pitcher_team.to_numpy())
    mo = df.game_month.to_numpy()
    dw = df.game_dayofweek.to_numpy()

    TESTS = [
        # (이름, 마스크, 한 줄 기전)
        ("거르기 조건: 1루 열림 & 득점권", ~b1 & (b2 | b3),
         "1루가 비면 강타자를 걸러도 손해가 없다 -> 의도적 존 밖 = 라벨상 실패"),
        ("거르기 조건 & 2아웃 & 접전", ~b1 & (b2 | b3) & (ou == 2) & (sd <= 2),
         "가장 전형적인 고의성 승부회피 국면"),
        ("퀵모션: 1루만 주자", b1 & ~b2 & ~b3,
         "도루 견제로 슬라이드스텝 -> 하체 사용이 줄어 제구 저하"),
        ("득점권 주자 (2·3루)", b2 | b3,
         "실점 직결 -> 한가운데 회피, 유인구 증가"),
        ("만루", b1 & b2 & b3,
         "거를 수 없다 -> 무조건 존 안, 한가운데 위험"),
        ("3루 주자 & 2아웃 미만", b3 & (ou < 2),
         "희생플라이 경계 -> 낮게 던지려다 존 이탈"),
        ("주자 없음", ~(b1 | b2 | b3), "와인드업 가능 = 기준 상태"),
        ("2아웃", ou == 2, "이닝 종료 압박"),
        ("선발 3순회 (6회 이상)", inn >= 6, "피로 + 타자가 이미 두 번 봄"),
        ("연장 (10회 이상)", inn >= 10, "불펜 소진, 급조 등판"),
        ("1회", inn == 1, "몸이 덜 풀림"),
        ("점수차 5점 이상", sd >= 5, "가비지 -> 스트라이크 꽂기"),
        ("접전 1점차", sd <= 1, "고압"),
        ("4월 (개막 추위)", mo == 4, "저온 -> 그립/감각 저하"),
        ("7~8월 (혹서·장마)", (mo >= 7) & (mo <= 8), "땀/습도 -> 미끄러짐"),
        ("9~10월 (시즌 말)", mo >= 9, "누적 피로"),
        ("일요일", dw == 6, "월 휴식 전 6연전 마지막 = 불펜 소진"),
        ("화요일", dw == 1, "월 휴식 직후 = 가장 신선"),
        ("동일손 매치업", (df.pitcher_hand.to_numpy()
                     == df.batter_hand.to_numpy()), "플래툰 우위"),
        ("신인 투수 (통산 1000구 미만)",
         df.asof_pitcher_n.fillna(0).to_numpy() < 1000, "경험 부족"),
    ]

    print(f"{'가설':<26}{'n':>10}{'비중':>7}{'raw':>9}{'adj':>9}"
          f"{'BSS상한':>9}   기전")
    out = []
    for name, mask, why in TESTS:
        n = int(mask.sum())
        if n < 5000:
            continue
        share = n / len(df)
        raw = y[mask].mean() - r
        adj = resid[mask].mean() - resid.mean()
        # 이 이분 상황만으로 얻을 수 있는 BSS 상한 (두 그룹 평균으로 예측할 때)
        gain = (adj ** 2) * share / (1 - share + 1e-12) / base * 1e5
        out.append((abs(adj), name, n, share, raw, adj, gain, why))
    for _, name, n, share, raw, adj, gain, why in sorted(out, reverse=True):
        print(f"{name:<26}{n:>10,}{share:>7.1%}{raw:>+9.4f}{adj:>+9.4f}"
              f"{gain:>9.1f}   {why}")

    # ---- 강타자 거르기: 상황이 '타자 위협도'의 기울기를 키우는가
    # asof_batter_success_rate 가 낮다 = 이 타자에게는 투수들이 존을 피해왔다 = 강타자.
    # 가설이 맞으면 1루가 열렸을 때 그 기울기가 **더 가팔라야** 한다.
    print("\n=== 강타자 거르기: 타자 위협도의 기울기가 상황에 따라 달라지는가 ===")
    bd = df["asof_batter_success_rate"].to_numpy(np.float64)
    ok = np.isfinite(bd) & (df["asof_batter_n"].fillna(0).to_numpy() >= 200)
    ctx = {
        "1루 열림 & 득점권 (거르기 가능)": ~b1 & (b2 | b3),
        "1루 막힘 (거를 수 없음)": b1,
        "주자 없음": ~(b1 | b2 | b3),
        "  └ 위 중 2아웃·접전만": ~b1 & (b2 | b3) & (ou == 2) & (sd <= 2),
    }
    print(f"{'국면':<34}{'n':>10}{'기울기':>10}{'표준오차':>10}{'t':>7}")
    for nm, m in ctx.items():
        mm = m & ok
        n = int(mm.sum())
        if n < 5000:
            continue
        x = bd[mm]
        v = resid[mm]
        x = x - x.mean()
        den = (x * x).sum()
        sl = float((x * v).sum() / den)
        rs = v - sl * x
        se = float(np.sqrt((rs @ rs) / max(n - 2, 1) / den))
        print(f"{nm:<34}{n:>10,}{sl:>10.3f}{se:>10.3f}{sl / se:>7.1f}")
    print("  기울기가 클수록 '타자가 강할수록 제구 실패'가 심하다는 뜻.")
    print("  거르기 가능 국면에서 더 가파르면 -> 의도적 승부회피가 라벨을 오염시킨다.")

    print("\n※ adj = 투수 고정효과 제거 후 상황 효과. |adj| < 0.003 이면 무의미.")
    print("※ 상위 항목이라도 트리가 이미 raw 컬럼에서 찾아냈을 수 있다 -")
    print("   교차항으로 만들 가치가 있는 건 '연속 스타일 x 이산 상황' 형태뿐이다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
