"""RT1 cutoff·복귀·행 독립 최소 검문."""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import roster_transition as rt


def main():
    hist = pd.DataFrame({
        "row_id": [1, 2, 3, 4, 5],
        "pitcher_id": [10, 10, 10, 20, 20],
        "season": [2020, 2020, 2022, 2022, 2023],
        "game_type": ["F", "F", "F", "R", "R"],
    })
    tab = rt.build_table(hist)
    q = pd.DataFrame({
        "row_id": [100, 101, 102], "pitcher_id": [10, 20, 30],
        "season": [2023, 2024, 2024], "game_type": ["R", "R", "R"],
    })
    out, cols = rt.add_features(q, tab)
    assert cols == rt.COLS
    assert out.loc[0, "rt_f_to_r"] == 1 and out.loc[0, "rt_gap_years"] == 1
    assert out.loc[1, "rt_same_cont"] == 1 and out.loc[1, "rt_prior_r_log"] > 0
    assert out.loc[2, "rt_seen_before"] == 0 and out.loc[2, "rt_gap_years"] == -1

    # 같은 테스트 배치의 다른 행을 바꾸거나 추가해도 첫 행 피처는 불변이다.
    q2 = pd.concat([q, pd.DataFrame({"row_id": [103], "pitcher_id": [10],
                                    "season": [2023], "game_type": ["F"]})],
                   ignore_index=True)
    out2, _ = rt.add_features(q2, tab)
    pd.testing.assert_series_equal(out.loc[0, cols], out2.loc[0, cols],
                                   check_names=False)
    print("roster transition cutoff/row-independence: OK")


if __name__ == "__main__":
    main()
