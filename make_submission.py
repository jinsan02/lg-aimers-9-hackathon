"""제출 zip 생성.

실행: python make_submission.py [rf|tabm]
산출: submissions/submit_<variant>_YYYYMMDD_HHMM.zip
"""

import os
import sys
import zipfile
from datetime import datetime

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(ROOT, "submissions")

# variant: [(로컬 경로, zip 내 경로), ...]
VARIANTS = {
    "rf": [
        ("src/script.py", "script.py"),
        ("requirements.txt", "requirements.txt"),
        ("model/rf.pkl", "model/rf.pkl"),
    ],
    "tabm": [
        ("src/script_tabm.py", "script.py"),
        ("src/tabm_reference.py", "tabm_reference.py"),
        ("src/rtdl_num_embeddings.py", "rtdl_num_embeddings.py"),
        ("requirements.txt", "requirements.txt"),
        ("model/prep_v1.pkl", "model/prep_v1.pkl"),
        ("model/tabm_v1_best.pt", "model/tabm_v1_best.pt"),
    ],
    "lgbm": [
        ("src/script_lgbm.py", "script.py"),
        ("requirements_lgbm.txt", "requirements.txt"),
        ("model/lgbm_v3.pkl", "model/lgbm_v3.pkl"),
    ],
    "cat_ens": [
        ("src/script_cat_ens.py", "script.py"),
        ("requirements_cat.txt", "requirements.txt"),
        ("model/cat_l2_3.pkl", "model/cat_l2_3.pkl"),
        ("model/cat_s1.pkl", "model/cat_s1.pkl"),
        ("model/cat_s2.pkl", "model/cat_s2.pkl"),
        ("model/cat_s3.pkl", "model/cat_s3.pkl"),
    ],
    "catfv2": [
        ("src/script_cat_fv2.py", "script.py"),
        ("src/features.py", "features.py"),
        ("requirements_cat.txt", "requirements.txt"),
        ("model/cat_fv2.pkl", "model/cat_fv2.pkl"),
        ("model/cat_fv2s2.pkl", "model/cat_fv2s2.pkl"),
        ("model/cat_fv2s3.pkl", "model/cat_fv2s3.pkl"),
    ],
    # v5: lr 0.02(E92) + 시즌 expanding 타깃인코딩(E91) + 2024×3 refit(E85).
    # 모델 목록은 tools/greedy_weights.py 결과에 맞춰 갱신할 것.
    "blendv5": [
        ("src/script_blend_v5.py", "script.py"),
        ("src/features.py", "features.py"),
        ("src/target_enc.py", "target_enc.py"),
        ("requirements_blend.txt", "requirements.txt"),
    ] + [(f"model/{m}.pkl", f"model/{m}.pkl") for m in [
        "cat_n6_d6", "cat_w3_hs_d7_l10_r0.08", "cat_w3_hs_d8_l3_r0.05",
        "cat_w3_hs_d8_l10_r0.05", "cat_n3_d8l10", "cat_n4_s3", "cat_n5_s4",
        "cat_n_s5", "cat_n_s6", "cat_n_s8", "cat_n_s9", "cat_n_s11",
        "cat_n_s21"]],
    # blendv3(LB 861.66)와 동일 구성·동일 greedy 가중, refit만 2024×3 (E85 순수 A/B)
    "blendw3v3": [
        ("src/script_blend_w3v3.py", "script.py"),
        ("src/features.py", "features.py"),
        ("requirements_blend.txt", "requirements.txt"),
        ("model/cat_w3_hs_d7_l10_r0.08.pkl", "model/cat_w3_hs_d7_l10_r0.08.pkl"),
        ("model/cat_w3_hs_d8_l3_r0.05.pkl", "model/cat_w3_hs_d8_l3_r0.05.pkl"),
        ("model/cat_w3_hs_d8_l10_r0.05.pkl", "model/cat_w3_hs_d8_l10_r0.05.pkl"),
        ("model/cat_w3_fv2.pkl", "model/cat_w3_fv2.pkl"),
        ("model/cat_w3_fv2s3.pkl", "model/cat_w3_fv2s3.pkl"),
        ("model/cat_w3_fv2s4.pkl", "model/cat_w3_fv2s4.pkl"),
        ("model/xgb_w3_fv2x.pkl", "model/xgb_w3_fv2x.pkl"),
    ],
    "blendw3": [
        ("src/script_blend_w3.py", "script.py"),
        ("src/features.py", "features.py"),
        ("src/pitcher_role.py", "pitcher_role.py"),
        ("src/manager_feat.py", "manager_feat.py"),
        ("src/rules.py", "rules.py"),
        ("requirements_blend.txt", "requirements.txt"),
        ("model/cat_w3_hs_d7_l10_r0.08.pkl", "model/cat_w3_hs_d7_l10_r0.08.pkl"),
        ("model/cat_w3_fv2.pkl", "model/cat_w3_fv2.pkl"),
        ("model/xgb_w3_fv2x.pkl", "model/xgb_w3_fv2x.pkl"),
        ("model/cat_w3_role42.pkl", "model/cat_w3_role42.pkl"),
        ("model/cat_w3_fatig.pkl", "model/cat_w3_fatig.pkl"),
        ("model/cat_w3_mgr42.pkl", "model/cat_w3_mgr42.pkl"),
    ],
    "blenddiv": [
        ("src/script_blend_div.py", "script.py"),
        ("src/features.py", "features.py"),
        ("src/pitcher_role.py", "pitcher_role.py"),
        ("src/manager_feat.py", "manager_feat.py"),
        ("src/rules.py", "rules.py"),
        ("requirements_blend.txt", "requirements.txt"),
        ("model/cat_hs_d7_l10_r0.08.pkl", "model/cat_hs_d7_l10_r0.08.pkl"),
        ("model/cat_fv2.pkl", "model/cat_fv2.pkl"),
        ("model/xgb_fv2.pkl", "model/xgb_fv2.pkl"),
        ("model/cat_role42.pkl", "model/cat_role42.pkl"),
        ("model/cat_fatig.pkl", "model/cat_fatig.pkl"),
        ("model/cat_mgr42.pkl", "model/cat_mgr42.pkl"),
    ],
    "blendv4": [
        ("src/script_blend_v4.py", "script.py"),
        ("src/features.py", "features.py"),
        ("src/pitcher_role.py", "pitcher_role.py"),
        ("src/rules.py", "rules.py"),
        ("requirements_blend.txt", "requirements.txt"),
        ("model/cat_hs_d7_l10_r0.08.pkl", "model/cat_hs_d7_l10_r0.08.pkl"),
        ("model/cat_hs_d8_l3_r0.05.pkl", "model/cat_hs_d8_l3_r0.05.pkl"),
        ("model/cat_fv2.pkl", "model/cat_fv2.pkl"),
        ("model/cat_fv2s3.pkl", "model/cat_fv2s3.pkl"),
        ("model/cat_fv2s4.pkl", "model/cat_fv2s4.pkl"),
        ("model/cat_role42.pkl", "model/cat_role42.pkl"),
        ("model/cat_fatig.pkl", "model/cat_fatig.pkl"),
        ("model/xgb_fv2.pkl", "model/xgb_fv2.pkl"),
    ],
    "blendv3": [
        ("src/script_blend_v3.py", "script.py"),
        ("src/features.py", "features.py"),
        ("requirements_blend.txt", "requirements.txt"),
        ("model/cat_hs_d7_l10_r0.08.pkl", "model/cat_hs_d7_l10_r0.08.pkl"),
        ("model/cat_hs_d8_l3_r0.05.pkl", "model/cat_hs_d8_l3_r0.05.pkl"),
        ("model/cat_hs_d8_l10_r0.05.pkl", "model/cat_hs_d8_l10_r0.05.pkl"),
        ("model/cat_fv2.pkl", "model/cat_fv2.pkl"),
        ("model/cat_fv2s3.pkl", "model/cat_fv2s3.pkl"),
        ("model/cat_fv2s4.pkl", "model/cat_fv2s4.pkl"),
        ("model/xgb_fv2.pkl", "model/xgb_fv2.pkl"),
    ],
    "blendv2": [
        ("src/script_blend_v2.py", "script.py"),
        ("src/features.py", "features.py"),
        ("requirements_blend.txt", "requirements.txt"),
        ("model/cat_fv2.pkl", "model/cat_fv2.pkl"),
        ("model/cat_fv2s2.pkl", "model/cat_fv2s2.pkl"),
        ("model/cat_fv2s3.pkl", "model/cat_fv2s3.pkl"),
        ("model/cat_fv2s4.pkl", "model/cat_fv2s4.pkl"),
        ("model/xgb_fv2.pkl", "model/xgb_fv2.pkl"),
    ],
    "catfv2cal": [
        ("src/script_cat_fv2cal.py", "script.py"),
        ("src/features.py", "features.py"),
        ("requirements_cat.txt", "requirements.txt"),
        ("model/cat_fv2.pkl", "model/cat_fv2.pkl"),
        ("model/cat_fv2s2.pkl", "model/cat_fv2s2.pkl"),
        ("model/cat_fv2s3.pkl", "model/cat_fv2s3.pkl"),
    ],
    "blend": [
        ("src/script_blend.py", "script.py"),
        ("src/tabm_reference.py", "tabm_reference.py"),
        ("src/rtdl_num_embeddings.py", "rtdl_num_embeddings.py"),
        ("requirements_lgbm.txt", "requirements.txt"),
        ("model/lgbm_v3.pkl", "model/lgbm_v3.pkl"),
        ("model/prep_v1.pkl", "model/prep_v1.pkl"),
        ("model/tabm_v4s0_best.pt", "model/tabm_v4s0_best.pt"),
        ("model/tabm_v4s1_best.pt", "model/tabm_v4s1_best.pt"),
        ("model/tabm_v4s2_best.pt", "model/tabm_v4s2_best.pt"),
    ],
}



def _check_imports(include):
    """script.py 가 import 하는 로컬 모듈이 zip 에 다 들어 있는가.

    `fpipe.py` 를 빠뜨린 적이 있다. 평가 서버는 zip 을 풀고 script.py 를 실행하므로
    모듈 하나만 없어도 즉시 ImportError 로 죽고, 그건 **설치 오류가 아니라 제출
    오류라 제출 횟수가 차감된다.** 사람이 목록을 손으로 관리하면 또 빠뜨린다.
    """
    import ast
    packed = {dst for _, dst in include}
    entry = next(s for s, d in include if d == "script.py")
    seen, todo = set(), [entry]
    while todo:
        f = todo.pop()
        if f in seen:
            continue
        seen.add(f)
        with open(os.path.join(ROOT, f), encoding="utf-8") as fh:
            tree = ast.parse(fh.read())
        for n in ast.walk(tree):
            if isinstance(n, ast.Import):
                mods = [a.name for a in n.names]
            elif isinstance(n, ast.ImportFrom):
                mods = [n.module or ""]
            else:
                continue
            for m in mods:
                cand = "src/%s.py" % m.split(".")[0]
                if os.path.exists(os.path.join(ROOT, cand)):
                    todo.append(cand)
    # 진입점 자신은 zip 안에서 script.py 로 이름이 바뀌므로 제외한다
    need = {os.path.basename(f) for f in seen} - {os.path.basename(entry)}
    missing = sorted(need - packed)
    if missing:
        raise SystemExit(
            "zip 에 빠진 모듈: %s -- script.py 가 import 하는데 include 목록에 "
            "없다. 이대로 제출하면 ImportError 로 제출 횟수만 차감된다." % missing)
    print("import 폐포 확인: %s" % sorted(need))


def blend_from_script(script_rel, extra):
    """추론 스크립트의 WEIGHTS를 파싱해 모델 목록을 자동 구성한다.

    모델 목록을 여기에 또 적으면 스크립트와 어긋날 수 있다(실제로 08-06에
    비슷한 복붙 불일치로 기능 하나가 통째로 죽었다). 단일 출처로 유지한다.
    """
    # 정규식으로 텍스트를 긁으면 f-string 이나 리스트 컴프리헨션을 못 읽는다
    # (실제로 v14 에서 `f"./model/cat_v14f_s{s}.pkl"` 을 통째로 놓쳤다).
    # 모듈을 **실제로 import 해서** WEIGHTS 를 읽으면 zip 과 스크립트가 어긋날 수 없다.
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_sub_script", os.path.join(ROOT, script_rel))
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, os.path.join(ROOT, "src"))
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.path.pop(0)
    models = [os.path.basename(p) for p, _ in mod.WEIGHTS]
    if not models:
        raise SystemExit(f"{script_rel} 의 WEIGHTS가 비어 있음 - 먼저 채울 것")
    w = sum(v for _, v in mod.WEIGHTS)
    print(f"  WEIGHTS 합 {w:.4f} | SHIFT {getattr(mod, 'SHIFT', 0):.4f} "
          f"| SLOPE {getattr(mod, 'SLOPE', 1.0):.4f}")
    files = [(script_rel, "script.py")] + extra
    files += [(f"model/{m}", f"model/{m}") for m in models]
    return files


def main():
    variant = sys.argv[1] if len(sys.argv) > 1 else "rf"
    if variant in ("blendv6", "blendv7", "blendv8", "blendv9"):
        include = blend_from_script(f"src/script_{variant.replace('blend', 'blend_')}.py", [
            ("src/features.py", "features.py"),
            ("src/target_enc.py", "target_enc.py"),
            ("src/season_std.py", "season_std.py"),   # E99 시즌내 복원
            ("src/skill.py", "skill.py"),             # E117 투수x카운트 회귀
            ("src/fpipe.py", "fpipe.py"),             # 피처 파이프라인(학습과 공유)
            ("requirements_blend.txt", "requirements.txt"),
        ])
        print(f"{variant}: script에서 모델 {len(include) - 4}개 자동 수집")
    elif variant in VARIANTS:
        include = VARIANTS[variant]
    else:
        raise SystemExit(
            f"variant는 {list(VARIANTS) + ['blendv6', 'blendv7']} 중 "
            f"하나여야 함: {variant}")
    for src, _ in include:
        if not os.path.exists(os.path.join(ROOT, src)):
            raise FileNotFoundError(f"필수 파일 없음: {src}")
    _check_imports(include)

    os.makedirs(OUT_DIR, exist_ok=True)
    # 제출 사이트 파일명 30자 제한 → {variant}_{MMdd_HHmm}.zip 로 짧게
    name = f"{variant}_{datetime.now():%m%d_%H%M}.zip"
    assert len(name) <= 30, f"파일명 30자 초과({len(name)}): {name}"
    out_path = os.path.join(OUT_DIR, name)

    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for src, arc in include:
            zf.write(os.path.join(ROOT, src), arc)

    size_mb = os.path.getsize(out_path) / 1e6
    print(f"생성 완료: {out_path} ({size_mb:.1f} MB)")
    with zipfile.ZipFile(out_path) as zf:
        for info in zf.infolist():
            print(f"  {info.filename}  ({info.file_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
