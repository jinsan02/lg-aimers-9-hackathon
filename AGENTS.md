# AGENTS.md — Claude / Codex 공용 진입점

> **어느 에이전트든 작업 전에 이 파일부터 읽는다.**
> Codex 는 `AGENTS.md`, Claude 는 `CLAUDE.md` 를 관례적으로 먼저 읽으므로,
> 공용 규칙은 전부 여기에 두고 `CLAUDE.md` 는 대회 스펙만 담는다.

## 1. Project

LG Aimers 9기 — KBO 투구 **제구 성공 확률** 예측 (Dacon 236743). 마감 2026-09-01.
투구가 이루어지기 **전까지 확인 가능한 정보만으로** 0~1 확률을 낸다.

### Goal

- 지표: **Brier Skill Score** `max(0, 1e5 × (1 − brier / r(1−r)))` — 높을수록 좋음
- Public = Private = 전체 테스트 100%. **hidden split 이 없다** → 과적합보다 순수 일반화
- 현재 **LB 1,093.85 (9위)** / 1위 1,126.33

전체 대회 스펙·데이터 명세·제출 zip 구조·평가 서버 사양은 **[CLAUDE.md](CLAUDE.md)** 에 있다.
Codex 도 첫 작업 전에 CLAUDE.md 를 한 번 읽을 것.

---

## 2. Tech Stack

Python 3.11 · CatBoost(GPU) · pandas 2.0.3 · numpy 1.26.4 · scikit-learn 1.8.0 · joblib 1.5.3

**평가 서버 버전에 고정돼 있다.** 로컬에서 임의로 올리지 말 것 — pkl 호환성이 깨진다.

---

## 3. Project Structure

```text
CLAUDE.md              대회 스펙 (규칙·데이터·서버·제출)
AGENTS.md              이 파일 — 공용 운영 규칙
EXPERIMENT.md          현재 상태판 (지금 뭘 하고 있나)
HANDOFF.md             에이전트 간 인수인계
LEDGER.tsv             ★ 자동 기록. 시드마다 1행 (머신·태그·시드·BSS·명령 전문)
                         원격 머신에 쌓이므로 `bash tools/ledger_sync.sh` 로 합친다
docs/SETTLED.md        ★ 닫힌 질문 — 다시 하지 말 것 (precheck 이 읽음)
docs/EXPERIMENTS_LOG.md  장문 실험 로그 (역사)
docs/experiment_guide.md 머신별 실행 방법
LEVERS.md              레버 백로그

src/
├── train_gbdt2.py     학습 본체 (CatBoost/LGB/XGB, 모든 플래그)
├── fpipe.py           ★ 피처 파이프라인 — fit/transform 이 같은 파일에 있다
├── failmode.py        실패모드 라벨 복원 (⚠ train 전용, 제출 zip 미포함)
├── teacher.py         증류 교사
├── script_blend_v*.py 제출용 추론 스크립트
tools/                 분석·검증 도구 (precheck / surf_report / audit_*)
model/  out/  submissions/
```

---

## 3-1. 원격 머신 — ssh 로 붙는다 (2026-08-08 실측)

학습은 전부 원격에서 돈다. 노트북은 **분석·판정·제출 패키징 전용**이다(발열).

| 별칭 | 하드웨어 | 접속 | 프로젝트 | 파이썬 |
|---|---|---|---|---|
| `hsu-server` | **A100 40GB** · 80 vCPU · RAM 503GB | `ssh hsu-server` (desktop-4070 경유 ProxyJump, 포트 8822) | `~/aimers` | `~/venv451/bin/python` (3.10.12) |
| `desktop-4070` | **RTX 4070 Ti SUPER 16GB** | `ssh desktop-4070` | `C:\aimers` | `.venv\Scripts\python.exe` (3.11.15) |
| 노트북 (여기) | RTX 5060 8GB | — | `C:\aimers` | `uv run python` (3.11.15) |

ssh 별칭은 `~/.ssh/config` 에 이미 있다. **`hsu-server` 는 `desktop-4070` 을 거쳐 간다** —
4070 이 꺼져 있으면 A100 에도 못 붙는다.

### 셸이 머신마다 다르다 (여기서 여러 번 헛발질했다)

- `hsu-server` → **bash**. 평범하게 쓰면 된다.
- `desktop-4070` → **cmd.exe** 다. PowerShell 도 bash 도 아니다.
  `;` 는 명령 구분자가 아니고, `&&` 대신 `&` 를 쓴다. `head`/`tail`/`grep` 이 없다
  (`findstr`). 한글 출력은 CP949 라 로컬에서 UTF-8 로 읽으면 깨진다 —
  **로그는 `.npz` 산출물로 판정하고 콘솔 텍스트에 의존하지 말 것.**
- 노트북 → Git Bash (POSIX). PowerShell 도 따로 있다.

### 코드 동기화

```bash
scp src/*.py   hsu-server:~/aimers/src/
scp src/*.py   desktop-4070:C:/aimers/src/     # 슬래시 방향 주의
scp tools/*.py hsu-server:~/aimers/tools/
```

원본은 항상 노트북. 원격에서 편집하지 않는다.

### 장기 실행 (ssh 끊겨도 살아남게)

```bash
# A100 — PPID 1 로 떨어진다
ssh hsu-server "cd ~/aimers && setsid nohup bash X.sh > out/X.log 2>&1 < /dev/null & disown"

# 4070 — 이 런처로만. 이유는 §4-15
bash tools/run4070.sh <이름> 'C:\aimers\X.bat' 'C:\aimers\out\X.log'
```

4070 에서 `setsid nohup` · `start /b` · `Start-Process` 는 전부 ssh 종료와 함께 죽는다.
WSL `tmux` 안에서는 Windows exe interop 이 깨진다(`UtilAcceptVsock accept4 failed 110`).
**남은 방법이 `schtasks` 뿐이고, 그건 `tools/run4070.sh` 가 안전하게 감싼다.**

### 환경 차이 — 알고 있을 것

| | pandas | numpy | sklearn |
|---|---|---|---|
| 평가 서버 · 노트북 · 4070 | 2.0.3 | 1.26.4 | 1.8.0 |
| **A100** | **2.3.3** | **2.2.6** | **1.7.2** |

**세 머신의 예측이 소수점 12자리까지 같다는 걸 실측했다**(`tools/env_check.py`).
파이프라인이 버전에 노출되는 연산을 안 쓴다. 그래도 새 라이브러리를 쓰면 다시 확인할 것.

**그리고 A100 결과와 4070 결과를 직접 비교하지 말 것** — 같은 설정·같은 시드가
11점 갈린다(§4-5). 머신마다 자기 기준선을 따로 만든다.

### 원장 합치기

`LEDGER.tsv` 는 학습이 돈 머신에 쌓인다. 판정 전에 모은다:

```bash
bash tools/ledger_sync.sh
```

---

## 4. 반드시 지킬 것 — 어긴 대가가 기록돼 있다

### 실행 전

1. **`python tools/precheck.py <플래그 전부>` 를 통과시킨다.** 종료코드 2 면 금지.
   스크립트째로도 된다: `--file some.sh`
   → 08-07 에 이미 −580 으로 끝난 `season` 제거를 "안 해본 축"이라며 다시 큐에 걸었다.
2. **`docs/SETTLED.md` 를 읽는다.** 닫힌 질문에 수치와 **기전**이 같이 적혀 있다.
3. GPU 점유를 확인한다. 한 머신에 한 작업.

### 측정

4. **판정 표면은 `--val-season S-1 --test-season S`** (배치 구조 = ≤S-1 재학습 → 미학습 S).
   자기검증 홀드아웃은 참고용 — 셀 멤버 D 가 두 표면에서 +2.8 vs −18.0 으로 갈렸다.
5. **비교군은 같은 머신에서 만든다.** 같은 설정·같은 6시드가 A100/4070 에서 **11점** 갈렸다.
   시드를 늘려도 못 막는다. `tools/surf_report.py` 가 LEDGER 의 hostname 으로 자동 무효 처리.
6. **채택 t ≥ 2.4 / 기각 95% 상한 < +3.** 홀드아웃 시드 n ≥ 6, 페어 비교.
   → refit 배수 2.0 이 6시드에서 +2.44(t=1.91) 였는데 18시드로 늘리니 +0.60(기각).
7. **세그먼트 BSS 를 손실로 읽지 말 것.** 기저율이 0.5 에서 멀면 BSS 가 낮아 보인다.
   F리그가 BSS 로는 −301 인데 **MSE 로는 오히려 낮았다**(.24690 vs .24770).

### 후처리·제출

8. **후처리 상수는 그 값을 잰 실행의 학습 데이터·플래그가 제출과 완전히 같을 때만 쓴다.**
   → v16 −6.15 (편향은 `--drop-f-pre 2022` 로 재고 제출엔 안 썼다)
9. **구조가 다른 실험의 결론을 옮기지 말 것.** → v17 −53.6
10. **제출 하나에 변경 하나.** → v16 은 SHIFT+SLOPE 동시 변경으로 역산이 필요했다
11. **제출 전 `python tools/audit_rowindep.py <script>` 통과 필수.**
    배치 구성을 바꿔도 예측이 비트 단위로 같아야 한다. 데이콘이 행 독립을 명시 요구한다.
12. **LB 를 보고 상수를 되맞추지 않는다.** Public=Private 이라 정답 세트 직접 적합이다.

### 코드

13. **로그를 grep 으로 거르지 말 것.** `2>&1 | grep -E "^\[cat|..."` 가 트레이스백을
    통째로 삼켜서, 죽은 실행이 "완료"로 보였다(증류 1차). 전문을 파일에 남긴다.
14. 기존 구조·스타일을 유지한다. 한 실험에서 여러 요소를 동시에 바꾸지 않는다.
15. 4070 장기 실행은 **`bash tools/run4070.sh`** 로만. `schtasks` 에 가까운 시각을
    더미로 쓰면 그 시각에 전부 자동 재발화한다(9개가 동시에 떠서 GPU 를 9등분했다).

---

## 5. Experiment Record

**사람이 적는 단계는 없앴다.** `train_gbdt2.py` 가 시드마다 `LEDGER.tsv` 에 자동 기록한다:

```text
날짜 · 머신 · 태그 · 모델 · val시즌 · test시즌 · 시드 · best_iter · val BSS · test BSS · 명령 전문
```

에이전트가 추가로 남길 것:

- 축이 **닫히면** `docs/SETTLED.md` 에 한 줄 (판정 · 수치 · **기전**)
- 제출하면 `docs/EXPERIMENTS_LOG.md` 에 LB 점수와 예측 대비 오차
- 작업이 끝나면 `HANDOFF.md` 갱신

---

## 6. Agent Roles

### Claude — 무엇을 측정할지 정한다

- 손실 분해·잔차 분석·가설 수립, 실험 우선순위
- 결과 해석과 **판정** (t 값, 표면, 머신 일치 확인)
- 제출 여부 결정, 후처리 상수 도출
- 규칙 위반 위험 심사 (행 독립·외부 데이터·LB 프로빙)

### Codex — 정해진 실험을 정확히 구현한다

- 실험 스크립트/배치 작성, 플래그 배선, 에러 수정
- 도구(`tools/*.py`) 구현, 리팩터링, 테스트
- 머신별 실행·모니터링·산출물 회수
- 제출 zip 패키징과 스모크

**경계**: Codex 는 *판정하지 않는다*. 수치를 내고 표로 정리해서 `HANDOFF.md` 에 남긴다.
Claude 는 *구현 세부에 개입하지 않는다*. 무엇을·왜·어떤 기준으로 판정할지를 적어 넘긴다.

---

## 7. Source of Truth

1. 실제 코드
2. `LEDGER.tsv` (실행된 명령 전문 — 기억보다 이걸 믿는다)
3. `docs/SETTLED.md`
4. `docs/EXPERIMENTS_LOG.md`
5. `EXPERIMENT.md` / `HANDOFF.md`

문서와 코드가 다르면 **코드를 확인하고 문서를 고친다.**

---

## 8. 작업 시작/종료 체크리스트

시작:
1. `AGENTS.md` → `EXPERIMENT.md` → `HANDOFF.md`
2. `docs/SETTLED.md` 에서 하려는 축이 닫혀 있는지
3. `tools/precheck.py` 통과
4. 두 머신 GPU 점유 확인

종료:
1. `HANDOFF.md` 갱신 (Status·결과·다음 권고)
2. 축이 닫혔으면 `docs/SETTLED.md` 에 추가
3. 제출했으면 `docs/EXPERIMENTS_LOG.md` 에 LB 기록
