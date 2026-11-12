# Baseball League Revenue Optimization — 2026 FIELD CAMP

가상 프로야구 리그에 **수요 예측 → 리그 스케줄 최적화(MILP) → 다이나믹 프라이싱**을 차례로 적용해 시즌 입장 수익을 극대화한 프로젝트입니다.
2026 FIELD CAMP(전국 산업공학도 학술교류캠프) 주제 1, 팀 6B · **우수상(2등)**.

| | |
|---|---|
| 기간 | 2026년 7월 |
| 팀 | 8명 (팀장: 변준형 — 팀 리딩, Phase 2 수리 모델링, 결선 발표) |
| 기술 | Python · pandas · scikit-learn · LightGBM · XGBoost · CatBoost · docplex / IBM CPLEX · SciPy |

## 결과 요약

| Phase | 내용 | 결과 |
|---|---|---|
| 1. 관중 점유율 예측 | 트렌드–잔차 분리 + 타겟 인코딩 + GBM 가중 앙상블 | 2025 테스트 RMSE **0.1616** |
| 2. 리그 스케줄 최적화 | 10팀·45시리즈 MILP, 네트워크 유량으로 이동거리 정식화 | 예상 수익 **78,292.66**, MIP Gap **3.28%** (8시간 제한) |
| 3. 다이나믹 프라이싱 | MNL 수요 모델에 구조·이벤트 지표 5개 설계, D-7/D-3 가격 최적화 | 시나리오별 이벤트 순효과 +16.5% / −4.9% / −3.8% |

## 폴더 구성

```
phase1_demand_forecasting/
  phase1_code.py            # 전처리, 타겟 인코딩, 시간 기반 검증, 가중 앙상블
  phase1_prediction.csv     # 2025 시즌 709경기 점유율 예측값
phase2_milp_scheduling/
  phase2_code.ipynb         # docplex MILP 모델 (제약식 번호별 셀 구성)
  phase2_modeling.docx      # 수식 정식화 문서 (집합·파라미터·변수·제약식)
  phase2_schedule.csv       # 도출된 45개 시리즈 일정
  phase2_cplex_log.txt      # CPLEX 풀이 로그
phase3_dynamic_pricing/
  phase3_code.ipynb         # 수요 지표 설계, 시나리오 시뮬레이션, 민감도 분석
docs/
  phase3_report.pdf         # 다이나믹 프라이싱 제안서 (12p)
  final_presentation.pdf    # 결선 발표 자료
```

## 주요 포인트

### Phase 1 — 외삽 가능한 예측 모델
- 트리 모델이 학습 범위 밖 상승 추세를 예측하지 못하는 문제를, 구장 규모별 `logit(점유율) ~ 시즌` 추세 + GBM 잔차 학습으로 해결
- K-Fold Out-of-Fold 타겟 인코딩(누수 방지·스무딩), 신규 구장 콜드스타트 폴백
- 2개 시간 분할 검증에서 가장 꾸준한 가중 앙상블(CatBoost 0.47 / XGBoost 0.30 / LightGBM 0.23) 채택

### Phase 2 — 풀 수 있는 크기로 정식화
- 목적함수: 예상 수익 최대화 − ε·(팀 간 이동거리 최대–최소 차)
- 이동거리 계산을 AND 선형화 대신 **네트워크 유량 보존식**으로 바꿔 연결 제약을 약 93% 줄임 (132,000 → 8,800)
- infeasible 진단에 CPLEX Conflict Refiner 사용, 실행 가능해 우선 탐색 후 Solution Polishing

### Phase 3 — 근거 기반 변수 설계와 효과 분해
- 요일·시즌 시기(구조) / 대기록·부상·강수확률(이벤트) 지표를 KBO·MLB 통계와 논문 근거로 보수적으로 설계
- 대조군·처치군의 요일 차이를 분리하지 않으면 이벤트 효과가 최대 5배 이상 과대평가된다는 점을 보이고, 가격 정책을 기본 가격표와 이벤트 조정으로 나눌 것을 제안
- 계수 민감도 분석에서 전 구간 내부해 유지(코너해 0건)

## 실행 방법

```bash
pip install pandas numpy scikit-learn lightgbm xgboost catboost scipy docplex cplex
```

- 대회 제공 원본 데이터(`phase1_final_without_ans.csv`, `p_phase2_series.csv`, `team_venue_phase2.csv`, `D_phase2.csv` 등)는 주최 측 자료라 저장소에 포함하지 않았습니다.
- Phase 2 전체 모델은 CPLEX 무료(Community) 버전의 변수 수 제한을 넘으므로 정식 또는 학생용 라이선스가 필요합니다.
