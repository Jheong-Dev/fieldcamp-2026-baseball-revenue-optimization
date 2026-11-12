import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

RANDOM_STATE = 42


def rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred) ** 0.5


def logit(p, eps=1e-3):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# =============================================================================
# 1. 데이터 로드 및 예측 대상 확인
#    - 예측 대상 시즌은 대회 규정상 2025로 고정 (자동 탐지 아님)
# =============================================================================
DATA_PATH = "phase1_final_without_ans.csv"   # 대회 당일 실제 파일명으로 교체
TARGET_SEASON = 2025

raw = pd.read_csv(DATA_PATH)

test_raw = raw[raw['season'] == TARGET_SEASON].copy()
labeled = raw[raw['season'] != TARGET_SEASON].copy()
EXPECTED_TEST_ROWS = len(test_raw)

train_raw = labeled[labeled['occupancy_rate'].notna()].reset_index(drop=True)

assert test_raw.shape[0] > 0, f"TARGET_SEASON={TARGET_SEASON}에 해당하는 행이 없습니다."
assert train_raw['occupancy_rate'].between(0, 1).all(), "occupancy_rate 범위(0~1)를 벗어난 값이 있습니다."


# =============================================================================
# 2. 전처리
#    - season_num(경과 시즌 수), 팀x구장 조합, 티어 순서화, 프리미엄 요인 합산
#    - 학습 데이터에 없던 구장(신규 구장 등)은 is_new_stadium=1로 플래그
# =============================================================================
def preprocess(frame):
    frame = frame.copy()
    frame['season_num'] = frame['season'] - frame['season'].min()
    frame['home_stadium_combo'] = frame['home_team'] + '_' + frame['stadium']
    cap_map = {'Low': 1, 'Mid': 2, 'High': 3}
    frame['capacity_tier_ord'] = frame['capacity_tier'].map(cap_map)
    frame['premium_count'] = frame['weekend'] + frame['opening'] + frame['holiday']
    return frame

season_anchor = train_raw['season'].min()
train_raw['season_num'] = train_raw['season'] - season_anchor
test_raw['season_num'] = test_raw['season'] - season_anchor
train_raw = preprocess(train_raw).assign(season_num=train_raw['season_num'])
test_raw = preprocess(test_raw).assign(season_num=test_raw['season_num'])

train_stadiums = set(train_raw['stadium'].unique())

def add_unseen_flag(fit_source_stadiums, frame):
    frame = frame.copy()
    frame['is_new_stadium'] = (~frame['stadium'].isin(fit_source_stadiums)).astype(int)
    return frame

train_raw = add_unseen_flag(train_stadiums, train_raw)
test_raw = add_unseen_flag(train_stadiums, test_raw)


# =============================================================================
# 3. 파생변수 - OOF 스무딩 타겟 인코딩
#    - home_team / away_team / matchup / game_time / home×stadium 조합별 과거 평균 점유율
#    - 누수 방지: K-Fold OOF로 계산 / 표본 부족 노이즈 방지: 베이지안 스무딩
#    - 콜드스타트(신규 구장 등): home_team_te로 폴백
# =============================================================================
def fit_encoding_map(fit_df, group_cols, target_col='occupancy_rate', smoothing=10):
    grp = fit_df.groupby(group_cols)[target_col].agg(['mean', 'count'])
    global_mean = fit_df[target_col].mean()
    smoothed = (grp['mean'] * grp['count'] + global_mean * smoothing) / (grp['count'] + smoothing)
    return smoothed.to_dict(), global_mean

def _make_keys(frame, group_cols):
    if len(group_cols) == 1:
        return frame[group_cols[0]]
    return list(zip(*[frame[c] for c in group_cols]))

def apply_encoding_map(frame, group_cols, mapping, global_mean, fallback=None):
    keys = _make_keys(frame, group_cols)
    out = pd.Series(keys, index=frame.index).map(mapping)
    if fallback is not None:
        out = out.fillna(fallback)
    return out.fillna(global_mean)

def oof_target_encoding(fit_df, group_cols, target_col='occupancy_rate',
                         n_splits=5, smoothing=10, seed=RANDOM_STATE, fallback_series=None):
    oof = pd.Series(index=fit_df.index, dtype=float)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr_idx, val_idx in kf.split(fit_df):
        fold_fit = fit_df.iloc[tr_idx]
        fold_val = fit_df.iloc[val_idx]
        mapping, gmean = fit_encoding_map(fold_fit, group_cols, target_col, smoothing)
        fb = fallback_series.iloc[val_idx] if fallback_series is not None else None
        oof.iloc[val_idx] = apply_encoding_map(fold_val, group_cols, mapping, gmean, fallback=fb).values
    return oof

def add_target_encodings(fit_source, target_frame, is_train_self):
    out = target_frame.copy()
    def enc(group_cols, colname, fallback_col=None):
        fb = out[fallback_col] if fallback_col is not None else None
        if is_train_self:
            out[colname] = oof_target_encoding(fit_source, group_cols, fallback_series=fb)
        else:
            mapping, gmean = fit_encoding_map(fit_source, group_cols)
            out[colname] = apply_encoding_map(out, group_cols, mapping, gmean, fallback=fb)
    enc(['home_team'], 'home_team_te')
    enc(['away_team'], 'away_team_te')
    enc(['home_team', 'away_team'], 'matchup_te')
    enc(['game_time'], 'game_time_te')
    enc(['home_team', 'stadium'], 'home_stadium_te', fallback_col='home_team_te')
    return out

CAT_COLS = ['home_team', 'away_team', 'stadium', 'game_time', 'capacity_tier', 'home_stadium_combo']
NUM_COLS = ['month', 'weekend', 'opening', 'holiday', 'capacity_tier_ord', 'premium_count',
            'is_new_stadium', 'home_team_te', 'away_team_te', 'matchup_te', 'game_time_te', 'home_stadium_te']
FEATURE_COLS = CAT_COLS + NUM_COLS

def build_features(fit_source, frame, is_train_self):
    out = add_target_encodings(fit_source, frame, is_train_self)
    seen = set(fit_source['stadium'].unique())
    out['is_new_stadium'] = (~out['stadium'].isin(seen)).astype(int)
    return out

def to_train_valid_frames(train_frame, valid_frame):
    # 카테고리 목록을 학습 프레임 기준으로 고정 (train/valid 각자 astype하면 범주 코드 어긋남)
    Xtr = train_frame[FEATURE_COLS].copy()
    cats = {}
    for c in CAT_COLS:
        Xtr[c] = Xtr[c].astype('category')
        cats[c] = Xtr[c].cat.categories
    Xva = valid_frame[FEATURE_COLS].copy()
    for c in CAT_COLS:
        Xva[c] = pd.Categorical(Xva[c], categories=cats[c])
    return Xtr, Xva

def to_cat_str_frame(frame):
    X = frame[FEATURE_COLS].copy()
    for c in CAT_COLS:
        X[c] = X[c].astype(str)
    return X


# =============================================================================
# 4. 시즌 트렌드-잔차 분리 (Trend-Residual Decomposition)
#    - 트리 모델은 학습 시 못 본 season_num 값으로 외삽이 안 되므로,
#      logit(occupancy_rate) ~ season_num 선형 추세를 티어별로 먼저 분리하고
#      GBM은 그 잔차만 학습 (sigmoid로 역변환해 0~1 범위 보장)
# =============================================================================
def fit_trend_models(frame, group_col='capacity_tier', x_col='season_num', y_col='occupancy_rate'):
    models = {}
    for key, sub in frame.groupby(group_col):
        models[key] = LinearRegression().fit(sub[[x_col]], logit(sub[y_col].values))
    models['__global__'] = LinearRegression().fit(frame[[x_col]], logit(frame[y_col].values))
    return models

def trend_predict(models, frame, group_col='capacity_tier', x_col='season_num'):
    out = np.zeros(len(frame))
    for key in frame[group_col].unique():
        mask = (frame[group_col] == key).values
        lm = models.get(key, models['__global__'])
        out[mask] = sigmoid(lm.predict(frame.loc[mask, [x_col]]))
    return out


# =============================================================================
# 4-1. 검증 RMSE 확인 (빠른 홀드아웃 체크)
#    - 라벨이 있는 가장 최신 시즌을 검증셋으로 떼어놓고, 나머지로 학습해 RMSE를 미리 가늠해본다.
#    - 실전 제출용 학습(5장)과는 완전히 분리된 변수(cv_ 접두어)를 써서 서로 영향 없음.
# =============================================================================
val_season = train_raw['season'].max()
cv_train = train_raw[train_raw['season'] < val_season].reset_index(drop=True)
cv_valid = train_raw[train_raw['season'] == val_season].reset_index(drop=True)

cv_ftr = build_features(cv_train, cv_train, is_train_self=True)
cv_fva = build_features(cv_train, cv_valid, is_train_self=False)

cv_trend_models = fit_trend_models(cv_ftr)
cv_trend_tr = trend_predict(cv_trend_models, cv_ftr)
cv_trend_va = trend_predict(cv_trend_models, cv_fva)

cv_resid_tr = cv_ftr['occupancy_rate'].values - cv_trend_tr
cv_sw = (cv_train['season_num'].values + 1).astype(float)

cv_Xtr, cv_Xva = to_train_valid_frames(cv_ftr, cv_fva)
cv_Xtr_c, cv_Xva_c = to_cat_str_frame(cv_ftr), to_cat_str_frame(cv_fva)

cv_lgb = LGBMRegressor(n_estimators=300, learning_rate=0.05, max_depth=4, num_leaves=15,
                        min_child_samples=15, subsample=0.8, colsample_bytree=0.8,
                        random_state=RANDOM_STATE, verbosity=-1)
cv_lgb.fit(cv_Xtr, cv_resid_tr, sample_weight=cv_sw, categorical_feature=CAT_COLS)
cv_pred_lgb = np.clip(cv_trend_va + cv_lgb.predict(cv_Xva), 0, 1)

cv_xgb = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=4, subsample=0.8,
                       colsample_bytree=0.8, enable_categorical=True, tree_method='hist',
                       random_state=RANDOM_STATE)
cv_xgb.fit(cv_Xtr, cv_resid_tr, sample_weight=cv_sw)
cv_pred_xgb = np.clip(cv_trend_va + cv_xgb.predict(cv_Xva), 0, 1)

cv_cat = CatBoostRegressor(iterations=300, learning_rate=0.05, depth=6,
                            random_seed=RANDOM_STATE, verbose=False)
cv_cat.fit(cv_Xtr_c, cv_resid_tr, sample_weight=cv_sw, cat_features=CAT_COLS)
cv_pred_cat = np.clip(cv_trend_va + cv_cat.predict(cv_Xva_c), 0, 1)

cv_y_valid = cv_fva['occupancy_rate'].values

print(f"[검증] {val_season}시즌 홀드아웃 앙상블 RMSE = {rmse(cv_y_valid, np.clip(0.4*cv_pred_cat+0.4*cv_pred_lgb+0.2*cv_pred_xgb,0,1)):.5f} (n={len(cv_y_valid)})")
print(f"[검증] 개별 모델 RMSE - LightGBM: {rmse(cv_y_valid, cv_pred_lgb):.5f} | "
      f"XGBoost: {rmse(cv_y_valid, cv_pred_xgb):.5f} | CatBoost: {rmse(cv_y_valid, cv_pred_cat):.5f}")


# =============================================================================
# 4-2. 앙상블 가중치 그리드서치
#    - 4-1의 검증 예측(최신 시즌 홀드아웃)에, 가장 이른 시즌만 학습 -> 두 번째 시즌 검증이라는
#      또 다른 시간 분할의 예측을 하나 더 만들어 행 단위로 풀링한다 (단일 분할로 가중치를 정하면
#      그 분할의 우연한 패턴에 과적합될 위험이 있어, 두 시나리오의 검증 표본을 합쳐서 사용).
#    - 풀링된 표본 위에서 가중치(cat, lgb, xgb / 합=1, 각 모델 최소 5% 보장)를 0.01 간격
#      그리드서치로 탐색해 RMSE를 최소화하는 조합을 앙상블 가중치로 채택한다.
# =============================================================================
seasons_sorted = sorted(train_raw['season'].unique())
spA_train_season = seasons_sorted[0]
spA_val_season = seasons_sorted[1]

spA_train = train_raw[train_raw['season'] == spA_train_season].reset_index(drop=True)
spA_valid = train_raw[train_raw['season'] == spA_val_season].reset_index(drop=True)

spA_ftr = build_features(spA_train, spA_train, is_train_self=True)
spA_fva = build_features(spA_train, spA_valid, is_train_self=False)

spA_trend_models = fit_trend_models(spA_ftr)
spA_trend_tr = trend_predict(spA_trend_models, spA_ftr)
spA_trend_va = trend_predict(spA_trend_models, spA_fva)

spA_resid_tr = spA_ftr['occupancy_rate'].values - spA_trend_tr
spA_sw = (spA_train['season_num'].values + 1).astype(float)

spA_Xtr, spA_Xva = to_train_valid_frames(spA_ftr, spA_fva)
spA_Xtr_c, spA_Xva_c = to_cat_str_frame(spA_ftr), to_cat_str_frame(spA_fva)

spA_lgb = LGBMRegressor(n_estimators=300, learning_rate=0.05, max_depth=4, num_leaves=15,
                         min_child_samples=15, subsample=0.8, colsample_bytree=0.8,
                         random_state=RANDOM_STATE, verbosity=-1)
spA_lgb.fit(spA_Xtr, spA_resid_tr, sample_weight=spA_sw, categorical_feature=CAT_COLS)
spA_pred_lgb = np.clip(spA_trend_va + spA_lgb.predict(spA_Xva), 0, 1)

spA_xgb = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=4, subsample=0.8,
                        colsample_bytree=0.8, enable_categorical=True, tree_method='hist',
                        random_state=RANDOM_STATE)
spA_xgb.fit(spA_Xtr, spA_resid_tr, sample_weight=spA_sw)
spA_pred_xgb = np.clip(spA_trend_va + spA_xgb.predict(spA_Xva), 0, 1)

spA_cat = CatBoostRegressor(iterations=300, learning_rate=0.05, depth=6,
                             random_seed=RANDOM_STATE, verbose=False)
spA_cat.fit(spA_Xtr_c, spA_resid_tr, sample_weight=spA_sw, cat_features=CAT_COLS)
spA_pred_cat = np.clip(spA_trend_va + spA_cat.predict(spA_Xva_c), 0, 1)

spA_y_valid = spA_fva['occupancy_rate'].values

# 4-1(cv_*)과 4-2(spA_*) 검증 예측을 행 단위로 풀링
y_pool = np.concatenate([spA_y_valid, cv_y_valid])
P_cat = np.concatenate([spA_pred_cat, cv_pred_cat])
P_lgb = np.concatenate([spA_pred_lgb, cv_pred_lgb])
P_xgb = np.concatenate([spA_pred_xgb, cv_pred_xgb])
print(f"[가중치 탐색] 풀링된 검증 표본 수: {len(y_pool)} "
      f"(SplitA {len(spA_y_valid)} + SplitB {len(cv_y_valid)})")

step = 0.01
grid = np.round(np.arange(0, 1.0001, step), 2)
search_results = []
for wc in grid:
    for wl in grid:
        wx = round(1 - wc - wl, 2)
        if wx < 0 or wx > 1:
            continue
        if wc < 0.05 or wl < 0.05 or wx < 0.05:   # 특정 모델을 완전히 배제하지 않도록 최소 5% 보장
            continue
        pred = np.clip(wc * P_cat + wl * P_lgb + wx * P_xgb, 0, 1)
        search_results.append((rmse(y_pool, pred), wc, wl, wx))
search_results.sort(key=lambda x: x[0])

ENS_W_CAT, ENS_W_LGB, ENS_W_XGB = search_results[0][1], search_results[0][2], search_results[0][3]
baseline_pooled_rmse = rmse(y_pool, np.clip(0.4 * P_cat + 0.4 * P_lgb + 0.2 * P_xgb, 0, 1))

print(f"[가중치 탐색] 기존 고정값(cat=0.4, lgb=0.4, xgb=0.2) 풀링 RMSE = {baseline_pooled_rmse:.5f}")
print(f"[가중치 탐색] 그리드서치 채택 가중치 -> cat={ENS_W_CAT}, lgb={ENS_W_LGB}, xgb={ENS_W_XGB} "
      f"(풀링 RMSE={search_results[0][0]:.5f})")


# =============================================================================
# 5. 최종 학습 및 예측
#    - LightGBM / XGBoost / CatBoost 3개 GBM을 잔차에 학습 후 가중 앙상블
#      (가중치는 4-2 그리드서치에서 채택한 ENS_W_CAT / ENS_W_LGB / ENS_W_XGB 사용)
#    - 최근 시즌일수록 큰 가중치(sample_weight)를 줘서 최신 트렌드에 더 민감하게 학습
# =============================================================================
full_train = build_features(train_raw, train_raw, is_train_self=True)
full_test = build_features(train_raw, test_raw, is_train_self=False)

trend_models_full = fit_trend_models(full_train)
trend_full = trend_predict(trend_models_full, full_train)
trend_test = trend_predict(trend_models_full, full_test)

y_full = full_train['occupancy_rate'].values
resid_full = y_full - trend_full
sample_weight_full = (train_raw['season_num'].values + 1).astype(float)

Xfull, Xtest = to_train_valid_frames(full_train, full_test)
Xfull_c, Xtest_c = to_cat_str_frame(full_train), to_cat_str_frame(full_test)

lgb_final = LGBMRegressor(n_estimators=300, learning_rate=0.05, max_depth=4, num_leaves=15,
                           min_child_samples=15, subsample=0.8, colsample_bytree=0.8,
                           random_state=RANDOM_STATE, verbosity=-1)
lgb_final.fit(Xfull, resid_full, sample_weight=sample_weight_full, categorical_feature=CAT_COLS)
pred_lgb = trend_test + lgb_final.predict(Xtest)

xgb_final = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=4, subsample=0.8,
                          colsample_bytree=0.8, enable_categorical=True, tree_method='hist',
                          random_state=RANDOM_STATE)
xgb_final.fit(Xfull, resid_full, sample_weight=sample_weight_full)
pred_xgb = trend_test + xgb_final.predict(Xtest)

cat_final = CatBoostRegressor(iterations=300, learning_rate=0.05, depth=6,
                               random_seed=RANDOM_STATE, verbose=False)
cat_final.fit(Xfull_c, resid_full, sample_weight=sample_weight_full, cat_features=CAT_COLS)
pred_cat = trend_test + cat_final.predict(Xtest_c)

# 최종 예측값 (그리드서치로 채택된 가중치 사용)
pred_ensemble = ENS_W_CAT * pred_cat + ENS_W_LGB * pred_lgb + ENS_W_XGB * pred_xgb


# =============================================================================
# 6. 후처리 및 제출 파일 생성
# =============================================================================
TEAM_NAME = "6B"

final_pred = np.round(np.clip(pred_ensemble, 0, 1), 4)

submission = pd.DataFrame({
    'index': np.arange(1, len(final_pred) + 1),
    'occupancy_rate': final_pred
})

assert submission.shape[0] == EXPECTED_TEST_ROWS, (
    f"제출 행 수({submission.shape[0]})가 예측 대상 행 수({EXPECTED_TEST_ROWS})와 다릅니다."
)
assert list(submission.columns) == ['index', 'occupancy_rate'], "컬럼명/순서를 확인하세요."
assert submission['occupancy_rate'].between(0, 1).all(), "0~1 범위를 벗어난 예측값이 있습니다."

out_path = f"{TEAM_NAME}_phase1_prediction.csv"
submission.to_csv(out_path, index=False, encoding='utf-8-sig')
print("저장 완료:", out_path, "| 행 수:", submission.shape[0])
