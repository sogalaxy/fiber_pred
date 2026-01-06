# app_nanofiber_inverse_design.py
# ------------------------------------------------------------
# GBR 결과를 이용한 나노섬유 직경 "역설계" 웹앱 (Streamlit)
# - 입력: 폴리머 종류/블렌드 비율(폴리머1, 폴리머2, wt%), 목표 직경(um)
# - 출력: 용매 추천 + 공정조건(V, D, F, 농도) 추천 (GBR 예측 기반)
#
# 사용법:
#   1) pip install streamlit pandas numpy scikit-learn openpyxl
#   2) (선택) input.xlsx를 같은 폴더에 두거나, 앱에서 업로드
#   3) streamlit run app_nanofiber_inverse_design.py
#
# 주의:
# - 아래 HSP/용매 DB, 공정 조건 탐색 범위는 "시작점"입니다. 실험실 조건에 맞춰 범위를 꼭 조정하세요.
# - 모델은 앱 실행 시 업로드된 데이터로 재학습합니다(GBR best-params 기본값 포함).
# ------------------------------------------------------------

import io
import math
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error


# -----------------------------
# HSP DB (필요시 확장/수정)
# -----------------------------
HSP = {
    # polymers
    "pcl": (17.7, 6.2, 7.8),
    "gelatin": (18.5, 14.4, 20.6),
    "plga": (19.0, 8.0, 10.0),  # TODO: 실제 조성(50:50 등)에 맞춰 업데이트 권장

    # solvents
    "hfip": (16.4, 6.1, 14.3),
    "acetic acid": (14.5, 8.0, 13.5),
    "glacial acetic acid": (14.5, 8.0, 13.5),
    "formic acid": (14.3, 11.9, 16.6),
    "tfe": (15.4, 9.4, 16.7),
    "dmf": (17.4, 13.7, 11.3),
    "dcm": (18.2, 6.3, 6.1),
    "chloroform": (17.8, 3.1, 5.7),
    "dmso": (18.4, 16.4, 10.2),
    "thf": (16.8, 5.7, 8.0),
    "acetone": (15.5, 10.4, 7.0),
    "methanol": (15.1, 12.3, 22.3),
    "lactic acid": (17.0, 8.3, 28.4),
}

SOLVENT_CANDIDATES = [
    "hfip",
    "acetic acid",
    "formic acid",
    "tfe",
    "dmf",
    "dcm",
    "chloroform",
    "dmso",
    "thf",
    "acetone",
    "methanol",
]


# -----------------------------
# Helpers
# -----------------------------

def _clean_num(val):
    if pd.isna(val):
        return np.nan
    if isinstance(val, str):
        v = val.replace(",", ".").strip()
        if v == "":
            return np.nan
        try:
            return float(v)
        except Exception:
            return np.nan
    try:
        return float(val)
    except Exception:
        return np.nan


def ra_hsp(hsp_poly, hsp_sol):
    dd = (hsp_poly[0] - hsp_sol[0])
    dp = (hsp_poly[1] - hsp_sol[1])
    dh = (hsp_poly[2] - hsp_sol[2])
    return float(math.sqrt(4.0 * dd * dd + dp * dp + dh * dh))


def mix_hsp(hsp1, w1, hsp2=None, w2=0.0):
    """Simple weighted mixing rule for HSP components."""
    if hsp2 is None or w2 <= 0:
        s = w1 if w1 > 0 else 1.0
        return (hsp1[0], hsp1[1], hsp1[2])
    s = w1 + w2
    if s <= 0:
        s = 1.0
    w1n = w1 / s
    w2n = w2 / s
    return (
        w1n * hsp1[0] + w2n * hsp2[0],
        w1n * hsp1[1] + w2n * hsp2[1],
        w1n * hsp1[2] + w2n * hsp2[2],
    )


def build_features(con_wt, vol_kv, fee_mlhr, dis_cm, berry):
    return np.array([[con_wt, vol_kv, fee_mlhr, dis_cm, berry]], dtype=float)


def load_training_data_from_xlsx(xlsx_bytes: bytes, sheet_name="Sheet1") -> pd.DataFrame:
    df = pd.read_excel(io.BytesIO(xlsx_bytes), sheet_name=sheet_name)

    # expected columns (your earlier pipeline)
    df["poly_percent"] = df.get("poly_percent", np.nan).apply(_clean_num)
    df["poly2_percent"] = df.get("poly2_percent", np.nan).apply(_clean_num)
    df["VOL"] = df["voltage (kV)"].apply(_clean_num)
    df["DIS"] = df["distance(cm)"].apply(_clean_num)
    df["FEE"] = df["flor rate(ml/hr)"].apply(_clean_num)
    df["DIM"] = df["diameter(um)"].apply(_clean_num)

    df["CON"] = df[["poly_percent", "poly2_percent"]].fillna(0).sum(axis=1)

    # names
    df["polymer1_"] = df["polymer1"].astype(str).str.lower().str.strip()
    df["solvent1_"] = df["solvent1"].astype(str).str.lower().str.strip()

    # Berry from Ra
    def _calc(row):
        p = HSP.get(row["polymer1_"], None)
        s = HSP.get(row["solvent1_"], None)
        con = row["CON"]
        if p is None or s is None or not np.isfinite(con) or con <= 0:
            return np.nan
        ra = ra_hsp(p, s)
        if ra == 0:
            return np.nan
        return con / ra

    df["Berry"] = df.apply(_calc, axis=1)

    features = ["CON", "VOL", "FEE", "DIS", "Berry"]
    df_ml = df.dropna(subset=features + ["DIM"]).copy()
    return df_ml


@st.cache_resource
def train_gbr_model(df_ml: pd.DataFrame):
    features = ["CON", "VOL", "FEE", "DIS", "Berry"]
    X = df_ml[features].values
    y = df_ml["DIM"].values

    # GBR best params (당신 출력값 기반): learning_rate=0.03, max_depth=2, n_estimators=900, subsample=1.0
    # (여기서 max_depth는 base estimator tree depth; sklearn GBR는 max_depth를 직접 받습니다)
    model = GradientBoostingRegressor(
        random_state=42,
        n_estimators=900,
        learning_rate=0.03,
        max_depth=2,
        subsample=1.0,
    )

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42)
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)

    report = {
        "n": int(len(df_ml)),
        "r2_test": float(r2_score(yte, pred)),
        "mae_test": float(mean_absolute_error(yte, pred)),
        "features": features,
    }
    return model, report


def inverse_design(
    model,
    polymer1: str,
    polymer2: str,
    w1: float,
    w2: float,
    target_um: float,
    allowed_solvents,
    con_grid,
    vol_grid,
    dis_grid,
    fee_grid,
    top_k=8,
    ra_soft_limit=10.0,
):
    """Brute-force search over solvents + process grids.

    Score = |pred - target| / target  + 0.05 * penalty(Ra)
    penalty(Ra)=max(0, (Ra - ra_soft_limit)/ra_soft_limit)

    Returns top_k rows sorted by score.
    """

    p1 = HSP.get(polymer1, None)
    p2 = HSP.get(polymer2, None) if polymer2 != "(none)" else None
    if p1 is None:
        raise ValueError(f"Unknown polymer: {polymer1}")

    p_mix = mix_hsp(p1, w1, p2, w2) if p2 is not None else mix_hsp(p1, 1.0)

    rows = []
    eps = 1e-9

    for sol in allowed_solvents:
        s = HSP.get(sol, None)
        if s is None:
            continue
        ra = ra_hsp(p_mix, s)

        for con in con_grid:
            berry = con / (ra + eps)
            for vol in vol_grid:
                for dis in dis_grid:
                    for fee in fee_grid:
                        X = build_features(con, vol, fee, dis, berry)
                        pred = float(model.predict(X)[0])

                        rel_err = abs(pred - target_um) / max(target_um, eps)
                        ra_pen = max(0.0, (ra - ra_soft_limit) / max(ra_soft_limit, eps))
                        score = rel_err + 0.05 * ra_pen

                        rows.append({
                            "score": score,
                            "solvent": sol,
                            "Ra(poly-sol)": ra,
                            "CON(wt%)": con,
                            "Voltage(kV)": vol,
                            "Distance(cm)": dis,
                            "Flow(ml/hr)": fee,
                            "Berry(CON/Ra)": berry,
                            "Predicted(um)": pred,
                            "Target(um)": target_um,
                            "AbsErr(um)": abs(pred - target_um),
                            "RelErr(%)": 100.0 * rel_err,
                        })

    out = pd.DataFrame(rows)
    out = out.sort_values("score", ascending=True).head(top_k).reset_index(drop=True)
    return out


# -----------------------------
# Streamlit UI
# -----------------------------

st.set_page_config(page_title="NanoFiber Inverse Design (GBR)", layout="wide")

st.title("GBR 기반 나노섬유 직경 역설계 (용매 + 공정조건 추천)")

with st.sidebar:
    st.header("1) 학습 데이터")
    st.caption("GBR 모델은 업로드된 input.xlsx(또는 기본 파일)로 재학습됩니다.")

    uploaded = st.file_uploader("input.xlsx 업로드", type=["xlsx"])
    sheet = st.text_input("Sheet name", value="Sheet1")

    st.divider()
    st.header("2) 탐색 범위(추천 설정)")
    st.caption("너무 넓으면 느려집니다. 먼저 좁게 시작하세요.")

    con_min, con_max = st.slider("총 농도 CON(wt%) 범위", 1.0, 30.0, (5.0, 15.0), step=0.5)
    con_step = st.select_slider("CON step", options=[0.5, 1.0, 2.5, 5.0], value=2.5)

    vol_min, vol_max = st.slider("전압(kV) 범위", 3.0, 30.0, (8.0, 20.0), step=0.5)
    vol_step = st.select_slider("V step", options=[0.5, 1.0, 2.0, 3.0], value=2.0)

    dis_min, dis_max = st.slider("거리(cm) 범위", 5.0, 30.0, (10.0, 20.0), step=0.5)
    dis_step = st.select_slider("D step", options=[0.5, 1.0, 2.0, 3.0], value=2.0)

    fee_min, fee_max = st.slider("공급속도(ml/hr) 범위", 0.05, 5.0, (0.2, 1.0), step=0.05)
    fee_step = st.select_slider("F step", options=[0.05, 0.1, 0.2, 0.3], value=0.2)

    st.divider()
    ra_soft = st.slider("Ra soft limit(용매 적합성 페널티 기준)", 4.0, 20.0, 10.0, step=0.5)
    top_k = st.slider("추천 개수 Top-K", 3, 20, 8)


colA, colB = st.columns([1.1, 1.2])

with colA:
    st.subheader("입력")

    polymer_options = sorted([k for k in HSP.keys() if k in ["pcl", "gelatin", "plga"]])
    if len(polymer_options) == 0:
        polymer_options = ["pcl", "gelatin", "plga"]

    p1 = st.selectbox("Polymer 1", polymer_options, index=0)
    p2 = st.selectbox("Polymer 2 (optional)", ["(none)"] + polymer_options, index=0)

    if p2 != "(none)" and p2 == p1:
        st.warning("Polymer2가 Polymer1과 같습니다. 블렌드가 아니라면 (none)으로 두세요.")

    w1 = st.slider("Polymer1 wt% fraction", 0.0, 1.0, 1.0 if p2 == "(none)" else 0.5, step=0.05)
    w2 = (1.0 - w1) if p2 != "(none)" else 0.0
    st.write(f"Polymer2 fraction: **{w2:.2f}**")

    target_um = st.number_input("목표 직경 (um)", min_value=0.01, value=0.80, step=0.05)

    st.subheader("용매 후보")
    selected_solvents = st.multiselect(
        "탐색할 용매 선택",
        options=SOLVENT_CANDIDATES,
        default=["hfip", "acetic acid", "formic acid", "tfe"],
    )

    run_btn = st.button("추천 실행", type="primary")


with colB:
    st.subheader("모델 상태")

    # Load data
    if uploaded is not None:
        xlsx_bytes = uploaded.read()
    else:
        # Try to read local file (same folder)
        try:
            with open("input.xlsx", "rb") as f:
                xlsx_bytes = f.read()
        except Exception:
            xlsx_bytes = None

    if xlsx_bytes is None:
        st.error("input.xlsx를 찾을 수 없습니다. 왼쪽에서 업로드해 주세요.")
        st.stop()

    df_ml = load_training_data_from_xlsx(xlsx_bytes, sheet_name=sheet)
    model, report = train_gbr_model(df_ml)

    st.write(
        f"- 학습 데이터 샘플수: **{report['n']}**\n"
        f"- Test R²: **{report['r2_test']:.3f}**\n"
        f"- Test MAE: **{report['mae_test']:.3f} um**\n"
        f"- Features: `{', '.join(report['features'])}`"
    )

    with st.expander("학습 데이터 미리보기"):
        st.dataframe(df_ml.head(20))


st.divider()

st.subheader("추천 결과")

if run_btn:
    if len(selected_solvents) == 0:
        st.warning("용매 후보를 1개 이상 선택해 주세요.")
        st.stop()

    # grids
    con_grid = np.round(np.arange(con_min, con_max + 1e-9, con_step), 3)
    vol_grid = np.round(np.arange(vol_min, vol_max + 1e-9, vol_step), 3)
    dis_grid = np.round(np.arange(dis_min, dis_max + 1e-9, dis_step), 3)
    fee_grid = np.round(np.arange(fee_min, fee_max + 1e-9, fee_step), 3)

    # sanity: avoid huge search
    total_cases = len(selected_solvents) * len(con_grid) * len(vol_grid) * len(dis_grid) * len(fee_grid)
    st.write(f"탐색 조합 수: **{total_cases:,}**")
    if total_cases > 350_000:
        st.warning("탐색 조합 수가 너무 큽니다. 범위를 줄이거나 step을 키우면 더 빨라집니다.")

    try:
        rec = inverse_design(
            model=model,
            polymer1=p1,
            polymer2=p2,
            w1=w1,
            w2=w2,
            target_um=target_um,
            allowed_solvents=selected_solvents,
            con_grid=con_grid,
            vol_grid=vol_grid,
            dis_grid=dis_grid,
            fee_grid=fee_grid,
            top_k=top_k,
            ra_soft_limit=ra_soft,
        )

        st.dataframe(rec, use_container_width=True)

        # download
        csv = rec.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "추천 결과 CSV 다운로드",
            data=csv,
            file_name="nanofiber_recommendations.csv",
            mime="text/csv",
        )

        st.markdown("#### 해석 가이드")
        st.write(
            "- **RelErr(%)**: 목표 직경 대비 상대오차(낮을수록 좋음)\n"
            "- **Ra(poly-sol)**: 폴리머-용매 HSP 거리(낮을수록 용해/상호작용 적합성이 높다는 가정)\n"
            "- **Berry(CON/Ra)**: (단순화된) 적합성/농도 지표. 모델 입력 feature로 사용됩니다.\n"
            "- 결과는 데이터 기반 추천이며, 실제 실험에서는 점도/전도도/습도/온도 등 미포함 요인으로 달라질 수 있습니다."
        )

    except Exception as e:
        st.error(f"추천 실행 중 오류: {e}")


st.caption("© GBR inverse design prototype. 실험실 조건에 맞춰 탐색 범위/용매 DB/HSP 값을 반드시 보정하세요.")
