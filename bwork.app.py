import io
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="지역별 유망/인기업종 분석", layout="wide")

st.title("지역별 유망/인기업종 분석 (당월/전월)")
st.caption("CSV 컬럼: 시도, 시군구, 업종, 월구분(전월/당월), 사업자수, 매출액(원)")

# ---- Sidebar ----
st.sidebar.header("데이터 불러오기")
uploaded = st.sidebar.file_uploader("CSV 업로드 (.csv, UTF-8-SIG 권장)", type=["csv"])
csv_path_text = st.sidebar.text_input("또는 CSV 경로 입력", value="")
top_n = st.sidebar.number_input("상위 N개", min_value=1, max_value=50, value=10, step=1)
growth_thresh = st.sidebar.number_input("유망업종 증가율 임계값(%)", min_value=0.0, max_value=10000.0, value=100.0, step=10.0)

@st.cache_data(show_spinner=False)
def load_csv_from_path(path: str) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")

@st.cache_data(show_spinner=False)
def load_csv_from_bytes(b: bytes) -> pd.DataFrame:
    bio = io.BytesIO(b)
    return pd.read_csv(bio, encoding="utf-8-sig")

def prep_df(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["시도", "시군구", "업종", "월구분", "사업자수", "매출액(원)"]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"CSV에 필요한 컬럼이 없습니다: {missing}")
    for c in ["시도", "시군구", "업종", "월구분"]:
        df[c] = df[c].astype(str).str.strip()
    df["월구분"] = pd.Categorical(df["월구분"], categories=["전월","당월"], ordered=True)
    return df[cols].copy()

def analyze_region(df: pd.DataFrame, sido: str, sigungu: str, top_n: int = 10, growth_thresh: float = 100.0):
    mask_region = (df["시도"] == str(sido).strip()) & (df["시군구"] == str(sigungu).strip())
    sub = df.loc[mask_region].copy()
    if sub.empty:
        raise ValueError(f"해당 지역 데이터를 찾을 수 없습니다: 시도={sido}, 시군구={sigungu}")
    pv = sub.pivot_table(index=["시도","시군구","업종"],
                         columns="월구분",
                         values=["사업자수","매출액(원)"],
                         aggfunc="sum")
    pv.columns = [f"{a}_{b}" for a, b in pv.columns.to_flat_index()]
    pv = pv.reset_index()

    # 증가율 계산 (전월 매출 0 제외)
    pv = pv[pv["매출액(원)_전월"] > 0].copy()
    pv["매출증가율(%)"] = (pv["매출액(원)_당월"] - pv["매출액(원)_전월"]) / pv["매출액(원)_전월"] * 100.0

    # 유망업종: 증가율 필터 & 정렬
    promising = pv[pv["매출증가율(%)"] >= growth_thresh].copy()
    promising = promising.sort_values(by=["매출증가율(%)", "매출액(원)_당월"], ascending=[False, False]).head(top_n)
    promising_view = promising[["업종","매출액(원)_전월","매출액(원)_당월","매출증가율(%)"]].reset_index(drop=True)

    # 인기업종: 당월 사업자수 상위
    pv["사업자수_당월"] = pv.get("사업자수_당월", pd.Series([0]*len(pv))).fillna(0)
    popular = pv.sort_values(by=["사업자수_당월", "매출액(원)_당월"], ascending=[False, False]).head(top_n)
    popular_view = popular[["업종","사업자수_당월","매출액(원)_당월"]].reset_index(drop=True)

    return promising_view, popular_view

# ---- Data load ----
df = None
if uploaded is not None:
    try:
        df = load_csv_from_bytes(uploaded.read())
    except Exception as e:
        st.error(f"업로드한 CSV를 읽는 중 오류: {e}")
elif csv_path_text:
    try:
        df = load_csv_from_path(csv_path_text)
    except Exception as e:
        st.error(f"경로에서 CSV를 읽는 중 오류: {e}")

if df is not None:
    try:
        df = prep_df(df)
        # 지역 선택 위젯
        col1, col2 = st.columns(2)
        with col1:
            sido = st.selectbox("시도 선택", sorted(df["시도"].unique().tolist()))
        with col2:
            candidates = sorted(df.loc[df["시도"] == sido, "시군구"].unique().tolist())
            sigungu = st.selectbox("시군구 선택", candidates)

        st.markdown("---")
        st.subheader(f"선택 지역: {sido} {sigungu}")
        promising_df, popular_df = analyze_region(df, sido, sigungu, top_n=int(top_n), growth_thresh=float(growth_thresh))

        # 표 보여주기
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### 유망업종 (증가율 상위)")
            st.dataframe(promising_df, use_container_width=True)
        with c2:
            st.markdown("### 인기업종 (사업자수 상위)")
            st.dataframe(popular_df, use_container_width=True)

        # ---- Charts with matplotlib (no seaborn, no specific colors) ----
        # 1) 유망업종 증가율 막대그래프
        if not promising_df.empty:
            fig1 = plt.figure()
            x = promising_df["업종"]
            y = promising_df["매출증가율(%)"]
            plt.bar(x, y)
            plt.xticks(rotation=70, ha="right")
            plt.ylabel("매출증가율(%)")
            plt.title("유망업종: 매출증가율(%) 상위")
            st.pyplot(fig1)

        # 2) 인기업종 사업자수 막대그래프
        if not popular_df.empty:
            fig2 = plt.figure()
            x2 = popular_df["업종"]
            y2 = popular_df["사업자수_당월"]
            plt.bar(x2, y2)
            plt.xticks(rotation=70, ha="right")
            plt.ylabel("사업자수(당월)")
            plt.title("인기업종: 사업자수(당월) 상위")
            st.pyplot(fig2)

    except Exception as e:
        st.error(f"분석 중 오류가 발생했습니다: {e}")
else:
    st.info("왼쪽에서 CSV를 업로드하거나 경로를 입력한 후 지역을 선택하세요.")
