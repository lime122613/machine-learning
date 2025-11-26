# app.py
# ---------------------------------------------
# 지도학습 실습용 Streamlit 웹 앱
# - CSV 업로드 → 특징/타깃 선택 → 상관관계 히트맵 → 분류/회귀 선택 → 모델 학습/평가
# ---------------------------------------------
from pandas.errors import EmptyDataError
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

import plotly.express as px


# ---------------------------------------------
# 기본 설정
# ---------------------------------------------
st.set_page_config(
    page_title="지도학습 실습 앱",
    page_icon="📘",
    layout="wide",
)


# ---------------------------------------------
# 0. 유틸 함수들
# ---------------------------------------------
def load_data(uploaded_file):
    """CSV 파일을 읽어 DataFrame으로 반환하는 함수 (인코딩/빈 파일 예외 처리 포함)"""
    if uploaded_file is None:
        return None

    try:
        # 첫 번째 시도 전에 항상 파일 포인터를 처음으로 돌려놓기
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file)
        return df

    except UnicodeDecodeError:
        # 인코딩 문제로 실패했으면 cp949로 다시 시도
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, encoding="cp949")
        return df

    except EmptyDataError:
        # 파일이 비어있을 때
        st.error("CSV 파일에 **데이터가 없습니다.** 내용이 있는 CSV 파일을 업로드해 주세요.")
        return None

    except Exception as e:
        # 그 외 예외는 메시지만 보여주고 None 반환
        st.error(f"CSV를 읽는 중 문제가 발생했습니다: {e}")
        return None


def show_data_overview(df):
    """데이터프레임 기본 정보 출력"""
    st.subheader("1️⃣ 데이터 미리보기")
    st.dataframe(df.head(10), use_container_width=True)

    st.markdown("#### 📁 데이터 요약 정보")
    col1, col2 = st.columns(2)

    with col1:
        st.write(f"- 행(row) 개수: **{df.shape[0]}**")
        st.write(f"- 열(column) 개수: **{df.shape[1]}**")
        st.write("- 열 이름:")
        st.write(list(df.columns))

    with col2:
        st.write("🔍 결측치 개수 (열별):")
        st.write(df.isna().sum())


def show_correlation_heatmap(df):
    """수치형 변수들 간의 상관관계 히트맵"""
    st.subheader("2️⃣ 수치형 변수 간 상관관계 히트맵")

    numeric_df = df.select_dtypes(include=[np.number])

    if numeric_df.shape[1] < 2:
        st.info("수치형 열이 2개 이상 있어야 상관관계 히트맵을 그릴 수 있습니다.")
        return

    corr = numeric_df.corr()

    fig = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale="RdBu",
        zmin=-1,
        zmax=1,
        aspect="auto",
        labels=dict(color="상관계수"),
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "상관계수는 -1에서 1 사이의 값이며, 절댓값이 1에 가까울수록 두 변수의 선형 관계가 강합니다.\n"
        "단, **상관관계가 곧 인과관계(원인-결과)를 의미하는 것은 아닙니다.**"
    )


def show_target_correlations(df, target_col):
    """타깃 변수와 다른 수치형 변수들의 상관관계 표"""
    st.subheader("3️⃣ 타깃 변수와의 상관관계")

    numeric_cols = df.select_dtypes(include=[np.number]).columns

    if target_col not in numeric_cols:
        st.info(
            "타깃 변수가 수치형이 아니어서 상관계수를 계산할 수 없습니다. "
            "분류 문제에서는 주로 범주형(문자형) 타깃을 사용합니다."
        )
        return

    corr_with_target = df[numeric_cols].corr()[target_col].drop(target_col)

    # 절댓값 기준으로 내림차순 정렬
    corr_sorted = corr_with_target.reindex(
        corr_with_target.abs().sort_values(ascending=False).index
    )

    st.markdown("#### 🎯 타깃과 수치형 변수들의 상관계수")
    st.dataframe(
        corr_sorted.to_frame("상관계수"),
        use_container_width=True,
    )

    # 상관계수 절댓값이 0.5 이상인 변수만 한 번 더 강조
    strong_corr = corr_sorted[ corr_sorted.abs() >= 0.5 ]
    if not strong_corr.empty:
        st.markdown("##### ⭐ 상관계수 절댓값이 0.5 이상인 변수들 (강한 상관관계)")
        st.write(strong_corr.to_frame("상관계수"))
    else:
        st.info("절댓값 0.5 이상으로 강한 상관관계를 보이는 수치형 변수는 없습니다.")

    st.caption(
        "상관계수의 절댓값이 0.5 이상이면 비교적 강한 선형 관계가 있다고 볼 수 있습니다.\n"
        "이 정보를 활용해 **타깃과 관련이 높아 보이는 특징 열을 선택해 볼 수 있습니다.**"
    )


def choose_features_and_target(df):
    """사이드바에서 독립변수(특징)와 타깃(정답) 변수 선택
       + 타깃과 상관관계가 높은 열들을 기본 선택으로 추천
    """
    st.sidebar.subheader("2️⃣ 입력/타깃 변수 선택")

    all_cols = df.columns.tolist()

    target_col = st.sidebar.selectbox(
        "타깃(정답) 변수 선택",
        options=["(선택 안 함)"] + all_cols,
    )

    if target_col == "(선택 안 함)":
        target_col = None

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # 기본 특징 선택: 타깃과 상관관계가 높은 상위 5개의 수치형 열
    default_features = []
    if target_col is not None and target_col in numeric_cols and len(numeric_cols) > 1:
        corr = df[numeric_cols].corr()[target_col].drop(target_col)
        top_features = (
            corr.abs()
            .sort_values(ascending=False)
            .head(5)
            .index
            .tolist()
        )
        default_features = top_features
        st.sidebar.caption(
            "※ 타깃과 상관계수가 높은 수치형 열 기준으로 상위 5개를 기본 선택했습니다.\n"
            "   (언제든지 아래에서 직접 수정할 수 있습니다.)"
        )
    else:
        # 타깃이 수치형이 아니거나, 상관관계 계산이 어려운 경우 → 수치형 열 전체 추천
        default_features = [c for c in numeric_cols if c != target_col]

    feature_cols = st.sidebar.multiselect(
        "입력(특징) 변수 선택 (여러 개 선택 가능)",
        options=all_cols,
        default=default_features,
    )

    # 타깃이 특징에 섞여 있으면 제거
    if target_col and target_col in feature_cols:
        st.sidebar.warning("타깃 변수는 독립변수에서 자동으로 제외됩니다.")
        feature_cols = [c for c in feature_cols if c != target_col]

    return feature_cols, target_col


def infer_problem_type(y_series: pd.Series):
    """타깃 y의 타입과 고유값 개수를 기반으로 문제 유형을 추정"""
    if y_series.dtype == "object":
        return "classification", "타깃이 문자형(범주형) 데이터라서 **분류 문제**로 판단했습니다."
    unique_vals = y_series.nunique()

    if unique_vals <= 10:
        return (
            "classification",
            f"타깃 값의 종류가 {unique_vals}개로 비교적 적어서 **분류 문제**로 판단했습니다.",
        )
    else:
        return (
            "regression",
            f"타깃이 수치형이고 값의 종류가 많아서 **회귀 문제**로 판단했습니다.",
        )


def select_algorithm(problem_type: str):
    """문제 유형에 따라 알고리즘과 하이퍼파라미터 UI를 구성하고 선택 결과 반환"""
    st.sidebar.subheader("3️⃣ 알고리즘 선택 및 설정")

    params = {}

    if problem_type == "classification":
        algo = st.sidebar.selectbox(
            "분류 알고리즘 선택",
            ["로지스틱 회귀", "결정트리", "랜덤 포레스트", "K-최근접 이웃(KNN)"],
        )

        if algo == "로지스틱 회귀":
            params["max_iter"] = st.sidebar.slider(
                "반복 횟수 (max_iter)", 100, 500, 200, step=50
            )

        elif algo == "결정트리":
            params["max_depth"] = st.sidebar.slider(
                "트리 최대 깊이 (max_depth)", 1, 20, 5
            )

        elif algo == "랜덤 포레스트":
            params["n_estimators"] = st.sidebar.slider(
                "트리 개수 (n_estimators)", 10, 200, 100, step=10
            )
            params["max_depth"] = st.sidebar.slider(
                "트리 최대 깊이 (max_depth)", 1, 20, 5
            )

        elif algo == "K-최근접 이웃(KNN)":
            params["n_neighbors"] = st.sidebar.slider(
                "이웃 개수 (n_neighbors)", 1, 20, 5
            )

    else:  # regression
        algo = st.sidebar.selectbox(
            "회귀 알고리즘 선택",
            ["선형 회귀", "결정트리 회귀", "랜덤 포레스트 회귀", "K-최근접 이웃 회귀"],
        )

        if algo == "결정트리 회귀":
            params["max_depth"] = st.sidebar.slider(
                "트리 최대 깊이 (max_depth)", 1, 20, 5
            )

        elif algo == "랜덤 포레스트 회귀":
            params["n_estimators"] = st.sidebar.slider(
                "트리 개수 (n_estimators)", 10, 200, 100, step=10
            )
            params["max_depth"] = st.sidebar.slider(
                "트리 최대 깊이 (max_depth)", 1, 20, 5
            )

        elif algo == "K-최근접 이웃 회귀":
            params["n_neighbors"] = st.sidebar.slider(
                "이웃 개수 (n_neighbors)", 1, 20, 5
            )

    return algo, params


def build_model(problem_type: str, algo: str, params: dict):
    """선택된 알고리즘과 하이퍼파라미터로 모델 객체 생성"""
    if problem_type == "classification":
        if algo == "로지스틱 회귀":
            model = LogisticRegression(
                max_iter=params.get("max_iter", 200),
            )
        elif algo == "결정트리":
            model = DecisionTreeClassifier(
                max_depth=params.get("max_depth", None),
                random_state=42,
            )
        elif algo == "랜덤 포레스트":
            model = RandomForestClassifier(
                n_estimators=params.get("n_estimators", 100),
                max_depth=params.get("max_depth", None),
                random_state=42,
                n_jobs=-1,
            )
        elif algo == "K-최근접 이웃(KNN)":
            model = KNeighborsClassifier(
                n_neighbors=params.get("n_neighbors", 5),
                n_jobs=-1,
            )
        else:
            raise ValueError("지원하지 않는 분류 알고리즘입니다.")
    else:
        if algo == "선형 회귀":
            model = LinearRegression()
        elif algo == "결정트리 회귀":
            model = DecisionTreeRegressor(
                max_depth=params.get("max_depth", None),
                random_state=42,
            )
        elif algo == "랜덤 포레스트 회귀":
            model = RandomForestRegressor(
                n_estimators=params.get("n_estimators", 100),
                max_depth=params.get("max_depth", None),
                random_state=42,
                n_jobs=-1,
            )
        elif algo == "K-최근접 이웃 회귀":
            model = KNeighborsRegressor(
                n_neighbors=params.get("n_neighbors", 5),
                n_jobs=-1,
            )
        else:
            raise ValueError("지원하지 않는 회귀 알고리즘입니다.")

    return model


def show_classification_results(y_test, y_pred):
    """분류 모델 평가 결과 출력"""
    st.subheader("4️⃣ 분류 모델 평가 결과 🔍")

    # 1) 정확도
    acc = accuracy_score(y_test, y_pred)
    st.metric("정확도 (accuracy)", f"{acc:.3f}")
    st.caption("정확도는 전체 예측 중에서 **맞춘 비율**을 의미합니다.")

    # 2) 혼동 행렬
    cm = confusion_matrix(y_test, y_pred)
    labels = sorted(list(set(y_test) | set(y_pred)))
    labels_str = [str(l) for l in labels]   # 문자열 라벨

    st.markdown("#### 🔢 혼동행렬 (Confusion Matrix)")
    fig_cm = px.imshow(
        cm,
        x=labels_str,
        y=labels_str,
        text_auto=True,
        color_continuous_scale="Blues",
        aspect="equal",
    )
    fig_cm.update_layout(
        xaxis_title="예측 값",
        yaxis_title="실제 값",
        xaxis=dict(type="category"),
        yaxis=dict(type="category"),
    )
    st.plotly_chart(fig_cm, use_container_width=True)

    # 3) 클래스별 정밀도/재현율/F1-score (혼동행렬 바로 아래)
    with st.expander("클래스별 정밀도/재현율/F1-score 자세히 보기"):
        report_text = classification_report(y_test, y_pred)
        st.text(report_text)
        st.caption(
            "- 정밀도(precision): 맞다고 예측한 것 중에서 실제로 맞은 비율\n"
            "- 재현율(recall): 실제로 맞는 것 중에서 모델이 맞다고 찾아낸 비율\n"
            "- F1-score: 정밀도와 재현율의 조화평균(둘 다 균형 있게 잘 나오는지)"
        )

    # 4) 혼동행렬 + 분포 그래프에 대한 설명
    st.markdown(
        """
**혼동행렬 & 분포 그래프 해석**

- **혼동행렬**은 `정답(실제 값)`과 `예측`을 짝지어서 **얼마나 맞았는지/틀렸는지**를 보여줍니다.  
- **실제 분포 vs 예측 분포 그래프**는 모델이 각 클래스를 **얼마나 자주 선택했는지**를 실제 데이터와 비교해서 보여줍니다.  

두 정보를 함께 보면  
- 단순히 **맞춘 비율(정확도)**뿐 아니라,  **특정 답만 너무 많이 고르는 건 아닌지(편향)**도 함께 살펴볼 수 있습니다.
        """
    )


    # 실제/예측 라벨 빈도
    actual_counts = pd.Series(y_test).value_counts()
    pred_counts = pd.Series(y_pred).value_counts()

    all_labels = sorted(set(actual_counts.index) | set(pred_counts.index))
    actual_counts = actual_counts.reindex(all_labels, fill_value=0)
    pred_counts = pred_counts.reindex(all_labels, fill_value=0)

    actual_ratio = actual_counts / actual_counts.sum()
    pred_ratio = pred_counts / pred_counts.sum()

    dist_df = pd.DataFrame({
        "클래스": [str(l) for l in all_labels] * 2,
        "비율": np.concatenate([actual_ratio.values, pred_ratio.values]),
        "데이터": ["실제"] * len(all_labels) + ["예측"] * len(all_labels),
    })

    fig_compare = px.bar(
        dist_df,
        x="클래스",
        y="비율",
        color="데이터",
        barmode="group",
        text_auto=".2f",
    )
    fig_compare.update_layout(
        yaxis=dict(range=[0, 1]),
        yaxis_title="비율",
    )
    st.plotly_chart(fig_compare, use_container_width=True)

    st.caption(
        "막대그래프에서 **실제 분포**와 **예측 분포**의 모양이 비슷할수록, "
        "모델이 각 클래스를 보다 균형 있게 예측하고 있다고 볼 수 있습니다."
    )



def show_regression_results(y_test, y_pred):
    """회귀 모델 평가 결과 출력"""
    st.subheader("4️⃣ 회귀 모델 평가 결과 🔍")

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    col1, col2 = st.columns(2)

    with col1:
        st.metric("RMSE", f"{rmse:.3f}")
        st.metric("MAE", f"{mae:.3f}")

    with col2:
        st.metric("MSE", f"{mse:.3f}")
        st.metric("R²", f"{r2:.3f}")

    st.caption(
        "RMSE, MAE, MSE는 **예측 값과 실제 값의 차이가 얼마나 큰지**를 나타내고, "
        "R²는 **모델이 데이터를 얼마나 잘 설명하는지**를 보여주는 지표입니다. (1에 가까울수록 좋습니다.)"
    )

    # 실제 값 vs 예측 값 산점도
    st.markdown("#### 📈 실제 값 vs 예측 값")
    result_df = pd.DataFrame({"실제 값": y_test, "예측 값": y_pred})
    fig_scatter = px.scatter(result_df, x="실제 값", y="예측 값")
    # 기준선(완벽 예측일 때 y=x) 추가
    min_val = min(result_df["실제 값"].min(), result_df["예측 값"].min())
    max_val = max(result_df["실제 값"].max(), result_df["예측 값"].max())
    fig_scatter.add_shape(
        type="line",
        x0=min_val,
        y0=min_val,
        x1=max_val,
        y1=max_val,
    )
    fig_scatter.update_layout(xaxis_title="실제 값", yaxis_title="예측 값")
    st.plotly_chart(fig_scatter, use_container_width=True)
    st.caption("점들이 대각선 근처에 모여 있을수록 **예측이 잘 된 것**입니다.")


def run_training_and_evaluation(
    df, feature_cols, target_col, problem_type, algo, params, test_size
):
    """전체 학습/평가 파이프라인 실행"""
    st.subheader("4️⃣ 학습 및 평가 결과")

    data = df[feature_cols + [target_col]].copy()

    st.write(f"- 선택된 입력(특징) 변수: **{feature_cols}**")
    st.write(f"- 타깃(정답) 변수: **{target_col}**")

    # 결측치 처리
    missing_before = data.isna().sum().sum()
    if missing_before > 0:
        st.warning(
            f"결측치(빈 값)가 총 {missing_before}개 발견되어, "
            "해당 행을 제거하고 학습을 진행합니다."
        )
        data = data.dropna()

    if data.shape[0] < 5:
        st.error("행이 5개 미만이면 모델 학습이 어렵습니다. 더 많은 데이터가 필요합니다.")
        return

    X = data[feature_cols]
    y = data[target_col]

    # 🔹 분류 문제인데 타깃이 연속적인 수치형이면 미리 체크 🔹
    if problem_type == "classification":
        if np.issubdtype(y.dtype, np.floating):
            unique_vals = np.sort(y.unique())
            # 값들이 정수처럼 보이고, 종류가 많지 않으면 정수 라벨로 변환
            if (
                np.allclose(unique_vals, unique_vals.astype(int))
                and len(unique_vals) <= 20
            ):
                y = y.astype(int)
                st.info(
                    "타깃 값이 숫자(float)지만 값의 종류가 적고 정수처럼 보여 "
                    "**범주형 라벨(정수)** 로 자동 변환하여 분류 문제로 학습합니다."
                )
            else:
                st.error(
                    "현재 선택한 타깃 변수는 **연속적인 수치형 데이터**로 보입니다.\n\n"
                    "- 분류(Random Forest, 로지스틱 회귀 등)는 이런 타깃에 사용할 수 없습니다.\n"
                    "- 👉 사이드바에서 문제 유형을 **'회귀'**로 바꾸거나,\n"
                    "  또는 **범주형(클래스)** 타깃 열을 선택해주세요."
                )
                return

    # 범주형(문자형) 특징을 원-핫 인코딩
    X_encoded = pd.get_dummies(X, drop_first=True)

    # 회귀 문제에서 y가 숫자가 아니면 변환 시도
    if problem_type == "regression" and not np.issubdtype(y.dtype, np.number):
        try:
            y = pd.to_numeric(y)
        except Exception:
            st.error(
                "회귀 문제에서는 타깃 변수가 숫자형이어야 합니다. "
                "다른 타깃을 선택하거나 문제 유형을 분류로 바꿔보세요."
            )
            return

    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=test_size, random_state=42
    )

    st.write(f"- 학습(train) 데이터 행 개수: **{X_train.shape[0]}**")
    st.write(f"- 테스트(test) 데이터 행 개수: **{X_test.shape[0]}**")

    model = build_model(problem_type, algo, params)

    # 모델 학습
    model.fit(X_train, y_train)
    st.success("✅ 모델 학습이 완료되었습니다!")

    # 예측 및 평가
    y_pred = model.predict(X_test)

    if problem_type == "classification":
        show_classification_results(y_test, y_pred)
    else:
        show_regression_results(y_test, y_pred)


# ---------------------------------------------
# 메인 앱
# ---------------------------------------------
def main():
    st.title("📘 지도학습 실습 웹 앱")
    st.write(
        """
        이 앱은 **CSV 데이터를 업로드 → 특징/타깃 선택 → 상관관계 탐색 → 분류/회귀 알고리즘 선택 → 모델 학습/평가**까지  
        지도학습(supervised learning)의 전체 흐름을 직접 체험해볼 수 있도록 만든 교육용 도구입니다.
        """
    )

    st.info(
        "- **독립변수(입력/특징)**: 모델이 참고하는 정보 (예: 공부 시간, 나이, 키)\n"
        "- **종속변수(타깃/정답)**: 모델이 맞히고 싶은 값 (예: 시험 점수, 합격/불합격)\n"
        "- 분류 모델은 '라벨(범주)'를 맞추는 문제, 회귀 모델은 '숫자'를 예측하는 문제입니다."
    )

    # -----------------------------------------
    # 1. 데이터 업로드
    # -----------------------------------------
    st.sidebar.header("0️⃣ CSV 데이터 업로드")

    uploaded_file = st.sidebar.file_uploader(
        "CSV 파일을 업로드하세요", type=["csv"]
    )

    if uploaded_file is None:
        st.warning("왼쪽 사이드바에서 CSV 파일을 먼저 업로드해주세요. 😊")
        st.stop()

    # 데이터 로드
    df = load_data(uploaded_file)
    
    if df is None or df.empty:
        st.error("CSV 파일을 읽을 수 없거나, 데이터가 비어 있습니다. 다른 파일을 업로드해 주세요.")
        st.stop()

    # 데이터 미리보기 및 요약
    show_data_overview(df)

    # 수치형 변수 간 상관관계 히트맵
    show_correlation_heatmap(df)

    # -----------------------------------------
    # 2. 특징/타깃 선택
    # -----------------------------------------
    feature_cols, target_col = choose_features_and_target(df)

    if target_col is None:
        st.warning("타깃(정답) 변수를 선택해주세요.")
        st.stop()

    if not feature_cols:
        st.warning("최소 1개의 입력(특징) 변수를 선택해주세요.")
        st.stop()

    # 타깃과의 상관관계 표 (수치형 타깃일 때)
    show_target_correlations(df, target_col)

    # -----------------------------------------
    # 3. 문제 유형 자동/수동 설정
    # -----------------------------------------
    y = df[target_col]
    inferred_type, reason = infer_problem_type(y)

    st.sidebar.subheader("3️⃣ 문제 유형 설정")
    st.sidebar.info(f"자동 판단 결과: **{inferred_type}** 문제로 추정됨\n\n사유: {reason}")

    problem_choice = st.sidebar.radio(
        "문제 유형 선택",
        options=["자동 판단", "분류", "회귀"],
        index=0,
        help="자동 판단이 마음에 들지 않으면 직접 분류/회귀를 선택할 수 있습니다.",
    )

    if problem_choice == "자동 판단":
        problem_type = inferred_type
    elif problem_choice == "분류":
        problem_type = "classification"
    else:
        problem_type = "regression"

    # -----------------------------------------
    # 4. 알고리즘 선택 & 하이퍼파라미터 설정
    # -----------------------------------------
    algo, params = select_algorithm(problem_type)

    # train/test 비율 설정
    st.sidebar.subheader("4️⃣ 학습/평가 비율 설정")
    test_size = st.sidebar.slider(
        "테스트 데이터 비율 (0.2 = 20%)",
        min_value=0.2,
        max_value=0.4,
        value=0.3,
        step=0.05,
    )
    st.sidebar.caption(
        "테스트 데이터는 모델의 성능을 평가하기 위해 따로 떼어놓는 데이터입니다."
    )

    # -----------------------------------------
    # 5. 모델 학습 버튼
    # -----------------------------------------
    st.sidebar.subheader("5️⃣ 모델 학습 실행")
    if st.sidebar.button("🚀 모델 학습하기"):
        run_training_and_evaluation(
            df=df,
            feature_cols=feature_cols,
            target_col=target_col,
            problem_type=problem_type,
            algo=algo,
            params=params,
            test_size=test_size,
        )
    else:
        st.info("사이드바에서 **🚀 모델 학습하기** 버튼을 누르면 결과가 여기에 표시됩니다.")


if __name__ == "__main__":
    main()
