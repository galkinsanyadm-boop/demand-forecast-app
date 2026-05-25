"""
MVP-сервис прогнозирования спроса на товары.
Логика обучения моделей повторяет ноутбук misha-proj.ipynb (LinearRegression + MLPRegressor).
Сверху добавлены: рекурсивный прогноз на 7-14 дней, сценарии (промо/праздник),
интерактивные графики Plotly и EDA-блок.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ──────────────────────────────────────────────────────────────────────────────
# КОНФИГ
# ──────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Прогноз спроса · MVP",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_CANDIDATES = [
    Path(__file__).parent / "data" / "synthetic_retail_demand_dataset.csv",
    Path(__file__).parent / "synthetic_retail_demand_dataset.csv",
]
SPLIT_DATE = "2024-10-01"

PALETTE = {
    "history": "#2563eb",   # blue
    "forecast": "#f59e0b",  # amber
    "linreg": "#8b5cf6",    # violet
    "mlp": "#10b981",       # green
    "grid": "#e2e8f0",
    "ink": "#0f172a",
}

# ──────────────────────────────────────────────────────────────────────────────
# CSS (минимально, для аккуратной типографики)
# ──────────────────────────────────────────────────────────────────────────────

st.markdown(
    """
    <style>
      /* Скрываем кнопку Deploy в правом верхнем углу */
      [data-testid="stAppDeployButton"] { display: none !important; }
      [data-testid="stDeployButton"] { display: none !important; }

      .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1300px; }
      h1, h2, h3 { letter-spacing: -0.01em; }
      div[data-testid="stMetric"] {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 14px 18px;
      }
      div[data-testid="stMetricLabel"] { color: #64748b; font-weight: 500; }
      div[data-testid="stMetricValue"] { font-size: 1.6rem; color: #0f172a; }
      .stTabs [data-baseweb="tab-list"] { gap: 4px; }
      .stTabs [data-baseweb="tab"] {
        padding: 10px 18px;
        border-radius: 8px 8px 0 0;
      }
      .small-caption { color: #64748b; font-size: 0.85rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────────────────────────────────────
# ДАННЫЕ И МОДЕЛИ
# ──────────────────────────────────────────────────────────────────────────────

def _find_data() -> Path | None:
    for p in DATA_CANDIDATES:
        if p.exists():
            return p
    return None


@st.cache_data(show_spinner="Загружаю данные…")
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["item_id", "date"]).reset_index(drop=True)
    df["dayofweek"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["lag_1"] = df.groupby("item_id")["demand"].shift(1)
    df["lag_7"] = df.groupby("item_id")["demand"].shift(7)
    df = df.dropna().reset_index(drop=True)
    return df


@st.cache_resource(show_spinner="Обучаю модели (один раз)…")
def train_models(df: pd.DataFrame) -> dict:
    """Повторяет логику из misha-proj.ipynb, но возвращает удобные артефакты."""
    df_enc = df.copy()
    item_dummies = pd.get_dummies(df_enc["item_id"], prefix="item_id", drop_first=True)
    df_enc = pd.concat([df_enc, item_dummies], axis=1)

    exclude = {"date", "demand", "item_id", "item_name", "category", "brand", "promo_type"}
    features = [c for c in df_enc.columns if c not in exclude]

    train = df_enc[df_enc["date"] < SPLIT_DATE].copy()
    test = df_enc[df_enc["date"] >= SPLIT_DATE].copy()

    X_train, y_train = train[features], train["demand"]
    X_test, y_test = test[features], test["demand"]

    lin = LinearRegression().fit(X_train, y_train)
    mlp = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "mlp",
                MLPRegressor(
                    hidden_layer_sizes=(64, 32),
                    activation="relu",
                    solver="adam",
                    learning_rate_init=0.001,
                    max_iter=300,
                    early_stopping=True,
                    random_state=42,
                ),
            ),
        ]
    ).fit(X_train, y_train)

    naive_pred = test["lag_1"].values
    seasonal_pred = test["lag_7"].values
    lin_pred = np.clip(lin.predict(X_test), 0, None)
    mlp_pred = np.clip(mlp.predict(X_test), 0, None)

    def m(y, p):
        return {
            "MAE": mean_absolute_error(y, p),
            "RMSE": float(np.sqrt(mean_squared_error(y, p))),
        }

    metrics = {
        "Naive lag-1": m(y_test, naive_pred),
        "Seasonal lag-7": m(y_test, seasonal_pred),
        "Linear Regression": m(y_test, lin_pred),
        "MLP Regressor": m(y_test, mlp_pred),
    }

    test_view = test[["date", "item_id", "demand"]].copy()
    test_view["lin_pred"] = lin_pred
    test_view["mlp_pred"] = mlp_pred
    test_view["naive_pred"] = naive_pred
    test_view["seasonal_pred"] = seasonal_pred

    loss_curve = mlp.named_steps["mlp"].loss_curve_

    return {
        "lin": lin,
        "mlp": mlp,
        "features": features,
        "metrics": metrics,
        "test_view": test_view,
        "loss_curve": loss_curve,
    }


def recursive_forecast(
    model,
    df: pd.DataFrame,
    item_id: str,
    horizon: int,
    features: list[str],
    *,
    promo: bool,
    holiday: bool,
) -> pd.DataFrame:
    """Рекурсивный прогноз на `horizon` дней вперёд для одного товара."""
    item_df = df[df["item_id"] == item_id].sort_values("date").reset_index(drop=True)
    last_date = item_df["date"].max()

    recent_demand = item_df["demand"].tail(8).tolist()  # для lag_1 и lag_7

    base_price = float(item_df["price"].tail(14).mean())
    base_comp_price = float(item_df["competitor_price"].tail(14).mean())
    base_temp = float(item_df["temperature"].tail(14).mean())

    rows = []
    for h in range(1, horizon + 1):
        next_date = last_date + pd.Timedelta(days=h)

        row = {
            "price": base_price * (0.95 if promo else 1.0),
            "competitor_price": base_comp_price,
            "promo": int(promo),
            "discount_depth": 0.10 if promo else 0.0,
            "holiday": int(holiday),
            "payday": 1 if next_date.day in (1, 15) else 0,
            "is_weekend": 1 if next_date.dayofweek >= 5 else 0,
            "temperature": base_temp,
            "stockout": 0,
            "dayofweek": next_date.dayofweek,
            "month": next_date.month,
            "lag_1": recent_demand[-1],
            "lag_7": recent_demand[-7] if len(recent_demand) >= 7 else recent_demand[0],
        }
        for col in features:
            if col.startswith("item_id_"):
                row[col] = 1 if col == f"item_id_{item_id}" else 0

        X_pred = pd.DataFrame([row])[features]
        pred = float(np.clip(model.predict(X_pred)[0], 0, None))

        rows.append({"date": next_date, "demand": pred})
        recent_demand.append(pred)

    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────────
# UI HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def styled_layout(fig: go.Figure, *, height: int = 460) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=20, r=20, t=50, b=20),
        font=dict(family="-apple-system, system-ui, sans-serif", color=PALETTE["ink"]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hoverlabel=dict(bgcolor="white", font_size=13),
    )
    fig.update_xaxes(gridcolor=PALETTE["grid"], zeroline=False)
    fig.update_yaxes(gridcolor=PALETTE["grid"], zeroline=False)
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# ЗАГРУЗКА
# ──────────────────────────────────────────────────────────────────────────────

st.title("📈 Прогноз спроса на товары")
st.markdown(
    "<span class='small-caption'>MVP · нейросетевая модель для оптимизации закупок и складских запасов</span>",
    unsafe_allow_html=True,
)
st.write("")

with st.sidebar:
    st.header("⚙️ Параметры")

    uploaded = st.file_uploader("Загрузить свой CSV", type=["csv"], help="Опционально. По умолчанию используется датасет из папки data/")

if uploaded is not None:
    df_raw = pd.read_csv(uploaded)
    df_raw["date"] = pd.to_datetime(df_raw["date"])
    df_raw = df_raw.sort_values(["item_id", "date"]).reset_index(drop=True)
    df_raw["dayofweek"] = df_raw["date"].dt.dayofweek
    df_raw["month"] = df_raw["date"].dt.month
    df_raw["lag_1"] = df_raw.groupby("item_id")["demand"].shift(1)
    df_raw["lag_7"] = df_raw.groupby("item_id")["demand"].shift(7)
    df = df_raw.dropna().reset_index(drop=True)
else:
    data_path = _find_data()
    if data_path is None:
        st.error(
            "Не найден CSV-файл. Положите `synthetic_retail_demand_dataset.csv` "
            "в папку `data/` рядом с `app.py` или загрузите файл через сайдбар."
        )
        st.stop()
    df = load_data(str(data_path))

models = train_models(df)

# ──────────────────────────────────────────────────────────────────────────────
# САЙДБАР · УПРАВЛЕНИЕ
# ──────────────────────────────────────────────────────────────────────────────

items_meta = df.drop_duplicates("item_id")[["item_id", "item_name", "category"]].reset_index(drop=True)

with st.sidebar:
    st.divider()
    item_id = st.selectbox(
        "Товар",
        options=items_meta["item_id"].tolist(),
        format_func=lambda x: f"{items_meta.loc[items_meta.item_id == x, 'item_name'].iloc[0]}",
    )

    horizon = st.slider("Горизонт прогноза, дней", min_value=7, max_value=14, value=14, step=1)

    model_choice = st.radio(
        "Модель",
        options=["MLP (нейросеть)", "Linear Regression"],
        index=0,
        horizontal=False,
    )

    st.divider()
    st.markdown("**Сценарий**")
    promo = st.checkbox("Промо-акция (−5% к цене)")
    holiday = st.checkbox("Праздничный период")

    st.divider()
    history_days = st.slider("Показ истории, дней", 30, 365, 120, step=30)

# ──────────────────────────────────────────────────────────────────────────────
# ВЫБРАННЫЕ МЕТА
# ──────────────────────────────────────────────────────────────────────────────

item_meta_row = items_meta.loc[items_meta.item_id == item_id].iloc[0]
item_name = item_meta_row["item_name"]
item_category = item_meta_row["category"]

active_model = models["mlp"] if "MLP" in model_choice else models["lin"]
active_metric_key = "MLP Regressor" if "MLP" in model_choice else "Linear Regression"
active_pred_col = "mlp_pred" if "MLP" in model_choice else "lin_pred"

# ──────────────────────────────────────────────────────────────────────────────
# ВКЛАДКИ
# ──────────────────────────────────────────────────────────────────────────────

tab_forecast, tab_compare, tab_eda, tab_about = st.tabs(
    ["🔮 Прогноз", "📊 Сравнение моделей", "📈 Анализ данных", "📋 О проекте"]
)

# ─── ВКЛАДКА 1: ПРОГНОЗ ───────────────────────────────────────────────────────

with tab_forecast:
    item_df = df[df["item_id"] == item_id].copy()

    forecast_df = recursive_forecast(
        active_model, df, item_id, horizon, models["features"], promo=promo, holiday=holiday
    )

    avg_hist = float(item_df["demand"].tail(90).mean())
    avg_forecast = float(forecast_df["demand"].mean())
    delta_pct = (avg_forecast - avg_hist) / avg_hist * 100 if avg_hist > 0 else 0.0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Товар", item_name, item_category)
    c2.metric(f"Σ прогноза, {horizon} дн.", f"{forecast_df['demand'].sum():.0f} шт")
    c3.metric(
        "Средний дневной прогноз",
        f"{avg_forecast:.0f} шт",
        delta=f"{delta_pct:+.1f}% vs история",
    )
    c4.metric(f"MAE модели ({active_metric_key})", f"{models['metrics'][active_metric_key]['MAE']:.2f}")

    st.write("")

    hist = item_df.tail(history_days)
    last_hist_point = pd.DataFrame(
        {"date": [hist["date"].iloc[-1]], "demand": [hist["demand"].iloc[-1]]}
    )
    forecast_for_plot = pd.concat([last_hist_point, forecast_df], ignore_index=True)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=hist["date"],
            y=hist["demand"],
            mode="lines",
            name="История",
            line=dict(color=PALETTE["history"], width=2),
            hovertemplate="<b>%{x|%d %b %Y}</b><br>Спрос: %{y:.0f} шт<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast_for_plot["date"],
            y=forecast_for_plot["demand"],
            mode="lines+markers",
            name="Прогноз",
            line=dict(color=PALETTE["forecast"], width=3, dash="dash"),
            marker=dict(size=7, color=PALETTE["forecast"]),
            hovertemplate="<b>%{x|%d %b %Y}</b><br>Прогноз: %{y:.0f} шт<extra></extra>",
        )
    )
    boundary = hist["date"].iloc[-1]
    fig.add_shape(
        type="line",
        x0=boundary, x1=boundary,
        yref="paper", y0=0, y1=1,
        line=dict(color="#94a3b8", width=1, dash="dot"),
    )
    fig.add_annotation(
        x=boundary, y=1, yref="paper",
        text="начало прогноза",
        showarrow=False,
        font=dict(color="#64748b", size=11),
        xanchor="left", yanchor="bottom",
        xshift=4,
    )
    fig.update_layout(
        title=f"Спрос на «{item_name}» · история и прогноз",
        xaxis_title=None,
        yaxis_title="Спрос, шт/день",
        hovermode="x unified",
    )
    st.plotly_chart(styled_layout(fig, height=480), width="stretch")

    with st.expander("📅 Прогноз по дням (таблица)"):
        show = forecast_df.copy()
        show["date"] = show["date"].dt.strftime("%Y-%m-%d (%a)")
        show["demand"] = show["demand"].round(0).astype(int)
        show.columns = ["Дата", "Прогноз, шт"]
        st.dataframe(show, width="stretch", hide_index=True)

# ─── ВКЛАДКА 2: СРАВНЕНИЕ МОДЕЛЕЙ ─────────────────────────────────────────────

with tab_compare:
    st.subheader("Метрики на тестовом периоде (с 1 октября 2024)")
    st.markdown(
        "<span class='small-caption'>Меньше — лучше. MAE — средняя абсолютная ошибка в штуках, "
        "RMSE — корень среднеквадратичной ошибки.</span>",
        unsafe_allow_html=True,
    )

    metrics_df = (
        pd.DataFrame(models["metrics"]).T.reset_index().rename(columns={"index": "Модель"})
    )

    col_a, col_b = st.columns(2)
    with col_a:
        fig_mae = px.bar(
            metrics_df.sort_values("MAE"),
            x="MAE",
            y="Модель",
            orientation="h",
            text=metrics_df.sort_values("MAE")["MAE"].round(2),
            color="Модель",
            color_discrete_sequence=["#94a3b8", "#cbd5e1", PALETTE["linreg"], PALETTE["mlp"]],
        )
        fig_mae.update_traces(textposition="outside")
        fig_mae.update_layout(showlegend=False, title="MAE по моделям")
        st.plotly_chart(styled_layout(fig_mae, height=320), width="stretch")

    with col_b:
        fig_rmse = px.bar(
            metrics_df.sort_values("RMSE"),
            x="RMSE",
            y="Модель",
            orientation="h",
            text=metrics_df.sort_values("RMSE")["RMSE"].round(2),
            color="Модель",
            color_discrete_sequence=["#94a3b8", "#cbd5e1", PALETTE["linreg"], PALETTE["mlp"]],
        )
        fig_rmse.update_traces(textposition="outside")
        fig_rmse.update_layout(showlegend=False, title="RMSE по моделям")
        st.plotly_chart(styled_layout(fig_rmse, height=320), width="stretch")

    st.dataframe(
        metrics_df.round(2),
        width="stretch",
        hide_index=True,
    )

    st.divider()
    st.subheader(f"Факт vs Прогноз на тестовом периоде · {item_name}")

    tv = models["test_view"]
    tv_item = tv[tv["item_id"] == item_id].sort_values("date")

    if len(tv_item) > 0:
        fig_vs = go.Figure()
        fig_vs.add_trace(
            go.Scatter(
                x=tv_item["date"],
                y=tv_item["demand"],
                mode="lines",
                name="Факт",
                line=dict(color=PALETTE["ink"], width=2),
            )
        )
        fig_vs.add_trace(
            go.Scatter(
                x=tv_item["date"],
                y=tv_item["lin_pred"],
                mode="lines",
                name="Linear Regression",
                line=dict(color=PALETTE["linreg"], width=2, dash="dot"),
            )
        )
        fig_vs.add_trace(
            go.Scatter(
                x=tv_item["date"],
                y=tv_item["mlp_pred"],
                mode="lines",
                name="MLP",
                line=dict(color=PALETTE["mlp"], width=2),
            )
        )
        fig_vs.update_layout(yaxis_title="Спрос, шт/день", hovermode="x unified")
        st.plotly_chart(styled_layout(fig_vs, height=420), width="stretch")

    st.divider()
    st.subheader("Кривая обучения MLP")
    loss = models["loss_curve"]
    fig_loss = go.Figure()
    fig_loss.add_trace(
        go.Scatter(
            x=list(range(1, len(loss) + 1)),
            y=loss,
            mode="lines",
            line=dict(color=PALETTE["mlp"], width=2),
            name="Loss",
        )
    )
    fig_loss.update_layout(
        xaxis_title="Эпоха",
        yaxis_title="Loss (MSE)",
        title=f"Обучение остановилось на {len(loss)} эпохах (early stopping)",
    )
    st.plotly_chart(styled_layout(fig_loss, height=320), width="stretch")

# ─── ВКЛАДКА 3: АНАЛИЗ ДАННЫХ (EDA) ───────────────────────────────────────────

with tab_eda:
    st.subheader("Структура датасета")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Строк", f"{len(df):,}".replace(",", " "))
    k2.metric("Уникальных товаров", df["item_id"].nunique())
    k3.metric("Категорий", df["category"].nunique())
    k4.metric("Период", f"{df['date'].min().strftime('%b %Y')} → {df['date'].max().strftime('%b %Y')}")

    st.divider()
    st.subheader(f"Динамика спроса · {item_name}")

    item_df = df[df["item_id"] == item_id].copy()
    fig_ts = px.line(item_df, x="date", y="demand")
    fig_ts.update_traces(line=dict(color=PALETTE["history"], width=1.5))
    fig_ts.update_layout(yaxis_title="Спрос, шт/день", xaxis_title=None)
    st.plotly_chart(styled_layout(fig_ts, height=320), width="stretch")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Спрос по дням недели**")
        dow_names = ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс"]
        dow = item_df.groupby("dayofweek")["demand"].mean().reset_index()
        dow["День"] = dow["dayofweek"].map(lambda i: dow_names[i])
        fig_dow = px.bar(dow, x="День", y="demand", color="demand", color_continuous_scale="Blues")
        fig_dow.update_layout(yaxis_title="Средний спрос", xaxis_title=None, coloraxis_showscale=False)
        st.plotly_chart(styled_layout(fig_dow, height=300), width="stretch")

    with col2:
        st.markdown("**Спрос по месяцам**")
        month_names = ["Янв", "Фев", "Мар", "Апр", "Май", "Июн", "Июл", "Авг", "Сен", "Окт", "Ноя", "Дек"]
        mo = item_df.groupby("month")["demand"].mean().reset_index()
        mo["Месяц"] = mo["month"].map(lambda i: month_names[i - 1])
        fig_mo = px.bar(mo, x="Месяц", y="demand", color="demand", color_continuous_scale="Blues")
        fig_mo.update_layout(yaxis_title="Средний спрос", xaxis_title=None, coloraxis_showscale=False)
        st.plotly_chart(styled_layout(fig_mo, height=300), width="stretch")

    st.divider()
    st.subheader("Эффект промо и праздников · все товары")

    col3, col4 = st.columns(2)
    with col3:
        promo_effect = df.groupby("promo")["demand"].mean().reset_index()
        promo_effect["Статус"] = promo_effect["promo"].map({0: "Без промо", 1: "С промо"})
        fig_p = px.bar(promo_effect, x="Статус", y="demand",
                       color="Статус", color_discrete_sequence=["#94a3b8", PALETTE["forecast"]])
        fig_p.update_layout(yaxis_title="Средний спрос", xaxis_title=None, showlegend=False)
        st.plotly_chart(styled_layout(fig_p, height=280), width="stretch")

    with col4:
        hol_effect = df.groupby("holiday")["demand"].mean().reset_index()
        hol_effect["Статус"] = hol_effect["holiday"].map({0: "Будни", 1: "Праздник"})
        fig_h = px.bar(hol_effect, x="Статус", y="demand",
                       color="Статус", color_discrete_sequence=["#94a3b8", "#ef4444"])
        fig_h.update_layout(yaxis_title="Средний спрос", xaxis_title=None, showlegend=False)
        st.plotly_chart(styled_layout(fig_h, height=280), width="stretch")

    with st.expander("🔍 Сырые данные (первые 200 строк выбранного товара)"):
        st.dataframe(
            item_df.head(200).reset_index(drop=True),
            width="stretch",
            hide_index=True,
        )

# ─── ВКЛАДКА 4: О ПРОЕКТЕ ─────────────────────────────────────────────────────

with tab_about:
    st.subheader("О проекте")
    st.markdown(
        """
**Кейс:** «Нейросетевая модель для прогнозирования спроса на товары»
**Цель:** прототип сервиса, который по истории продаж предсказывает спрос на 7–14 дней вперёд.

**Что использует приложение**
- Датасет: синтетическая история продаж по 18 товарам за 2 года (2023–2024).
- Модели: `LinearRegression` как бейзлайн и `MLPRegressor` (полносвязная нейросеть, 64→32 нейрона) — обе из ноутбука `misha-proj.ipynb`.
- Прогноз на горизонт > 1 дня делается рекурсивно: предсказание дня *t* подставляется как `lag_1` для дня *t+1*.

**Метрики качества (тест: октябрь–декабрь 2024)**
"""
    )

    metrics_df = (
        pd.DataFrame(models["metrics"]).T.reset_index().rename(columns={"index": "Модель"}).round(2)
    )
    st.dataframe(metrics_df, width="stretch", hide_index=True)

    st.markdown(
        """
**Стек**
- Python 3.11, scikit-learn, pandas
- Streamlit, Plotly

**Ограничения MVP**
- Будущие цена / промо / температура задаются как среднее за последние 14 дней или через тумблеры «сценарий».
- `stockout` в фичах используется как есть (наследовано из ноутбука).
- Один общий MLP на все 18 товаров (item_id через one-hot).
"""
    )
