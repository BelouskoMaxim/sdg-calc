import dash
from dash import dcc, html, Input, Output, State
import dash_mantine_components as dmc
import pandas as pd
import numpy as np
import random
from dash_iconify import DashIconify
import plotly.express as px
from dash import dcc, html
import plotly.graph_objects as go

quarters_total = 20
# ===== Конфиг =====
CONFIG = {
    "quarters_total": 20,
    "life_min": 13,
    "life_max": 17,
    "stages": {
        "q1": {
            "range": (1, 1),
            "revenue_min": 50000,
            "revenue_max": 150000,
            "cqgr_min": None,
            "cqgr_max": None,
            "margin_min": -2.5,
            "margin_max": -1.0,
        },
        "q24": {
            "range": (2, 4),
            "cqgr_min": 0.7,
            "cqgr_max": 1.0,
            "margin_min": -0.4,
            "margin_max": 0.2,
        },
        "q58": {
            "range": (5, 8),
            "cqgr_min": 0.4,
            "cqgr_max": 0.6,
            "margin_min": 0.4,
            "margin_max": 0.5,
        },
        "q912": {
            "range": (9, 12),
            "cqgr_min": 0.1,
            "cqgr_max": 0.2,
            "margin_min": 0.4,
            "margin_max": 0.5,
        },
        "late": {
            "range": (13, 20),
            "cqgr_min": 0.01,
            "cqgr_max": 0.09,
            "margin_min": 0.5,
            "margin_max": 0.6,
        }
    }, 
    "startegyInputs": {
        "startups_per_quarter": 5,
        "stage1_check": 100000,
        "survival_stage2": 50,   # %
        "stage2_check": 500000,
        "survival_stage3": 20,   # %
        "stage3_check": 2000000,
    }
}

NET_VALUE_COMPACT = "200M USD$"          # текстовая компактная запись
NET_VALUE_NUMERIC = 200_000_000          # если захочешь форматировать числом

def simulate_startup_track_v2(config=CONFIG):
    q_total = config["quarters_total"]
    lifetime = random.randint(config["life_min"], config["life_max"])
    mrr = [None] * q_total
    current = None

    # Проходим по стадиям
    for stage, params in config["stages"].items():
        q_start, q_end = params["range"]

        # Q1 — инициализация revenue
        if stage == "q1":
            current = random.uniform(params["revenue_min"], params["revenue_max"])
            margin = random.uniform(params["margin_min"], params["margin_max"])
            mrr[q_start - 1] = {
                "revenue": current,
                "net": current * margin
            }
            continue

        # Для остальных стадий
        cqgr = random.uniform(params["cqgr_min"], params["cqgr_max"])
        for i in range(q_start - 1, min(q_end, lifetime)):
            current *= (1 + cqgr)
            margin = random.uniform(params["margin_min"], params["margin_max"])
            mrr[i] = {
                "revenue": current,
                "net": current * margin
            }

    return mrr
def generate_data_v2(config=CONFIG):
    rows = []
    startup_id = 0

    for q in range(config['quarters_total']):
        for _ in range(config["startegyInputs"]["startups_per_quarter"]):
            startup_id += 1
            track = simulate_startup_track_v2(config=config)

            # сдвигаем трек стартапа к кварталу инвестиций
            shifted_track = [None] * q + track[:config['quarters_total'] - q]

            rows.append({
                "startup": f"Startup {startup_id}",
                "track": shifted_track
            })

    return rows
def generate_data_df(config=CONFIG):
    """
    Генерация данных и возврат двух DataFrame:
    - df_revenue: значения revenue по стартапам и кварталам
    - df_net: значения net по стартапам и кварталам
    """
    startups = generate_data_v2(config=config)

    cols = [f"Q{i+1}" for i in range(config["quarters_total"])]

    revenue_rows = []
    net_rows = []
    names = []

    for s in startups:
        names.append(s["startup"])
        rev_track = []
        net_track = []
        for val in s["track"]:
            if val is None:
                rev_track.append(None)
                net_track.append(None)
            else:
                rev_track.append(val["revenue"])
                net_track.append(val["net"])
        revenue_rows.append(rev_track)
        net_rows.append(net_track)

    df_revenue = pd.DataFrame(revenue_rows, columns=cols, index=names)
    df_net = pd.DataFrame(net_rows, columns=cols, index=names)

    return df_revenue, df_net
def apply_stage_mask_by_year(
    revenue_df: pd.DataFrame,
    net_df: pd.DataFrame,
    quarters_per_year: int = 4,
    cum_shares=None,      # (после Q1, после Q4, полный трек)
    cuts=(1, 4, None),           # сколько кварталов жизни оставить: 1, 4, None=без обрезки
    random_state=None
):
    """
    Маскирует треки стартапов по стадиям, равномерно по годам запуска (блоки по 4 квартала).

    - shares: доли в каждой годовой корзине (сумма ≈ 1.0)
      0 -> оставляем только 1 квартал жизни (после этого NaN)
      1 -> оставляем 4 квартала жизни (после этого NaN)
      2 -> полный трек
    - cuts: соответствующие "длины жизни" в кварталах (относительно старта)
      None = не обрезать

    Требования к входу:
      revenue_df, net_df — строки = стартапы, колонки = глобальные кварталы Q1..Qn,
      ряды уже сдвинуты (до старта — NaN).
    """

    assert revenue_df.shape == net_df.shape, "revenue_df и net_df должны быть одинаковой формы"
    n_rows, n_cols = revenue_df.shape
    cols = list(revenue_df.columns)

    rng = np.random.default_rng(random_state)


    # --- Преобразуем cum_shares → shares ---
    if cum_shares is not None:
        if len(cum_shares) != 2:
            raise ValueError("cum_shares должно быть длины 2: (p2, p3)")
        p2, p3 = cum_shares
        shares = (1 - p2, p2 - p3, p3)

    if shares is None:
        raise ValueError("Нужно задать либо shares, либо cum_shares")

    # Находим стартовый глобальный квартал каждого стартапа (первая НЕ NaN ячейка)
    mask_valid = revenue_df.notna().values
    start_pos = mask_valid.argmax(axis=1)  # индекс колонки первого True
    # Если у строки все NaN, argmax вернёт 0 — проверим и исключим такие случаи
    all_nan_rows = (~mask_valid).all(axis=1)
    if all_nan_rows.any():
        raise ValueError("Обнаружены строки без данных (все NaN). Проверь генерацию данных.")

    # Год запуска (0-базный) по глобальному кварталу старта
    start_year = start_pos // quarters_per_year

    # Подготовим копии для маскирования
    rev_out = revenue_df.copy()
    net_out = net_df.copy()

    # Группируем по годам запуска и применяем 50/30/20
    df_index = np.arange(n_rows)
    for yr in np.unique(start_year):
        idxs = df_index[start_year == yr]
        if len(idxs) == 0:
            continue

        # Перемешаем индексы внутри года
        shuffled = idxs.copy()
        rng.shuffle(shuffled)

        # Считаем численности по долям
        n = len(shuffled)
        n_after_q1 = int(round(n * (shares[0])))  # ~50%
        n_after_q4 = int(round(n * (shares[1])))  # ~30%
        # оставшиеся — полный трек
        n_full = max(0, n - n_after_q1 - n_after_q4)

        # Разбиваем
        group_after_q1 = shuffled[:n_after_q1]
        group_after_q4 = shuffled[n_after_q1:n_after_q1 + n_after_q4]
        group_full     = shuffled[n_after_q1 + n_after_q4:]

        # Функция маскирования "после k кварталов жизни"
        def _mask_after_k(row_idx: int, keep_k: int | None):
            s = start_pos[row_idx]  # глобальный индекс старта
            if keep_k is None:
                return  # полный трек, ничего не делаем
            cut_from = s + keep_k  # начиная ОТСЮДА ставим NaN
            if cut_from < n_cols:
                rev_out.iloc[row_idx, cut_from:] = np.nan
                net_out.iloc[row_idx, cut_from:] = np.nan

        # Применяем:
        for r in group_after_q1:
            _mask_after_k(r, cuts[0])  # оставить 1 квартал жизни
        for r in group_after_q4:
            _mask_after_k(r, cuts[1])  # оставить 4 квартала жизни
        for r in group_full:
            _mask_after_k(r, cuts[2])  # None -> без обрезки

        # (Опционально можно убедиться, что в каждом годе примерно 50/30/20)
        # print(f"year {yr}: n={n} -> 1Q:{len(group_after_q1)}, 4Q:{len(group_after_q4)}, full:{len(group_full)}")

    return rev_out, net_out
def simulate(n=100, config=CONFIG):
    """
    Run n simulations.

    Возвращает два массива NumPy формы (n, quarters_total):
      - revenue_sims: суммы revenue по кварталам
      - net_sims: суммы net по кварталам
    """
    revenue_results = np.zeros((n, quarters_total))
    net_results = np.zeros((n, quarters_total))

    for sim in range(n):
        # Генерируем данные стартапов
        df_rev, df_net = generate_data_df(config=config)
        df_rev_masked, df_net_masked = apply_stage_mask_by_year( df_rev, df_net, quarters_per_year=4, 
                                                                cum_shares=(config["startegyInputs"]["survival_stage2"] / 100,  config["startegyInputs"]["survival_stage3"] / 100),
                                                                #cum_shares=(0.6, 0.2),
                                                                cuts=(1, 4, None), random_state=None)
        # Складываем по всем стартапам => получаем ряд суммарного ревенью/нетто
        revenue_results[sim, :] = df_rev_masked.sum(axis=0, skipna=True).values
        net_results[sim, :] = df_net_masked.sum(axis=0, skipna=True).values

    # ✅ Возвращаем результаты
    return revenue_results, net_results      
def plot_boxplots(net_sims):
    quarters_total = net_sims.shape[1]

    # ===== Revenue =====
    # df_rev = pd.DataFrame(rev_sims, columns=[f"Q{i+1}" for i in range(quarters_total)])
    # df_rev_long = df_rev.melt(var_name="Quarter", value_name="Revenue")

    # fig_rev = px.box(
    #     df_rev_long,
    #     x="Quarter",
    #     y="Revenue",
    #     title="Box plot по кварталам (Revenue)"
    # )
    # fig_rev.show()

    # ===== Net =====
    df_net = pd.DataFrame(net_sims, columns=[f"Q{i+1}" for i in range(quarters_total)])
    df_net_long = df_net.melt(var_name="Quarter", value_name="Net")

    fig_net = px.box(
        df_net_long,
        x="Quarter",
        y="Net",
    )

    # убираем стандартный фон
    fig_net.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis_title=None,
        yaxis_title=None,
        showlegend=False,
        margin=dict(l=20, r=20, t=20, b=20)
    )

    # группировка по годам (каждые 4 квартала)
    quarters = df_net_long["Quarter"].unique()
    quarters_sorted = sorted(quarters, key=lambda q: int(q[1:]))  # Q1, Q2, ...
    year_blocks = [quarters_sorted[i:i+4] for i in range(0, len(quarters_sorted), 4)]

    # Добавляем фоновые прямоугольники по блокам кварталов
    for i, block in enumerate(year_blocks):
        x0 = block[0]
        x1 = block[-1]
        fig_net.add_vrect(
            x0=x0,
            x1=x1,
            fillcolor="lightgrey",
            opacity=0.15,
            layer="below",
            line_width=0,
        )

    # убрать grid
    fig_net.update_xaxes(showgrid=False)
    fig_net.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.05)")
    fig_net.update_traces(marker_color="green", line_color="green")


    
    return fig_net
def plot_startup_revenue_net(data):
    """
    Строит график Revenue и Net (в млн USD) по кварталам.
    
    data: список словарей или None
          [{'revenue': float, 'net': float}, None, ...]
    """
    # Извлекаем значения (None остаются None, Plotly сам сделает разрыв)
    quarters = list(range(1, len(data) + 1))
    revenue = [d['revenue'] / 1_000_000 if d else None for d in data]  # млн
    net = [d['net'] / 1_000_000 if d else None for d in data]

    # Создаем график
    fig = go.Figure()

    # Revenue (синяя линия)
    fig.add_trace(go.Scatter(
        x=quarters, y=revenue,
        mode='lines+markers',
        name='Revenue',
        line=dict(color='blue', width=3),
        hovertemplate='Quarter: %{x}<br>Revenue: %{y:.2f}M<extra></extra>',
        showlegend=False
    ))

    # Net (зелёная линия)
    fig.add_trace(go.Scatter(
        x=quarters, y=net,
        mode='lines+markers',
        name='Net',
        line=dict(color='green', width=3),
        hovertemplate='Quarter: %{x}<br>Net: %{y:.2f}M<extra></extra>',
        showlegend=False
    ))

    # Находим последние точки, где есть данные
    last_rev_idx = max(i for i, v in enumerate(revenue) if v is not None)
    last_net_idx = max(i for i, v in enumerate(net) if v is not None)

    # Подписи к линиям
    fig.add_annotation(
        x=quarters[last_rev_idx], y=revenue[last_rev_idx],
        text="Revenue", showarrow=False,
        font=dict(color="blue", size=14),
        xanchor="left", yanchor="middle", xshift=20
    )
    fig.add_annotation(
        x=quarters[last_net_idx], y=net[last_net_idx],
        text="Net", showarrow=False,
        font=dict(color="green", size=14),
        xanchor="left", yanchor="middle", xshift=20
    )

    # Настройки графика
    fig.update_layout(
        title="Revenue & Net by Quarter (Startup)",
        xaxis_title="Quarter",
        yaxis_title="Revenue / Net, M USD$",
        template="plotly_white",
        xaxis=dict(tickmode="array", tickvals=quarters, ticktext=quarters)
    )

    fig.show()
#rev, net = simulate(n=1000)
#figNet = plot_boxplots(net)

# === KPI cards ===
cards = dmc.Stack(
    gap="md", 
    children=[
        dmc.Card(
            shadow="sm",
            padding="lg",
            radius="md",
            withBorder=True,
            children=[
                dmc.Group(
                    [
                        #DashIconify(icon="mdi:cash-multiple", width=28, color="blue"),
                        dmc.Text("TOTAL INVESTMENTS", tt="uppercase", fw=700, c="red"),
                    ],
                    gap="sm"
                ),
                dmc.Text(children=[], id="total-invest", fw=900, fz="xl", c="dark", mt="xs"),
                dmc.Text(children=[], id="total-count-invest", size="sm", c="dimmed", mt="sm"),
            ],
        ),
        dmc.Card(
            shadow="sm",
            padding="lg",
            radius="md",
            withBorder=True,
            children=[
                dmc.Group(
                    [
                        #DashIconify(icon="mdi:chart-line", width=28, color="teal"),
                        dmc.Text("TOTAL NET", tt="uppercase", fw=700, c="green"),
                    ],
                    gap="sm"
                ),
                dmc.Text(children=[], id='total-net', fw=900, fz="xl", c="dark", mt="xs"),
                dmc.Text("Net profit after costs", size="sm", c="dimmed", mt="sm"),
            ],
        ),
    ]
)

strategy_card = dmc.Card(
    withBorder=True,
    shadow="sm",
    radius="md",
    p="lg",
    bg="#fff8e1",
    children=[
        dmc.Text(
            "🎛️ Investment Strategy Parameters",
            fw=700,
            fz="lg",
            mb="md"
        ),

        dmc.Flex(
            gap="md",
            wrap="wrap",
            justify="flex-start",
            children=[
                dmc.NumberInput(
                    id="input-startups",
                    label="Startups / Quarter",
                    value=5,
                    min=1,
                    step=1,
                    w=140,
                    styles={"label": {"minWidth": "150px"}}
                ),
                dmc.NumberInput(
                    id="input-stage1",
                    label="Stage 1 Check ($)",
                    value=100000,
                    min=10000,
                    step=10000,
                    w=160,
                    thousandSeparator=True,
                    styles={"label": {"minWidth": "150px"}}
                ),
                dmc.NumberInput(
                    id="input-survival2",
                    label="Survival to Stage 2 (%)",
                    value=50,
                    min=0,
                    max=100,
                    step=5,
                    w=160,
                    suffix=" %",
                    styles={"label": {"minWidth": "150px"}}
                ),
                dmc.NumberInput(
                    id="input-stage2",
                    label="Stage 2 Check ($)",
                    value=500000,
                    min=50000,
                    step=50000,
                    w=160,
                    thousandSeparator=True,
                    styles={"label": {"minWidth": "150px"}}
                ),
                dmc.NumberInput(
                    id="input-survival3",
                    label="Survival to Stage 3 (%)",
                    value=20,
                    min=0,
                    max=100,
                    step=5,
                    w=160,
                    suffix=" %",
                    styles={"label": {"minWidth": "150px"}}
                ),
                dmc.NumberInput(
                    id="input-stage3",
                    label="Stage 3 Check ($)",
                    value=2000000,
                    min=100000,
                    step=100000,
                    w=160,
                    thousandSeparator=True,
                    styles={"label": {"minWidth": "150px"}}
                )
            ],
        )
    ],
    style={"marginTop": 20, "width": "100%"}
)








# === Dash App ===
app = dash.Dash(__name__)

app.layout = dmc.MantineProvider(
    children=dmc.Container(
        size="xl",
        children=[
            # Храним конфиг
            dcc.Store(id="config-store", data=CONFIG),

            dmc.Title("Investment Revenue Calculator", order=2, mb=20),
            dmc.Box(
                # === Layout 2 колонки ===
                dmc.Grid(
                    gutter="md",
                    children=[
                        dmc.GridCol(cards, span=3),   # 30% ширины
                        dmc.GridCol(dcc.Graph(id='net-box-plot', figure=None), span=9),  # 70% ширины
                    ],
                ),
                mt="md",   # margin-top
                mb="lg",   # margin-bottom
                

            ),
            dmc.Button("🔄 Generate Simulation", id="btn-recalc", variant="filled", color="yellow", disabled=False, loading=False, w="100%" ),
            dmc.Space(h=30),
            strategy_card,
            dmc.Space(h=30),
        


            # ====== Inputs ======
            dmc.Fieldset(
                children=[
                    dmc.Title(f"Revenue | NET Benchmark Params", order=3),

                    dmc.Group(grow=True, children=[
                        dmc.NumberInput(id="initial-revenue-min", label="Q1 Revenue Min", value=50000, step=10000, min=0),
                        dmc.NumberInput(id="initial-revenue-max", label="Q1 Revenue Max", value=150000, step=10000, min=0),
                        dmc.NumberInput(id="initial-margin-min", label="MARGIN INIT Min", value=-2.5, step=0.1, min=-3),
                        dmc.NumberInput(id="initial-margin-max", label="MARGIN INIT Max", value=-1, step=0.1, min=-3),
                    ]),

                    dmc.Group(grow=True, children=[
                        dmc.NumberInput(id="CQGR-q24-min", label="CQGR Q2-Q4 Min", value=0.7, step=0.1, min=0),
                        dmc.NumberInput(id="CQGR-q24-max", label="CQGR Q2-Q4 Max", value=1.0, step=0.1, min=0),
                        dmc.NumberInput(id="margin-q24-min", label="MARGIN Q2-Q4 Min", value=-0.4, step=0.1, min=-3),
                        dmc.NumberInput(id="margin-q24-max", label="MARGIN Q2-Q4 Max", value=0.2, step=0.1, min=-3),
                    ]),

                    dmc.Group(grow=True, children=[
                        dmc.NumberInput(id="CQGR-q58-min", label="CQGR Q5-Q8 Min", value=0.4, step=0.1, min=0),
                        dmc.NumberInput(id="CQGR-q58-max", label="CQGR Q5-Q8 Max", value=0.6, step=0.1, min=0),
                        dmc.NumberInput(id="margin-q58-min", label="MARGIN Q5-Q8 Min", value=0.2, step=0.1, min=-2),
                        dmc.NumberInput(id="margin-q58-max", label="MARGIN Q5-Q8 Max", value=0.4, step=0.1, min=-2),
                    ]),

                    dmc.Group(grow=True, children=[
                        dmc.NumberInput(id="CQGR-q912-min", label="CQGR Q9-Q12 Min", value=0.1, step=0.05, min=0),
                        dmc.NumberInput(id="CQGR-q912-max", label="CQGR Q9-Q12 Max", value=0.2, step=0.05, min=0),
                        dmc.NumberInput(id="margin-q912-min", label="MARGIN Q9-Q12 Min", value=0.3, step=0.1, min=-2),
                        dmc.NumberInput(id="margin-q912-max", label="MARGIN Q9-Q12 Max", value=0.5, step=0.1, min=-2),
                    ]),

                    dmc.Group(grow=True, children=[
                        dmc.NumberInput(id="CQGR-qlate-min", label="CQGR Q13+ Min", value=0.01, step=0.01, min=0),
                        dmc.NumberInput(id="CQGR-qlate-max", label="CQGR Q13+ Max", value=0.08, step=0.01, min=0),
                        dmc.NumberInput(id="margin-qlate-min", label="MARGIN Q13+ Min", value=0.4, step=0.1, min=0),
                        dmc.NumberInput(id="margin-qlate-max", label="MARGIN Q13+ Max", value=0.6, step=0.1, min=0),
                    ]),

                    dmc.Group(children=[
                       # dmc.Button("🔄 Random Startup Track", id="btn-generate-startup-track", variant="filled"),
                       # dcc.Graph(id='startup-track-plot', figure=None)
                    ]),

                    dmc.Grid([
                            # Левая колонка (70%) — график
                            dmc.GridCol(
                                dcc.Graph(figure=None, id='startup-track-plot'),
                                span=8  # 8/12 = 66.6% ≈ 70%
                            ),
                            # Правая колонка (30%) — описание
                            dmc.GridCol(
                                children=[

                                    dmc.Blockquote(
                                        children=[
                                            dmc.Text([
                                                "We simulate the startup track, assuming that a project survives ",
                                                dmc.Text("13–17 quarters", fw="bold", span=True),
                                                " (approximately ",
                                                dmc.Text("3–4 years", fw="bold", span=True),
                                                ")."
                                            ]),
                                            dmc.Space(h=10),
                                            dmc.Text([
                                                "During this period, the project passes through ",
                                                dmc.Text("5 distinct stages", fw="bold", span=True)
                                            ]),
                                            dmc.Space(h=5),
                                            dmc.Text("Each stage has:"),
                                            dmc.Text([
                                                "- Its own ",
                                                dmc.Text("growth rate (CQGR)", fw="bold", span=True)
                                            ]),
                                            dmc.Text([
                                                "- A defined ",
                                                dmc.Text("margin percentage of Revenue", fw="bold", span=True)
                                            ]),
                                        ],
                                        color="gray"
                                    ),
                                    dmc.Button("🔄 Random Startup Track", id="btn-generate-startup-track", variant="outline", color="grey", fullWidth=True, mt="md", mb="md"   ),

                                ],
                                span=4  # 4/12 = 33.3% ≈ 30%
                            )
                        ], gutter="md", mt=50,  mb=20,   ml=10,  mr=10  ),

                    
                    dmc.Grid([
                            # Левая колонка (70%) — график
                             dmc.GridCol(
                                dmc.ScrollArea(
                                    dmc.Table(
                                        id="table-simulation",
                                        highlightOnHover=True,
                                        striped=True,
                                        withColumnBorders=True,
                                        style={"minWidth": "100%"}  # таблица растянется только в границах обертки
                                    ),
                                    style={"width": "100%", "maxWidth": "100%", "overflowX": "auto"}  # скролл если много колонок
                                ),
                                span=8  # 70%
                            ),
                            # Правая колонка (30%) — описание
                            dmc.GridCol(
                                children=[

                                    dmc.Blockquote(
                                        children=[
                                            dmc.Text([
                                                "Now we simulate the portfolio for the first year with ",
                                                dmc.Text("5 projects", fw="bold", span=True),
                                                " per quarter."
                                            ]),
                                            dmc.Space(h=10),
                                            dmc.Text([
                                                "Of these projects, ",
                                                dmc.Text("50%", fw="bold", span=True),
                                                " survive to the second stage and ",
                                                dmc.Text("20%", fw="bold", span=True),
                                                " survive to the third stage."
                                            ]),
                                            dmc.Space(h=10),
                                            dmc.Text("Revenue is illustrated by quarter for each project."),
                                            dmc.Space(h=10),
                                            dmc.Text([
                                                "Investments per quarter:",
                                            ]),
                                            dmc.Text([
                                                "- Q1: ", dmc.Text("$100k USD", fw="bold", span=True)
                                            ]),
                                            dmc.Text([
                                                "- Q2: ", dmc.Text("$500k USD", fw="bold", span=True)
                                            ]),
                                            dmc.Text([
                                                "- Q5: ", dmc.Text("$2M USD", fw="bold", span=True)
                                            ]),
                                        ],
                                        color="gray"
                                    ),
                                    dmc.Button("🔄 Random Simulation For 1 Year", id="btn-simulation-1y", variant="outline", color="grey", fullWidth=True, mt="md", mb="md"   ),

                                ],
                                span=4  # 4/12 = 33.3% ≈ 30%
                            )
                        ], gutter="md", mt=50,  mb=20,   ml=10,  mr=10  ),


                        

                ]
            ),
           
            
            
            

            dmc.Space(h=30),

            # ====== Table ======
            dmc.ScrollArea(
                style={"height": 300},
                children=html.Div(id="revenue-table"),
            ),
            dmc.Space(h=30),
            dmc.ScrollArea(
                style={"height": 300},
                children=html.Div(id="net-table"),
            ),
            dmc.Space(h=30),

            html.Pre(id="config-display"),
        ]
    )
)

@app.callback(
    Output("table-simulation", "children"),
    Input("btn-simulation-1y", "n_clicks"),
    State("config-store", "data")
        
)
def update_config(n_clicks, config):
    df_rev, df_net = generate_data_df(config=config)
    df_rev_masked, df_net_masked = apply_stage_mask_by_year(
        df_rev, df_net, quarters_per_year=4,
        cum_shares=(
            config["startegyInputs"]["survival_stage2"] / 100,
            config["startegyInputs"]["survival_stage3"] / 100,
        ),
        cuts=(1, 4, None), random_state=None
    )

    n = int(len(df_rev_masked) * 0.2)
    df_1Y = df_rev_masked[:n]

    

    

    # форматируем: первая колонка — имя стартапа, остальные — $M с 2 знаками
    def fmt(val, is_name=False):
        if is_name:  # для названий стартапов
            return val if val is not None else ""
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return ""
        return f"${val/1_000_000:.2f}M"

    head = dmc.TableThead(
        dmc.TableTr([
            dmc.TableTh(col) for col in df_1Y.columns
        ])
    )

    body = dmc.TableTbody([
        dmc.TableTr([
            dmc.TableTd(fmt(df_1Y.iloc[i, j]), style={"fontSize": "12px", "textAlign": "center"}) for j in range(len(df_1Y.columns))
        ]) for i in range(len(df_1Y))
    ])

    return [head, body]


# --- Callback для обновления CONFIG ---
@app.callback(
    Output("startup-track-plot", "figure"),
    Input("btn-generate-startup-track", "n_clicks"),
    State("config-store", "data")
        
)
def update_config(n_clicks, config):
    data = simulate_startup_track_v2(config=config)

    # Извлекаем значения (None остаются None, Plotly сам сделает разрыв)
    quarters = list(range(1, len(data) + 1))
    revenue = [d['revenue'] / 1_000_000 if d else None for d in data]  # делим на млн
    net = [d['net'] / 1_000_000 if d else None for d in data]  

    # Создаем график
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=quarters, y=revenue,
        mode='lines+markers',
        name='Revenue',
        line=dict(color='blue'),
        hovertemplate='Quarter: %{x}<br>Revenue: %{y:.2f}M<extra></extra>',
        showlegend=False
    ))

    fig.add_trace(go.Scatter(
        x=quarters, y=net,
        mode='lines+markers',
        name='Net',
        line=dict(color='green'),
        hovertemplate='Quarter: %{x}<br>Net: %{y:.2f}M<extra></extra>',
        showlegend=False
    ))

    # Находим последнюю точку, где есть данные
    last_rev_idx = max(i for i, v in enumerate(revenue) if v is not None)
    last_net_idx = max(i for i, v in enumerate(net) if v is not None)

    # Добавляем подписи к линиям
    fig.add_annotation(
        x=quarters[last_rev_idx], y=revenue[last_rev_idx],
        text="Revenue", showarrow=False, font=dict(color="blue", size=14), xanchor="left", yanchor="middle", xshift=20
    )

    fig.add_annotation(
        x=quarters[last_net_idx], y=net[last_net_idx],
        text="Net", showarrow=False, font=dict(color="green", size=14), xanchor="left", yanchor="middle", xshift=20
    )

    # --- Background стадии ---
    # вычисляем максимум только по числам
    all_values = [v for v in (revenue + net) if v is not None]
    y_max = max(all_values) * 1.05 if all_values else 1

    # --- Background стадии ---
    stages = [
        dict(x0=1,  x1=1,  color="rgba(255,0,0,0.1)",   label="PMF"),
        dict(x0=2,  x1=4,  color="rgba(255,165,0,0.1)", label="LTV/CAC > 3"),
        dict(x0=5,  x1=9,  color="rgba(0,200,0,0.1)",   label="Go-To-Market"),
        dict(x0=10, x1=12, color="rgba(0,150,255,0.1)", label="Expansion"),
        dict(x0=13, x1=17, color="rgba(128,128,128,0.1)", label="Late Stage"),
    ]


    for s in stages:
        # фон
        fig.add_vrect(
            x0=s["x0"] - 0.5, x1=s["x1"] + 0.5,
            fillcolor=s["color"], opacity=0.8,
            layer="below", line_width=0
        )
        # подпись по центру блока, выше графика

       
        fig.add_annotation(

            x=(s["x0"] + s["x1"]) / 2,
            y=y_max,
            text=s["label"],
            showarrow=False,
            font=dict(size=12, color="black"),
            xanchor="center",
            yanchor="bottom"
        )

    # Настройка оформления
    # Настройка оформления
    fig.update_layout(
        xaxis_title="Quarter",
        yaxis_title="Revenue / NET, M USD$",
        template="plotly_white",
        xaxis=dict(tickmode="array", tickvals=quarters, ticktext=[f"Q{i}" for i in quarters]),
        height=300,  # высота в пикселях
        margin=dict(l=0, r=0, t=0, b=0) 
    )

    return fig

# --- Callback для обновления CONFIG ---
@app.callback(
    Output("config-store", "data"),
    Output("config-display", "children"),
    Input("input-startups", "value"),
    Input("input-stage1", "value"),
    Input("input-survival2", "value"),
    Input("input-stage2", "value"),
    Input("input-survival3", "value"),
    Input("input-stage3", "value"),
    Input("initial-revenue-min", "value"),
    Input("initial-revenue-max", "value"),
    Input("initial-margin-min", "value"),
    Input("initial-margin-max", "value"),

    Input("CQGR-q24-min", "value"),
    Input("CQGR-q24-max", "value"),
    Input("margin-q24-min", "value"),
    Input("margin-q24-max", "value"),

    Input("CQGR-q58-min", "value"),
    Input("CQGR-q58-max", "value"),
    Input("margin-q58-min", "value"),
    Input("margin-q58-max", "value"),

    Input("CQGR-q912-min", "value"),
    Input("CQGR-q912-max", "value"),
    Input("margin-q912-min", "value"),
    Input("margin-q912-max", "value"),

    Input("CQGR-qlate-min", "value"),
    Input("CQGR-qlate-max", "value"),
    Input("margin-qlate-min", "value"),
    Input("margin-qlate-max", "value"),
    State("config-store", "data"),
)
def update_config(startups, stage1, surv2, stage2, surv3, stage3, init_rev_min, init_rev_max, init_margin_min, init_margin_max,
    cqgr24_min, cqgr24_max, margin24_min, margin24_max,
    cqgr58_min, cqgr58_max, margin58_min, margin58_max,
    cqgr912_min, cqgr912_max, margin912_min, margin912_max,
    cqgrlate_min, cqgrlate_max, marginlate_min, marginlate_max, config):

    config["startegyInputs"]["startups_per_quarter"] = startups
    config["startegyInputs"]["stage1_check"] = stage1
    config["startegyInputs"]["survival_stage2"] = surv2
    config["startegyInputs"]["stage2_check"] = stage2
    config["startegyInputs"]["survival_stage3"] = surv3
    config["startegyInputs"]["stage3_check"] = stage3

     # Q1
    config["stages"]["q1"]["revenue_min"] = init_rev_min
    config["stages"]["q1"]["revenue_max"] = init_rev_max
    config["stages"]["q1"]["margin_min"] = init_margin_min
    config["stages"]["q1"]["margin_max"] = init_margin_max

    # Q2-Q4
    config["stages"]["q24"]["cqgr_min"] = cqgr24_min
    config["stages"]["q24"]["cqgr_max"] = cqgr24_max
    config["stages"]["q24"]["margin_min"] = margin24_min
    config["stages"]["q24"]["margin_max"] = margin24_max

    # Q5-Q8
    config["stages"]["q58"]["cqgr_min"] = cqgr58_min
    config["stages"]["q58"]["cqgr_max"] = cqgr58_max
    config["stages"]["q58"]["margin_min"] = margin58_min
    config["stages"]["q58"]["margin_max"] = margin58_max

    # Q9-Q12
    config["stages"]["q912"]["cqgr_min"] = cqgr912_min
    config["stages"]["q912"]["cqgr_max"] = cqgr912_max
    config["stages"]["q912"]["margin_min"] = margin912_min
    config["stages"]["q912"]["margin_max"] = margin912_max

    # Late
    config["stages"]["late"]["cqgr_min"] = cqgrlate_min
    config["stages"]["late"]["cqgr_max"] = cqgrlate_max
    config["stages"]["late"]["margin_min"] = marginlate_min
    config["stages"]["late"]["margin_max"] = marginlate_max



    return config, str(config)

@app.callback(
    [
        Output("total-invest", "children"),
        Output("total-count-invest", "children"),
        Output("total-net", "children"),
        Output("net-box-plot", "figure")
    ],
    Input("btn-recalc", "n_clicks"),
    State("config-store", "data")
)
def recalc(n_clicks, config):
    startups_total = config['startegyInputs']["startups_per_quarter"] * quarters_total

    # Стадия 1
    stage1_count = startups_total
    stage1_total = stage1_count * config['startegyInputs']["stage1_check"]

    # Стадия 2
    stage2_count = int(stage1_count * config['startegyInputs']["survival_stage2"] / 100)
    stage2_total = stage2_count * config['startegyInputs']["stage2_check"]

    # Стадия 3
    stage3_count = int(stage1_count * config['startegyInputs']["survival_stage3"] / 100)
    stage3_total = stage3_count * config['startegyInputs']["stage3_check"]

    total_investment = stage1_total + stage2_total + stage3_total


    rev, net = simulate(n=1000, config=config)
    # 1. Суммируем по кварталам для каждой симуляции
    net_sum_per_sim = net.sum(axis=1)  # shape = (n,)

    # 2. Находим медиану по всем симуляциям
    net_median_total = np.median(net_sum_per_sim)
    q1 = np.percentile(net_sum_per_sim, 25)
    q3 = np.percentile(net_sum_per_sim, 75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    
    figNet = plot_boxplots(net)





    return f"{round(total_investment / 1000000, 0)}M, USD", f"{stage1_count} | {stage2_count} | {stage3_count}", f"{round(net_median_total / 1000000, 0)}M, USD",  figNet




if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8050, debug=True)
