import pathlib
import os

import pandas as pd
import numpy as np

import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
from dash.dependencies import Input, Output, State

import constants

# app initialize
app = dash.Dash(
    __name__,
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1.0"}
    ],
)
server = app.server
app.config["suppress_callback_exceptions"] = True

# mapbox
mapbox_access_token = "pk.eyJ1IjoieWNhb2tyaXMiLCJhIjoiY2p1MDR5c3JmMzJsbjQ1cGlhNHA3MHFkaCJ9.xb3lXp5JskCYFewsv5uU1w"

# Load data
APP_PATH = str(pathlib.Path(__file__).parent.resolve())

'''
df = pd.read_csv(os.path.join(
    APP_PATH, os.path.join("data", "test_composition.csv")))
df_prod = pd.read_csv(
    os.path.join(APP_PATH, os.path.join(
        "data", "YearlyProduction_table_1.csv"))
)'''
'''
# Assign color to legend
colormap = {}
for ind, formation_name in enumerate(df["fm_name"].unique().tolist()):
    colormap[formation_name] = constants.colors[ind]

'''


def build_banner():
    return html.Div(
        id="banner",
        className="banner",
        children=[
            # html.Img(src=app.get_asset_url("logo.gif")),
            html.H6("LIFELINE"),
        ],
    )


def build_graph_title(title):
    return html.P(className="graph-title", children=title)


'''
def generate_production_plot(processed_data):
    """
    :param processed_data: List containing two lists, one containing well ID information, and the second containing
    rock formation type associated with the well
    :return: Figure object
    """
    layout = dict(
        xaxis=dict(title="Year"), yaxis=dict(title="GAS Production (mcf)", type="log")
    )

    data = []
    for well_id, formation in list(
        zip(processed_data["well_id"], processed_data["formation"])
    ):
        well_prod = df_prod[df_prod["RecordNumber"] == well_id]
        new_trace = dict(
            x=well_prod["Year"],
            y=well_prod["VolumeMCF"],
            name=str(well_id),
            mode="lines+markers",
            hoverinfo="x+y+name",
            marker=dict(
                symbol="hexagram-open", line={"width": "0.5"}, color=colormap[formation]
            ),
            line=dict(shape="spline"),
            showlegend=True,
        )
        data.append(new_trace)
    return {"data": data, "layout": layout}
'''
'''
def generate_well_map(dff, selected_data, style):
    """
    Generate well map based on selected data.
    :param dff: dataframe for generate plot.
    :param selected_data: Processed dictionary for plot generation with defined selected points.
    :param style: mapbox visual style.
    :return: Plotly figure object.
    """

    layout = go.Layout(
        clickmode="event+select",
        dragmode="lasso",
        showlegend=True,
        autosize=True,
        hovermode="closest",
        margin=dict(l=0, r=0, t=0, b=0),
        mapbox=go.layout.Mapbox(
            accesstoken=mapbox_access_token,
            bearing=0,
            center=go.layout.mapbox.Center(lat=37.497562, lon=-82.755728),
            pitch=0,
            zoom=8,
            style=style,
        ),
        legend=dict(
            bgcolor="#1f2c56",
            orientation="h",
            font=dict(color="white"),
            x=0,
            y=0,
            yanchor="bottom",
        ),
    )

    formations = dff["fm_name"].unique().tolist()

    data = []

    for formation in formations:
        selected_index = None
        if formation in selected_data:
            selected_index = selected_data[formation]

        text_list = list(
            map(
                lambda item: "Well ID:" + str(int(item)),
                dff[dff["fm_name"] == formation]["RecordNumber"],
            )
        )
        op_list = dff[dff["fm_name"] == formation]["op"].tolist()

        text_list = [op_list[i] + "<br>" + text_list[i]
                     for i in range(len(text_list))]

        new_trace = go.Scattermapbox(
            lat=dff[dff["fm_name"] == formation]["nlat83"],
            lon=dff[dff["fm_name"] == formation]["wlon83"],
            mode="markers",
            marker={"color": colormap[formation], "size": 9},
            text=text_list,
            name=formation,
            selectedpoints=selected_index,
            customdata=dff[dff["fm_name"] == formation]["RecordNumber"],
        )
        data.append(new_trace)

    return {"data": data, "layout": layout}

'''
'''
def generate_ternary_map(dff, selected_data, contour_visible, marker_visible):
    """
    Generate ternary plot based on selected data.
    :param dff: dataframe for generate plot.
    :param selected_data: Processed dictionary for plot generation with defined selected points.
    :param contour_visible: Contour trace visibility.
    :param marker_visible: Marker trace visibility.
    :return: ternary map figure object.
    """

    # Generate contour

    contour_traces = []
    for ind, key in enumerate(constants.ternary_contour.keys()):
        trace = dict(
            name=key,
            type="scatterternary",
            a=[k["Quartz"] for k in constants.ternary_contour[key]],
            b=[k["Carbonate"] for k in constants.ternary_contour[key]],
            c=[k["Clay"] for k in constants.ternary_contour[key]],
            mode="lines",
            line=dict(color="#444", width=0.5),
            fill="toself",
            fillcolor=constants.ternary_color[ind],
            opacity=0.8,
            hoverinfo="none",
            showlegend=False,
            visible=contour_visible,
        )
        contour_traces.append(trace)

    contour_text = generate_contour_text_layer(contour_visible)

    # Layout
    layout = {
        "dragmode": "lasso",
        "ternary": {
            "sum": 100,
            "aaxis": {
                "title": {
                    "text": "Quartz",
                    "font": {"family": "Open Sans", "size": 15, "color": "white"},
                },
                "min": -2,
                "linewidth": 1.5,
                "ticks": "outside",
            },
            "baxis": {
                "title": {
                    "text": "Carbonate",
                    "font": {"family": "Open Sans", "size": 15, "color": "white"},
                },
                "min": -2,
                "linewidth": 1.5,
                "ticks": "outside",
            },
            "caxis": {
                "title": {
                    "text": "Clay",
                    "font": {"family": "Open Sans", "size": 15, "color": "white"},
                },
                "min": -2,
                "linewidth": 1.5,
                "ticks": "outside",
            },
        },
        "margin": dict(l=110, r=50, t=50, b=50),
        "paper_bgcolor": "#192444",
        "plot_bgcolor": "#192444",
        "showLegend": False,
        "font": {"color": "white"},
        "annotations": {"visible": False},
        "autosize": True,
    }

    hovertemplate = "<b> %{text}</b><br><br> Quartz: %{a:.0f}<br>Carbonate : %{b:.0f}<br> Clay: %{c:.0f}<extra></extra>"

    formations = dff["fm_name"].unique().tolist()

    data_traces = []
    for key in formations:
        if selected_data:
            select_indices = selected_data[key]
        else:
            select_indices = None

        new_data_trace = dict(
            text=list(
                map(
                    lambda item: "Well ID:" + str(int(item)),
                    dff[dff["fm_name"] == key]["RecordNumber"],
                )
            ),
            name=key,
            customdata=dff[dff["fm_name"] == key]["RecordNumber"],
            type="scatterternary",
            a=dff[dff["fm_name"] == key]["Quartz"],
            b=dff[dff["fm_name"] == key]["Carbonate"],
            c=dff[dff["fm_name"] == key]["Clay"],
            mode="markers",
            hovertemplate=hovertemplate,
            showlegend=False,
            marker={
                "color": colormap[key],
                "size": 8,
                "line": {"color": "#000000", "width": 0.2},
            },
            selectedpoints=select_indices,
            visible=marker_visible,
        )
        data_traces.append(new_data_trace)

    return {"data": contour_traces + contour_text + data_traces, "layout": layout}

'''


def generate_contour_text_layer(contour_visible):
    layer = []
    for key, value in constants.ternary_contour.items():
        a = np.mean([i["Quartz"] for i in value])
        b = np.mean([i["Carbonate"] for i in value])
        c = np.mean([i["Clay"] for i in value])

        key_br = key.replace(" ", "<br>")

        new_trace = generate_contour_text(
            a, b, c, key, key_br, contour_visible)
        layer.append(new_trace)

    return layer


def generate_contour_text(a, b, c, name, text, visible):
    return dict(
        type="scatterternary",
        a=[a],
        b=[b],
        c=[c],
        name=name,
        text=text,
        mode="text",
        hoverinfo="none",
        textposition="middle center",
        textfont={"size": 11, "color": "#000000", "family": "sans-serif"},
        showlegend=False,
        legendgroup="Rock type",
        visible=visible,
    )


def generate_formation_bar(dff, selected_data):
    """
    Generate bar plot based on selected data.
        :param dff: dataframe for generate plot.
        :param selected_data: Processed dictionary for plot generation with defined selected points.
        :return: ternary map figure object.
    """

    layout = go.Layout(
        showlegend=False,
        hovermode="closest",
        xaxis=dict(tickangle=-45, title="Formations"),
        yaxis=dict(title="Well Counts"),
        clickmode="event+select",
    )

    formations = dff["fm_name"].unique().tolist()

    if selected_data:
        data = []
        for i in formations:
            selected_points = []
            select_indices = selected_data[i]
            if select_indices is not None and len(select_indices) > 0:
                selected_points = [0]
            new_trace = go.Bar(
                x=[i],
                y=[len(dff[dff["fm_name"] == i])],
                name=i,
                hoverinfo="x+y",
                marker={"color": colormap[i]},
                selectedpoints=selected_points,
            )
            data.append(new_trace)

    else:
        data = []
        for i in formations:
            new_trace = go.Bar(
                x=[i],
                y=[len(dff[dff["fm_name"] == i])],
                name=i,
                marker={"color": colormap[i]},
                selectedpoints=None,
            )
            data.append(new_trace)

    return {"data": data, "layout": layout}


# Helper for extracting select index from mapbox and tern selectData
def get_selection(data, formation, selection_data, starting_index):
    ind = []
    current_curve = data["fm_name"].unique().tolist().index(formation)
    for point in selection_data["points"]:
        if point["curveNumber"] - starting_index == current_curve:
            ind.append(point["pointNumber"])
    return ind


# Helper for extracting select index from bar
def get_selection_by_bar(bar_selected_data):
    dict = {}
    if bar_selected_data is not None:
        for point in bar_selected_data["points"]:
            if point["x"] is not None:
                dict[(point["x"])] = list(range(0, point["y"]))
    return dict


app.layout = html.Div(
    children=[
        html.Div(
            id="top-row",
            style={'align': 'center'},
            children=[
                html.Div(
                    className="row",
                    id="top-row-header",
                    children=[
                        html.Div(
                            id="header-container",
                            children=[
                                build_banner(),
                                html.P(
                                    id="instructions",
                                    children="The vision? Doctors are replaced with wearables(future healthcare) and Robo Advisors - (Insert number here about the potential size of the market). Up to date & accurate health information about yourself no matter where you go that helps to make you better decisions to improve your health + mental health. Our mission? To make health information accessible and actionable for everyone(with a wearables). Put you in control of your health.",
                                ),
                                html.H4(style={'color': '#ff4136'},
                                        children="So, where do you stand?"
                                        )
                            ],
                        )
                    ],
                ),
                html.Div(
                    style={'paddingLeft': '30px'},
                    className="row",
                    id="top-row-graphs",
                    children=[
                        # Well map
                        html.Div(
                            id="well-map-container",
                            children=[
                                build_graph_title("Your Chronological Age"),

                                html.Div(style={'color': '#ff4136', 'fontSize': '60px', 'paddingLeft': '100px'}, children=[
                                    '23'

                                ])
                            ],
                        ),
                        # Ternary map
                        html.Div(
                            id="ternary-map-container",
                            style={'paddingLeft': '100px'},
                            children=[
                                build_graph_title("Your Body's Physical Age"),

                                html.Div(style={'color': '#ff4136', 'fontSize': '60px', 'paddingLeft': '100px'}, children=[
                                    '30'

                                ])
                            ],
                        ),
                    ],
                ),
            ],
        ),
        html.Div(
            className="row",
            id="bottom-row",
            children=[


            ],
        ),
    ]
)


# Running the server
if __name__ == "__main__":
    app.run_server(debug=True)
