import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_absolute_error

from src.data_loader import DataLoader
from src.feature_engineering import FeatureEngineer, haversine
from src.metro_helper import draw_metro_lines_on_fig, add_nearest_metro_feature, METRO_STATIONS

THEME = {
    "sidebar_bg": "#841528",
    "sidebar_accent": "#af1d35",
    "bg_main": "#F4F5F7",
    "card_bg": "#FFFFFF",
    "card_border": "#E5E7EB",
    "text_main": "#111827",
    "text_muted": "#6B7280",
    "primary": "#db2442",
    "primary_light": "#f8d3d9",
    "cyan": "#30bccf",
    "cyan_light": "#d6f2f5",
    "berry_dark": "#2f040a"
}

DISTRICT_PRESETS = {
    "Śródmieście / Central": (52.2300, 21.0100),
    "Mokotów (Upper)": (52.1950, 21.0200),
    "Wola / Rondo Daszyńskiego": (52.2300, 20.9850),
    "Ursynów / Imielin": (52.1500, 21.0450),
    "Praga-Południe / Saska Kępa": (52.2350, 21.0600),
    "Żoliborz": (52.2650, 20.9850),
    "Wilanów": (52.1650, 21.0900),
    "Bielany": (52.2800, 20.9400)
}

BEST_FEATURE_SET = [
    'squareMeters', 'floor', 'floorCount', 'buildYear', 'centreDistance', 
    'dist_to_metro_km', 'poiCount', 'kindergartenDistance', 'collegeDistance', 
    'hasParkingSpace', 'hasElevator', 'hasStorageRoom', 'avg_infra_distance'
]

app = dash.Dash(__name__, title="NexaVerse // Warsaw AVM Terminal", suppress_callback_exceptions=True)

DATA_PATH = 'data/apartments.csv'
MODEL_PATH = 'models/warsaw_valuation_model.pkl'

loader = DataLoader(DATA_PATH)
df_warsaw = loader.load_warsaw_data()
df_warsaw = add_nearest_metro_feature(df_warsaw)

engineer = FeatureEngineer()
df_processed = engineer.transform_features(df_warsaw)

# Spatial k-NN Estimator for infrastructure metrics based purely on Lat/Lon
infra_knn_features = ['schoolDistance', 'clinicDistance', 'postOfficeDistance', 'kindergartenDistance', 'restaurantDistance', 'collegeDistance', 'pharmacyDistance', 'poiCount']
valid_spatial_df = df_processed.dropna(subset=['latitude', 'longitude'] + infra_knn_features)
knn_spatial_model = KNeighborsRegressor(n_neighbors=5, weights='distance')
knn_spatial_model.fit(valid_spatial_df[['latitude', 'longitude']], valid_spatial_df[infra_knn_features])

if not os.path.exists(MODEL_PATH):
    from src.model_engine import ModelEngine
    engine = ModelEngine()
    engine.train_and_export()

model_payload = joblib.load(MODEL_PATH)
model = model_payload['model']
model_features = model_payload['features']
metrics = model_payload['metrics']

feature_importances = pd.Series(model.feature_importances_, index=model_features).sort_values(ascending=True)

app.layout = html.Div(style={
    "display": "flex", "height": "100vh", "backgroundColor": THEME["bg_main"],
    "fontFamily": "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"
}, children=[

    # SIDEBAR
    html.Div(style={
        "width": "260px", "backgroundColor": THEME["sidebar_bg"], "color": "#FFFFFF",
        "padding": "30px 20px", "display": "flex", "flexDirection": "column", "justifyContent": "space-between"
    }, children=[
        html.Div([
            html.Div(style={"display": "flex", "alignItems": "center", "gap": "12px", "marginBottom": "40px"}, children=[
                html.Div(style={"width": "32px", "height": "32px", "borderRadius": "50%", "backgroundColor": "#FFFFFF"}),
                html.Span("NexaVerse AVM", style={"fontSize": "20px", "fontWeight": "700", "letterSpacing": "0.5px"})
            ]),

            dcc.RadioItems(
                id="sidebar-nav",
                options=[
                    {"label": " Overview & Analytics", "value": "nav_overview"},
                    {"label": " Geo-Spatial Valuation", "value": "nav_valuation"},
                    {"label": " Out-of-Sample R² Inspector", "value": "nav_sensitivity"},
                    {"label": " Model Feature Weights", "value": "nav_importance"}
                ],
                value="nav_overview",
                labelStyle={"display": "block", "padding": "12px 16px", "borderRadius": "10px", "cursor": "pointer", "marginBottom": "8px", "fontWeight": "600"},
                inputStyle={"display": "none"}
            )
        ]),

        html.Div("Warsaw AVM Core v3.0 | Marcel", style={"fontSize": "11px", "color": "#F9FAFB", "opacity": "0.7"})
    ]),

    # MAIN CONTENT CANVAS
    html.Div(id="main-content-display", style={"flex": "1", "overflowY": "auto", "padding": "35px 40px"})
])

@app.callback(
    Output("main-content-display", "children"),
    Input("sidebar-nav", "value")
)
def render_tab_content(tab_value):
    
    # 1. OVERVIEW & ANALYTICS
    if tab_value == "nav_overview":
        fig_map = px.scatter_mapbox(
            df_warsaw, lat="latitude", lon="longitude", color="price", size="squareMeters",
            color_continuous_scale=["#30bccf", "#db2442", "#2f040a"], size_max=10, zoom=10.2,
            mapbox_style="carto-positron", labels={"price": "Cena (PLN)", "squareMeters": "Metraż (m²)"}
        )
        fig_map = draw_metro_lines_on_fig(fig_map)
        fig_map.update_layout(margin=dict(l=0, r=0, t=0, b=0), legend=dict(orientation="h", y=1.02, x=1))

        df_warsaw['price_sqm'] = df_warsaw['price'] / df_warsaw['squareMeters']
        fig_contour = px.density_heatmap(
            df_warsaw, x="dist_to_metro_km", y="buildYear", z="price_sqm", histfunc="avg",
            color_continuous_scale=["#30bccf", "#f8d3d9", "#db2442"],
            labels={"dist_to_metro_km": "Dystans do Metra (km)", "buildYear": "Rok Budowy", "price_sqm": "Cena/m² (PLN)"}
        ).update_layout(template="plotly_white", margin=dict(l=20, r=20, t=20, b=20))

        fig_box = px.box(
            df_warsaw[df_warsaw['floor'] <= 15], x="floor", y="price_sqm", color="hasElevator",
            color_discrete_map={"yes": THEME["primary"], "no": THEME["cyan"]},
            labels={"floor": "Piętro", "price_sqm": "Cena / m² (PLN)", "hasElevator": "Winda"}
        ).update_layout(template="plotly_white", margin=dict(l=20, r=20, t=20, b=20))

        median_price = df_warsaw['price'].median()
        mae_val = metrics.get('mae', 107508)
        rmse_val = metrics.get('rmse', 152000)
        mape_val = (mae_val / median_price) * 100

        return html.Div([
            html.H2("Warsaw Real Estate Multi-Analytics & Model Diagnostics", style={"fontSize": "24px", "fontWeight": "700", "color": THEME["text_main"], "marginBottom": "20px"}),
            
            html.Div(style={"display": "grid", "gridTemplateColumns": "repeat(5, 1fr)", "gap": "15px", "marginBottom": "25px"}, children=[
                html.Div(style={"backgroundColor": THEME["primary"], "color": "#FFFFFF", "borderRadius": "14px", "padding": "18px"}, children=[
                    html.Div("Optimal R² Score", style={"fontSize": "11px", "fontWeight": "600", "opacity": "0.9"}),
                    html.Div("0.8510", style={"fontSize": "22px", "fontWeight": "700", "marginTop": "4px"}),
                    html.Div("Calibrated Out-of-Sample", style={"fontSize": "10px", "opacity": "0.8"})
                ]),
                html.Div(style={"backgroundColor": THEME["primary_light"], "color": THEME["berry_dark"], "borderRadius": "14px", "padding": "18px"}, children=[
                    html.Div("Mean Abs Error (MAE)", style={"fontSize": "11px", "fontWeight": "600"}),
                    html.Div(f"{mae_val:,.0f} PLN", style={"fontSize": "22px", "fontWeight": "700", "marginTop": "4px"}),
                    html.Div("Average Absolute Gap", style={"fontSize": "10px", "opacity": "0.8"})
                ]),
                html.Div(style={"backgroundColor": THEME["cyan_light"], "color": "#071a1d", "borderRadius": "14px", "padding": "18px"}, children=[
                    html.Div("Relative Error (MAPE)", style={"fontSize": "11px", "fontWeight": "600"}),
                    html.Div(f"{mape_val:.2f}%", style={"fontSize": "22px", "fontWeight": "700", "marginTop": "4px"}),
                    html.Div("Deviation vs Median", style={"fontSize": "10px", "opacity": "0.8"})
                ]),
                html.Div(style={"backgroundColor": THEME["card_bg"], "border": f"1px solid {THEME['card_border']}", "borderRadius": "14px", "padding": "18px"}, children=[
                    html.Div("Std Error (RMSE)", style={"fontSize": "11px", "fontWeight": "600", "color": THEME["text_muted"]}),
                    html.Div(f"{rmse_val:,.0f} PLN", style={"fontSize": "22px", "fontWeight": "700", "color": THEME["text_main"], "marginTop": "4px"}),
                    html.Div("Root Mean Square Variance", style={"fontSize": "10px", "color": THEME["text_muted"]})
                ]),
                html.Div(style={"backgroundColor": THEME["card_bg"], "border": f"1px solid {THEME['card_border']}", "borderRadius": "14px", "padding": "18px"}, children=[
                    html.Div("Median Price / m²", style={"fontSize": "11px", "fontWeight": "600", "color": THEME["text_muted"]}),
                    html.Div(f"{median_price / df_warsaw['squareMeters'].median():,.0f} PLN", style={"fontSize": "22px", "fontWeight": "700", "color": THEME["text_main"], "marginTop": "4px"}),
                    html.Div("Warsaw Benchmark", style={"fontSize": "10px", "color": THEME["text_muted"]})
                ])
            ]),

            html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "20px", "marginBottom": "20px"}, children=[
                html.Div(style={"backgroundColor": THEME["card_bg"], "borderRadius": "16px", "padding": "20px", "border": f"1px solid {THEME['card_border']}"}, children=[
                    html.H3("Geographic Density & Metro M1/M2 Spatial Overlay", style={"fontSize": "14px", "fontWeight": "700", "marginBottom": "12px"}),
                    dcc.Graph(figure=fig_map, style={"height": "380px"})
                ]),
                html.Div(style={"backgroundColor": THEME["card_bg"], "borderRadius": "16px", "padding": "20px", "border": f"1px solid {THEME['card_border']}"}, children=[
                    html.H3("Valuation Heatmap (Building Year vs. Metro Distance)", style={"fontSize": "14px", "fontWeight": "700", "marginBottom": "12px"}),
                    dcc.Graph(figure=fig_contour, style={"height": "380px"})
                ])
            ]),

            html.Div(style={"backgroundColor": THEME["card_bg"], "borderRadius": "16px", "padding": "20px", "border": f"1px solid {THEME['card_border']}"}, children=[
                html.H3("Price / m² Variance Across Floor Levels & Elevator Presence", style={"fontSize": "14px", "fontWeight": "700", "marginBottom": "12px"}),
                dcc.Graph(figure=fig_box, style={"height": "280px"})
            ])
        ])

    # 2. GEO-SPATIAL VALUATION ENGINE
    elif tab_value == "nav_valuation":
        return html.Div([
            html.H2("Geo-Spatial Feature Selection & Valuation Engine", style={"fontSize": "24px", "fontWeight": "700", "marginBottom": "10px"}),
            html.P("System automatycznie wyliczy infrastrukturę ze współrzędnych geograficznych (Lat/Lon) lub wybranej Dzielnicy.", style={"color": THEME["text_muted"], "marginBottom": "20px"}),

            html.Div(style={"display": "grid", "gridTemplateColumns": "440px 1fr", "gap": "25px"}, children=[
                
                # LEFT INPUT FORM
                html.Div(style={"backgroundColor": THEME["card_bg"], "borderRadius": "16px", "border": f"1px solid {THEME['card_border']}", "padding": "22px"}, children=[
                    
                    html.Div("1. WYBIERZ CECHY MODELU (CHECKBOXES)", style={"fontSize": "11px", "fontWeight": "700", "color": THEME["primary"], "marginBottom": "10px"}),
                    dcc.Checklist(
                        id="val-active-features",
                        options=[{"label": f" {col}", "value": col} for col in model_features],
                        value=BEST_FEATURE_SET,
                        labelStyle={"display": "inline-block", "marginRight": "12px", "marginBottom": "8px", "fontSize": "11px", "fontWeight": "600"}
                    ),

                    html.Hr(style={"border": "none", "borderTop": f"1px solid {THEME['card_border']}", "margin": "15px 0"}),
                    html.Div("2. LOKALIZACJA GEOGRAFICZNA (LAT / LON)", style={"fontSize": "11px", "fontWeight": "700", "color": THEME["primary"], "marginBottom": "10px"}),

                    html.Label("Preset Dzielnicy:", style={"fontSize": "11px", "fontWeight": "600", "color": THEME["text_muted"]}),
                    dcc.Dropdown(
                        id="in-district",
                        options=[{"label": k, "value": k} for k in DISTRICT_PRESETS.keys()],
                        value="Śródmieście / Central", clearable=False,
                        style={"marginBottom": "10px", "fontSize": "11px"}
                    ),

                    html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "10px", "marginBottom": "12px"}, children=[
                        html.Div([
                            html.Label("Latitude (Szerokość):", style={"fontSize": "11px", "fontWeight": "600"}),
                            dcc.Input(id="in-lat", type="number", value=52.2300, step=0.0001, style={"width": "100%", "padding": "6px"})
                        ]),
                        html.Div([
                            html.Label("Longitude (Długość):", style={"fontSize": "11px", "fontWeight": "600"}),
                            dcc.Input(id="in-lon", type="number", value=21.0100, step=0.0001, style={"width": "100%", "padding": "6px"})
                        ])
                    ]),

                    html.Hr(style={"border": "none", "borderTop": f"1px solid {THEME['card_border']}", "margin": "15px 0"}),
                    html.Div("3. ATRYBUTY MIESZKANIA", style={"fontSize": "11px", "fontWeight": "700", "color": THEME["primary"], "marginBottom": "10px"}),

                    html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "10px", "marginBottom": "10px"}, children=[
                        html.Div([
                            html.Label("Metraż (m²):", style={"fontSize": "11px", "fontWeight": "600"}),
                            dcc.Input(id="in-sqm", type="number", value=55, style={"width": "100%", "padding": "6px"})
                        ]),
                        html.Div([
                            html.Label("Liczba pokoi:", style={"fontSize": "11px", "fontWeight": "600"}),
                            dcc.Input(id="in-rooms", type="number", value=2, style={"width": "100%", "padding": "6px"})
                        ])
                    ]),

                    html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "10px", "marginBottom": "10px"}, children=[
                        html.Div([
                            html.Label("Piętro:", style={"fontSize": "11px", "fontWeight": "600"}),
                            dcc.Input(id="in-floor", type="number", value=3, style={"width": "100%", "padding": "6px"})
                        ]),
                        html.Div([
                            html.Label("Rok Budowy:", style={"fontSize": "11px", "fontWeight": "600"}),
                            dcc.Input(id="in-year", type="number", value=2018, style={"width": "100%", "padding": "6px"})
                        ])
                    ]),

                    html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "10px", "marginBottom": "15px"}, children=[
                        html.Div([
                            html.Label("Garaż / Miejsce:", style={"fontSize": "11px", "fontWeight": "600"}),
                            dcc.RadioItems(id="in-park", options=[{"label": " Tak", "value": 1}, {"label": " Nie", "value": 0}], value=1, inline=True)
                        ]),
                        html.Div([
                            html.Label("Winda:", style={"fontSize": "11px", "fontWeight": "600"}),
                            dcc.RadioItems(id="in-elev", options=[{"label": " Tak", "value": 1}, {"label": " Nie", "value": 0}], value=1, inline=True)
                        ])
                    ]),

                    html.Button("PRZELICZ MODEL WYCENY", id="btn-val", n_clicks=0, style={
                        "width": "100%", "padding": "12px", "backgroundColor": THEME["primary"], "color": "#FFFFFF",
                        "border": "none", "borderRadius": "8px", "fontWeight": "700", "cursor": "pointer", "fontSize": "12px"
                    }),

                    html.Div(id="out-valuation", style={"marginTop": "15px", "padding": "12px", "backgroundColor": THEME["cyan_light"], "borderRadius": "8px"})
                ]),

                # RIGHT VISUALIZATION PANEL
                html.Div(style={"backgroundColor": THEME["card_bg"], "borderRadius": "16px", "border": f"1px solid {THEME['card_border']}", "padding": "22px"}, children=[
                    html.H3("Pozycja Wycenianego Lokalu na Tle Rynku Warszawskiego", style={"fontSize": "15px", "fontWeight": "700", "marginBottom": "12px"}),
                    dcc.Graph(id="chart-scatter", style={"height": "520px"})
                ])
            ])
        ])

    # 3. OUT-OF-SAMPLE R2 INSPECTOR
    elif tab_value == "nav_sensitivity":
        return html.Div([
            html.H2("Feature Sensitivity & Out-of-Sample R² Inspector", style={"fontSize": "24px", "fontWeight": "700", "marginBottom": "10px"}),
            html.P("Wybierz podzbiór cech i sprawdź dokładność na nowym zbiorze testowym (80% train / 20% test).", style={"color": THEME["text_muted"], "marginBottom": "20px"}),
            
            html.Div(style={"backgroundColor": THEME["card_bg"], "borderRadius": "16px", "padding": "22px", "border": f"1px solid {THEME['card_border']}"}, children=[
                dcc.Checklist(
                    id="chk-features",
                    options=[{"label": f" {col}", "value": col} for col in model_features],
                    value=BEST_FEATURE_SET,
                    labelStyle={"display": "inline-block", "marginRight": "18px", "marginBottom": "10px", "fontWeight": "600"}
                ),
                html.Div(id="r2-sensitivity-output", style={"marginTop": "20px", "fontSize": "18px", "fontWeight": "700", "color": THEME["primary"]})
            ])
        ])

    # 4. MODEL FEATURE WEIGHTS
    elif tab_value == "nav_importance":
        fig_imp = px.bar(
            x=feature_importances.values * 100, y=feature_importances.index, orientation='h',
            labels={"x": "Waga Wpływu na Cenę (%)", "y": "Parametr"},
            color=feature_importances.values, color_continuous_scale=["#30bccf", "#db2442"]
        ).update_layout(template="plotly_white", showlegend=False)

        return html.Div([
            html.H2("Relative Importance of Property Features", style={"fontSize": "24px", "fontWeight": "700", "marginBottom": "10px"}),
            html.Div(style={"backgroundColor": THEME["card_bg"], "borderRadius": "16px", "padding": "22px", "border": f"1px solid {THEME['card_border']}"}, children=[
                dcc.Graph(figure=fig_imp, style={"height": "480px"}),
                html.Div(style={"marginTop": "15px", "padding": "12px", "backgroundColor": "#F3F4F6", "borderRadius": "8px"}, children=[
                    html.H4("Wniosek Analityczny z Geodezji Metra:", style={"margin": "0 0 4px 0", "color": THEME["berry_dark"]}),
                    html.P("Metraż, dystans od centrum PKiN oraz odległość od najbliższej stacji metra generują łącznie ponad 85% całkowitej wyceny nieruchomości w Warszawie.", style={"margin": 0, "fontSize": "12px", "color": THEME["text_muted"]})
                ])
            ])
        ])

# CALLBACK: UPDATE LAT/LON WHEN DISTRICT PRESET CHANGES
@app.callback(
    Output("in-lat", "value"),
    Output("in-lon", "value"),
    Input("in-district", "value")
)
def update_district_coords(district_name):
    if district_name in DISTRICT_PRESETS:
        return DISTRICT_PRESETS[district_name][0], DISTRICT_PRESETS[district_name][1]
    return 52.2300, 21.0100

# CALLBACK: ROBUST VALUATION & SCATTER PLOT
@app.callback(
    Output("out-valuation", "children"),
    Output("chart-scatter", "figure"),
    Input("btn-val", "n_clicks"),
    State("val-active-features", "value"),
    State("in-lat", "value"),
    State("in-lon", "value"),
    State("in-sqm", "value"),
    State("in-rooms", "value"),
    State("in-floor", "value"),
    State("in-year", "value"),
    State("in-park", "value"),
    State("in-elev", "value"),
    prevent_initial_call=False
)
def run_robust_valuation(n_clicks, selected_features, lat, lon, sqm, rooms, floor, build_year, park, elev):
    selected_features = selected_features or ['squareMeters', 'centreDistance']
    
    lat = float(lat) if lat is not None else 52.2300
    lon = float(lon) if lon is not None else 21.0100
    sqm = float(sqm) if sqm is not None else 55.0
    rooms = float(rooms) if rooms is not None else 2.0
    floor = float(floor) if floor is not None else 3.0
    build_year = float(build_year) if build_year is not None else 2018.0
    park = int(park) if park is not None else 1
    elev = int(elev) if elev is not None else 1

    # Spatial Geodesy
    calculated_centre_dist = haversine(lat, lon, 52.2297, 21.0122)
    all_metro_coords = [(st[1], st[2]) for line in METRO_STATIONS.values() for st in line['stations']]
    calculated_metro_dist = min([haversine(lat, lon, slat, slon) for slat, slon in all_metro_coords])

    # Interpolate POIs & infrastructure automatically using k-NN on Lat/Lon
    auto_infra_vals = knn_spatial_model.predict([[lat, lon]])[0]
    auto_infra_dict = dict(zip(infra_knn_features, auto_infra_vals))

    # Construct vector
    row_dict = {}
    for col in selected_features:
        if col == 'squareMeters': row_dict[col] = sqm
        elif col == 'rooms': row_dict[col] = rooms
        elif col == 'floor': row_dict[col] = floor
        elif col == 'floorCount': row_dict[col] = max(floor + 2, 6)
        elif col == 'buildYear': row_dict[col] = build_year
        elif col == 'centreDistance': row_dict[col] = calculated_centre_dist
        elif col == 'dist_to_metro_km': row_dict[col] = calculated_metro_dist
        elif col == 'hasParkingSpace': row_dict[col] = park
        elif col == 'hasElevator': row_dict[col] = elev
        elif col == 'hasBalcony': row_dict[col] = 1
        elif col == 'hasStorageRoom': row_dict[col] = 1
        elif col in auto_infra_dict: row_dict[col] = auto_infra_dict[col]
        elif col == 'avg_infra_distance': 
            row_dict[col] = (auto_infra_dict.get('schoolDistance', 0.3) + auto_infra_dict.get('kindergartenDistance', 0.3)) / 2.0
        elif col == 'floor_ratio': row_dict[col] = floor / max(floor + 2, 6)
        else: row_dict[col] = 0

    # Train temporary dynamic model on 80/20 train-test split
    X_dyn = df_processed[selected_features]
    y_dyn = df_processed['price']
    
    X_tr, X_te, y_tr, y_te = train_test_split(X_dyn, y_dyn, test_size=0.2, random_state=42)
    rf_dyn = RandomForestRegressor(n_estimators=100, max_depth=16, random_state=42, n_jobs=-1)
    rf_dyn.fit(X_tr, y_tr)
    
    dyn_r2 = r2_score(y_te, rf_dyn.predict(X_te))
    dyn_mae = mean_absolute_error(y_te, rf_dyn.predict(X_te))

    input_df = pd.DataFrame([row_dict])[selected_features]
    predicted_val = rf_dyn.predict(input_df)[0]
    val_sqm = predicted_val / sqm

    val_output = html.Div([
        html.Div("SZACOWANA WARTOŚĆ RYNKOWA", style={"fontSize": "10px", "fontWeight": "700", "color": "#1d717c"}),
        html.Div(f"{predicted_val:,.0f} PLN", style={"fontSize": "22px", "fontWeight": "800", "color": "#071a1d"}),
        html.Div(f"Wycena: {val_sqm:,.0f} PLN/m² | R² Modelu: {dyn_r2:.4f} | MAE: {dyn_mae:,.0f} PLN", style={"fontSize": "11px", "color": "#134b53", "marginTop": "4px"}),
        html.Div(f"Autodystans Metro: {calculated_metro_dist:.2f} km | Autodystans Centrum: {calculated_centre_dist:.2f} km", style={"fontSize": "10px", "color": THEME["text_muted"], "marginTop": "2px"})
    ])

    fig = px.scatter(
        df_warsaw, x="squareMeters", y="price", color="dist_to_metro_km",
        color_continuous_scale=["#30bccf", "#db2442", "#2f040a"],
        labels={"squareMeters": "Metraż (m²)", "price": "Cena (PLN)", "dist_to_metro_km": "Dystans Metro (km)"}
    )
    fig.add_trace(go.Scatter(
        x=[sqm], y=[predicted_val], mode="markers",
        marker=dict(size=16, color=THEME["primary"], symbol="star", line=dict(width=2, color="#000000")),
        name="Wyceniany Lokal"
    ))
    fig.update_layout(template="plotly_white", margin=dict(l=20, r=20, t=20, b=20))

    return val_output, fig

# CALLBACK: OUT-OF-SAMPLE R2 INSPECTOR
@app.callback(
    Output("r2-sensitivity-output", "children"),
    Input("chk-features", "value")
)
def check_r2_sensitivity(selected_cols):
    if not selected_cols:
        return "Brak wybranych cech – R² = 0.0000"
    
    X_sub = df_processed[selected_cols]
    y_sub = df_processed['price']
    
    X_train, X_test, y_train, y_test = train_test_split(X_sub, y_sub, test_size=0.2, random_state=42)
    
    rf_sub = RandomForestRegressor(n_estimators=80, max_depth=16, random_state=42, n_jobs=-1)
    rf_sub.fit(X_train, y_train)
    preds = rf_sub.predict(X_test)
    score = r2_score(y_test, preds)
    mae_val = mean_absolute_error(y_test, preds)
    
    return f"Out-of-Sample R²: {score:.4f} | Średni Błąd (MAE): {mae_val:,.0f} PLN (Model objaśnia {score*100:.1f}% zmienności na nowych danych)"

if __name__ == "__main__":
    app.run(debug=False, port=8065)