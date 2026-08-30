import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Precyzyjne punkty stacji Metra Warszawskiego (M1 & M2)
METRO_STATIONS = {
    'M1': {
        'color': '#E31E24',  # Czerwona M1
        'stations': [
            ("Kabaty", 52.1311, 21.0653), ("Natolin", 52.1397, 21.0583), ("Imielin", 52.1494, 21.0456),
            ("Stokłosy", 52.1558, 21.0353), ("Ursynów", 52.1622, 21.0278), ("Służew", 52.1736, 21.0256),
            ("Wilanowska", 52.1808, 21.0236), ("Wierzbno", 52.1897, 21.0186), ("Racławicka", 52.1994, 21.0122),
            ("Pole Mokotowskie", 52.2092, 21.0083), ("Politechnika", 52.2197, 21.0153), ("Centrum", 52.2308, 21.0089),
            ("Świętokrzyska", 52.2353, 21.0083), ("Ratusz Arsenał", 52.2450, 21.0006), ("Dworzec Gdański", 52.2581, 20.9986),
            ("Plac Lelewela/Merymont", 52.2686, 20.9844), ("Słodowiec", 52.2775, 20.9706), ("Stare Bielany", 52.2828, 20.9572),
            ("Wawrzyszew", 52.2872, 20.9392), ("Młociny", 52.2908, 20.9292)
        ]
    },
    'M2': {
        'color': '#0072BC',  # Niebieska M2
        'stations': [
            ("Bemowo", 52.2389, 20.9083), ("Ulrychów", 52.2394, 20.9256), ("Księcia Janusza", 52.2397, 20.9428),
            ("Młynów", 52.2383, 20.9608), ("Płocka", 52.2339, 20.9678), ("Rondo Daszyńskiego", 52.2300, 20.9842),
            ("Rondo ONZ", 52.2325, 20.9986), ("Świętokrzyska", 52.2353, 21.0083), ("Nowy Świat-Uniwersytet", 52.2381, 21.0183),
            ("Stadion Narodowy", 52.2478, 21.0369), ("Dworzec Wileński", 52.2544, 21.0347), ("Szwedzka", 52.2619, 21.0358),
            ("Targówek Mieszkaniowy", 52.2694, 21.0508), ("Trocka", 52.2758, 21.0583), ("Zacisze", 52.2850, 21.0667),
            ("Kondratowicza", 52.2903, 21.0475), ("Bródno", 52.2958, 21.0342)
        ]
    }
}

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371.0 # Promień ziemi w km
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2)**2
    return R * (2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a)))

def add_nearest_metro_feature(df):
    """Oblicza najkrótszy dystans (w km) od danego lokalu do najbliższej stacji metra M1/M2"""
    all_coords = [(st[1], st[2]) for line in METRO_STATIONS.values() for st in line['stations']]
    
    def get_min_dist(row):
        lat, lon = row['latitude'], row['longitude']
        if pd.isna(lat) or pd.isna(lon):
            return np.nan
        return min([haversine_distance(lat, lon, slat, slon) for slat, slon in all_coords])

    df['dist_to_metro_km'] = df.apply(get_min_dist, axis=1)
    return df

def draw_metro_lines_on_fig(fig):
    """Rysuje linie i punkty stacji M1/M2 na istniejącym wykresie Mapbox"""
    for line_code, data in METRO_STATIONS.items():
        lats = [st[1] for st in data['stations']]
        lons = [st[2] for st in data['stations']]
        names = [f"Stacja {st[0]} ({line_code})" for st in data['stations']]
        
        # Rysujemy linię metra
        fig.add_trace(go.Scattermapbox(
            lat=lats, lon=lons, mode='lines+markers',
            marker=dict(size=8, color=data['color']),
            line=dict(width=4, color=data['color']),
            text=names, hoverinfo='text',
            name=f"Linia Metra {line_code}"
        ))
    return fig