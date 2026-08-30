import pandas as pd
import numpy as np

# Współrzędne centrum Warszawy (PKiN)
WARSAW_CENTER_LAT = 52.2297
WARSAW_CENTER_LON = 21.0122

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2)**2
    return R * (2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a)))

class FeatureEngineer:
    def __init__(self):
        self.binary_cols = ['hasParkingSpace', 'hasBalcony', 'hasElevator', 'hasSecurity', 'hasStorageRoom']
        
    def transform_features(self, df: pd.DataFrame) -> pd.DataFrame:
        data = df.copy()
        
        # 1. Obliczenie odległości od centrum (PKiN) na podstawie lat/lon, jeśli brak
        if 'latitude' in data.columns and 'longitude' in data.columns:
            data['centreDistance_calc'] = haversine(data['latitude'], data['longitude'], WARSAW_CENTER_LAT, WARSAW_CENTER_LON)
            data['centreDistance'] = data['centreDistance'].fillna(data['centreDistance_calc'])
            data.drop(columns=['centreDistance_calc'], inplace=True, errors='ignore')

        # 2. Binary Flags (1/0)
        for col in self.binary_cols:
            if col in data.columns:
                data[col] = data[col].apply(lambda x: 1 if str(x).strip().lower() in ['yes', '1', 'true'] else 0)

        # 3. Infrastruktura społeczna (Szkoły, Przedszkola, Przychodnie)
        poi_cols = ['schoolDistance', 'kindergartenDistance', 'clinicDistance', 'pharmacyDistance']
        for c in poi_cols:
            if c in data.columns:
                data[c] = data[c].fillna(data[c].median())

        # Prosta uśredniona metryka bliskości usług
        available_poi = [c for c in poi_cols if c in data.columns]
        if available_poi:
            data['avg_infra_distance'] = data[available_poi].mean(axis=1)

        # 4. Wskaźnik piętra
        data['floor_ratio'] = np.where(data['floorCount'] > 0, data['floor'] / data['floorCount'], 0.5)

        return data