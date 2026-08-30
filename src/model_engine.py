import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib
import os

from src.data_loader import DataLoader
from src.feature_engineering import FeatureEngineer
from src.metro_helper import add_nearest_metro_feature

class ModelEngine:
    def __init__(self, data_path: str = 'data/apartments.csv', model_dir: str = 'models'):
        self.data_path = data_path
        self.model_dir = model_dir
        self.model_path = os.path.join(model_dir, 'warsaw_valuation_model.pkl')
        self.model = None

    def train_and_export(self):
        loader = DataLoader(self.data_path)
        df_raw = loader.load_warsaw_data()
        
        # Dodajemy odległość od metra M1/M2
        df_raw = add_nearest_metro_feature(df_raw)

        engineer = FeatureEngineer()
        df_processed = engineer.transform_features(df_raw)

        # Odrzucamy wszystkie nierozpoznawalne ciągi tekstowe i zmienne niepotrzebne
        text_cols = ['id', 'city', 'type', 'ownership', 'buildingMaterial', 'condition', 'latitude', 'longitude']
        drop_cols = text_cols + ['price']
        
        features = [col for col in df_processed.columns if col not in drop_cols and pd.api.types.is_numeric_dtype(df_processed[col])]
        
        X = df_processed[features]
        y = df_processed['price']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.model = RandomForestRegressor(n_estimators=180, max_depth=18, random_state=42, n_jobs=-1)
        self.model.fit(X_train, y_train)

        preds = self.model.predict(X_test)
        r2 = r2_score(y_test, preds)
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))

        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        payload = {
            'model': self.model,
            'features': features,
            'metrics': {'r2': r2, 'mae': mae, 'rmse': rmse},
            'sample_size': len(df_raw)
        }
        joblib.dump(payload, self.model_path)
        print(f"Model Training Complete. R2: {r2:.4f} | MAE: {mae:,.2f} PLN | Exported to {self.model_path}")
        return payload

if __name__ == "__main__":
    engine = ModelEngine()
    engine.train_and_export()