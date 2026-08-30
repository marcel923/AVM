import pandas as pd
import numpy as np

class DataLoader:
    def __init__(self, filepath: str):
        self.filepath = filepath

    def load_warsaw_data(self) -> pd.DataFrame:
        df = pd.read_csv(self.filepath)
        # Filtrujemy dane wyłącznie dla Warszawy
        df_warsaw = df[df['city'].str.lower() == 'warszawa'].copy()
        
        # Uzupełniamy brakujące wartości numeryczne medianą
        num_cols = ['squareMeters', 'rooms', 'floor', 'floorCount', 'buildYear', 
                    'centreDistance', 'schoolDistance', 'clinicDistance', 'postOfficeDistance', 
                    'kindergartenDistance', 'restaurantDistance', 'pharmacyDistance']
        
        for col in num_cols:
            if col in df_warsaw.columns:
                df_warsaw[col] = df_warsaw[col].fillna(df_warsaw[col].median())

        return df_warsaw