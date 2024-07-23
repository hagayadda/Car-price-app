import pandas as pd

def prepare_data(df, is_training=True):
    df = df.copy()
    
    # נוודא שהשדות ממירים למספרים
    df['Engine_type'] = df['Engine_type'].apply(lambda x: 1 if x == 'בנזין' else 0)
    df['Prev_ownership'] = df['Prev_ownership'].apply(lambda x: 1 if x == 'פרטית' else 0)
    df['Curr_ownership'] = df['Curr_ownership'].apply(lambda x: 1 if x == 'פרטית' else 0)
    df['Hand'] = pd.to_numeric(df['Hand'], errors='coerce')  # המרת העמודה החדשה למספרים
    
    # המרה של משתנים קטגוריים לדמיונים (Dummy variables)
    df = pd.get_dummies(df, columns=['manufactor', 'model', 'Gear'], drop_first=True)
    
    # המרת עמודות למספרים
    df['Km'] = pd.to_numeric(df['Km'], errors='coerce')
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    df['capacity_Engine'] = pd.to_numeric(df['capacity_Engine'], errors='coerce')

    # שמירת רק העמודות הרלוונטיות
    relevant_columns = ['Year', 'Km', 'capacity_Engine', 'Engine_type', 'Prev_ownership', 'Curr_ownership', 'Hand'] + [col for col in df.columns if 'manufactor_' in col or 'model_' in col or 'Gear_' in col]
    
    if is_training:
        df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
        df = df[relevant_columns + ['Price']].dropna()
    else:
        df = df[relevant_columns].dropna()

    print("Columns after preparation:", df.columns)  # הדפסה של העמודות אחרי העיבוד
    print("DataFrame shape:", df.shape)  # הדפסה של צורת DataFrame

    return df
