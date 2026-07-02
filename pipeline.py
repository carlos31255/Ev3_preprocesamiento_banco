import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

def load_and_split_data(filepath='bank-additional-full.csv'):
    """Paso 1: Cargar datos y realizar la división Train/Test inmediatamente."""
    df = pd.read_csv(filepath, sep=';')
    df = df.drop_duplicates()

    if 'duration' in df.columns:
        df = df.drop(columns=['duration'])

    # Separar variable objetivo (y) de las features (X)
    X = df.drop(columns=['y'])
    y = df['y']

    # Division estratificada 80/20
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    # Mapear el target a valores binarios (1/0)
    y_train = y_train.map({'yes': 1, 'no': 0})
    y_test = y_test.map({'yes': 1, 'no': 0})
    
    return X_train, X_test, y_train, y_test

def apply_static_formatting(df):
    """Aplica transformaciones de texto que no dependen de la distribución de los datos."""
    rename_columns = {
        'age':'edad','job':'trabajo','marital':'estado_civil',
        'education':'educacion','default':'mora','housing':'vivienda','loan':'prestamo',
        'contact':'contacto','month':'mes','day_of_week':'dia_semana','campaign':'campana',
        'pdays':'dias_previos','previous':'anterior','poutcome':'resultado_anterior',
        'emp.var.rate':'var_empleo','cons.price.idx':'indice_precios',
        'cons.conf.idx':'indice_confianza','euribor3m':'tasa_euribor','nr.employed':'num_empleados'
    }
    df = df.rename(columns=rename_columns)

    df['trabajo'] = df['trabajo'].replace({
        'admin.':'administrativo','blue-collar':'obrero','entrepreneur':'empresario',
        'housemaid':'empleada_hogar','management':'gerencia','retired':'jubilado',
        'self-employed':'independiente','services':'servicios','student':'estudiante',
        'technician':'tecnico','unemployed':'desempleado','unknown':'desconocido'
    })
    df['estado_civil'] = df['estado_civil'].replace({
        'married':'casado','single':'soltero','divorced':'divorciado','unknown':'desconocido'})
    df['educacion'] = df['educacion'].replace({
        'basic.4y':'basica_4','basic.6y':'basica_6','basic.9y':'basica_9',
        'high.school':'media','illiterate':'analfabeto',
        'professional.course':'tecnico_prof','university.degree':'universitario','unknown':'desconocido'
    })
    df['mora'] = df['mora'].replace({'yes':'si','no':'no','unknown':'desconocido'})
    df['vivienda'] = df['vivienda'].replace({'yes':'si','no':'no','unknown':'desconocido'})
    df['prestamo'] = df['prestamo'].replace({'yes':'si','no':'no','unknown':'desconocido'})
    df['contacto'] = df['contacto'].replace({'cellular':'celular','telephone':'telefono_fijo'})
    df['resultado_anterior'] = df['resultado_anterior'].replace({
        'failure':'fracaso','success':'exito','nonexistent':'sin_contacto'})
    
    return df

def run_pipeline(X_train_raw, X_test_raw):
    """
    Ejecuta el pipeline de preprocesamiento, aprendiendo parámetros SOLO de X_train
    y aplicándolos a X_train y X_test para evitar data leakage.
    """
    X_train = apply_static_formatting(X_train_raw.copy())
    X_test  = apply_static_formatting(X_test_raw.copy())

    # 1. Imputacion aprendiendo modas SOLO del Train
    cols_unknown = ['trabajo','estado_civil','educacion','mora','vivienda','prestamo']
    for col in cols_unknown:
        # Aprender moda en train ignorando los 'desconocido'
        moda = X_train[X_train[col] != 'desconocido'][col].mode()[0]
        # Aplicar la moda de train a ambos conjuntos
        X_train[col] = X_train[col].replace('desconocido', moda)
        X_test[col]  = X_test[col].replace('desconocido', moda)

    # 2. Capping de outliers aprendiendo limites (IQR) SOLO del Train
    for col in ['edad', 'campana']:
        Q1 = X_train[col].quantile(0.25)
        Q3 = X_train[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        X_train[col] = X_train[col].clip(lower=lower_bound, upper=upper_bound)
        X_test[col]  = X_test[col].clip(lower=lower_bound, upper=upper_bound)

    # 3. Creacion de Nuevas Variables
    for df in [X_train, X_test]:
        bins = [0, 30, 60, 100]
        labels = ['Joven', 'Adulto', 'Mayor']
        df['grupo_etario'] = pd.cut(df['edad'], bins=bins, labels=labels)
        df['fue_contactado_antes'] = (df['dias_previos'] != 999).astype(int)
        df['euribor_alto'] = (df['tasa_euribor'] > 3.0).astype(int)

        # Ordinal Encoding
        orden_edu_map = {
            'analfabeto': 1, 'basica_4': 2, 'basica_6': 3, 
            'basica_9': 4, 'media': 5, 'tecnico_prof': 6, 'universitario': 7
        }
        df['educacion_ordinal'] = df['educacion'].map(orden_edu_map)
        df.drop('educacion', axis=1, inplace=True)

    # 4. One-Hot Encoding alineado
    cat_cols_ml = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Get dummies de cada dataset
    X_train = pd.get_dummies(X_train, columns=cat_cols_ml, drop_first=True)
    X_test  = pd.get_dummies(X_test, columns=cat_cols_ml, drop_first=True)
    
    # Asegurar que el test tenga las mismas columnas que el train
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
    
    # Convertir booleanos a enteros
    for df in [X_train, X_test]:
        bool_cols = df.select_dtypes(include=['bool']).columns
        df[bool_cols] = df[bool_cols].astype(int)

    # 5. Escalamiento (Fit en Train, Transform en Train y Test)
    num_cols_ml = ['edad','campana','dias_previos','anterior',
                   'var_empleo','indice_precios','indice_confianza','tasa_euribor','num_empleados']

    scaler = StandardScaler()
    X_train[num_cols_ml] = scaler.fit_transform(X_train[num_cols_ml])
    X_test[num_cols_ml]  = scaler.transform(X_test[num_cols_ml])

    return X_train, X_test

def save_splits(X_train, X_test, y_train, y_test, output_dir='data'):
    """Guarda los conjuntos de datos en archivos CSV."""
    os.makedirs(output_dir, exist_ok=True)
    
    X_train.to_csv(os.path.join(output_dir, 'X_train.csv'), index=False)
    X_test.to_csv(os.path.join(output_dir, 'X_test.csv'), index=False)
    y_train.to_csv(os.path.join(output_dir, 'y_train.csv'), index=False)
    y_test.to_csv(os.path.join(output_dir, 'y_test.csv'), index=False)
    
    print(f"Archivos guardados exitosamente en el directorio '{output_dir}':")
    print(f"- X_train.csv ({X_train.shape[0]} filas, {X_train.shape[1]} columnas)")
    print(f"- X_test.csv  ({X_test.shape[0]} filas, {X_test.shape[1]} columnas)")
    print(f"- y_train.csv ({len(y_train)} muestras)")
    print(f"- y_test.csv  ({len(y_test)} muestras)")

if __name__ == '__main__':
    print("1. Cargando datos y realizando Train/Test Split inicial...")
    X_train_raw, X_test_raw, y_train, y_test = load_and_split_data('bank-additional-full.csv')
    
    print("2. Ejecutando pipeline (aprendiendo parametros solo del Train)...")
    X_train, X_test = run_pipeline(X_train_raw, X_test_raw)
    
    print("3. Guardando conjuntos resultantes libres de Data Leakage...")
    save_splits(X_train, X_test, y_train, y_test)
    print("Proceso completado correctamente.")
