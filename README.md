# EV4 — Preprocesamiento de Datos (ADY1100)

Este proyecto realiza el análisis exploratorio y preprocesamiento del dataset "Bank Marketing" para preparar los datos para modelos de Machine Learning.

## Metodología y Orden del Proceso (Evitando Data Leakage)

Para seguir las mejores prácticas de Ciencia de Datos y asegurar que no exista fuga de información (*data leakage*), el proceso está estrictamente ordenado de la siguiente manera:

1. **Carga y Train/Test Split:** Los datos se dividen inmediatamente en Entrenamiento (80%) y Prueba (20%).
2. **Análisis Exploratorio (EDA):** Toda la exploración, gráficos y decisiones lógicas se toman en el notebook basándose idealmente en el conjunto de entrenamiento.
3. **Pipeline de Preprocesamiento:** Se aplican las transformaciones matemáticas y de limpieza. Parámetros como la imputación por la moda, los límites de cuartiles (IQR) para outliers y la media/desviación estándar del `StandardScaler` **se aprenden exclusivamente del conjunto de Train** y luego se aplican a ambos conjuntos.

## Cómo ejecutar

1. Crear y activar el entorno virtual:
   ```bash
   python -m venv ven
   source ven/bin/activate      # Windows: ven\Scripts\activate
   ```

2. Instalar dependencias:
   ```bash
   pip install -r requirements.txt
   ```

3. **Ejecutar el Pipeline Separado (Recomendado)**
   Hemos empaquetado el preprocesamiento limpio en un script de Python. Esto generará los datasets listos para modelos en la carpeta `data/`.
   ```bash
   python pipeline.py
   ```

4. **Explorar el Notebook Interactivo**
   ```bash
   jupyter notebook
   ```
   Abre `EV4_Analisis Exploratorio.ipynb` para visualizar los gráficos de distribuciones, el análisis bivariado y las conclusiones del EDA.

> Asegúrate de tener el archivo CSV original (`bank-additional-full.csv`) en la misma carpeta antes de ejecutar.
