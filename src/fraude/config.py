"""Rutas y parámetros compartidos por los notebooks y los módulos del proyecto."""
from pathlib import Path

# Raíz del repositorio: src/fraude/config.py -> src/fraude -> src -> raíz
RAIZ = Path(__file__).resolve().parents[2]

DATA = RAIZ / "data"
BASE = DATA / "ieee_fraud.sqlite"
CSV_TRANSACCIONES = DATA / "train_transaction.csv"
CSV_IDENTIDAD = DATA / "train_identity.csv"
URL_DATOS = "https://www.kaggle.com/competitions/ieee-fraud-detection/data"

SALIDAS = RAIZ / "outputs"
GRAFICAS = SALIDAS / "graficas"
MODELOS = SALIDAS / "modelos"

SQL = Path(__file__).resolve().parent / "sql"

# Columnas clave
ID = "TransactionID"
OBJETIVO = "isFraud"
TIEMPO = "TransactionDT"
MONTO = "TransactionAmt"

# Tablas de la base
TABLA_TRANSACCIONES = "transactions"
TABLA_IDENTIDAD = "identity"
TABLA_FEATURES = "features_tarjeta"

# Particiones cronológicas: entrenamiento, validación y prueba
PROPORCIONES = (0.6, 0.2, 0.2)

UMBRAL_NULOS = 0.5
SEMILLA = 42
