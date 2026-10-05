"""Transformadores y pipeline de preprocesamiento.

Al vivir en el paquete y no en el notebook, el pipeline guardado con joblib se
puede volver a cargar desde cualquier notebook o script que importe fraude.
"""
import re

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, RobustScaler

from fraude import config
from fraude.datos import FEATURES_SQL

PATRON_ID = re.compile(r"^id_\d+$")
PATRON_D = re.compile(r"^D\d+$")
PATRON_C = re.compile(r"^C\d+$")

# Valores de relleno para las variables SQL cuando la tarjeta no tiene historial:
# sin transacción anterior no hay tiempo transcurrido (-1 lo marca), el z-score
# neutro es 0 y la razón neutra frente al promedio previo es 1.
RELLENO_FEATURES_SQL = {"seg_desde_anterior": -1.0, "z_monto_previo": 0.0, "ratio_monto_previo": 1.0}


class CreadorFeaturesTemporales(BaseEstimator, TransformerMixin):
    """Extrae hora y día desde TransactionDT.

    TransactionDT es un delta en segundos desde una referencia desconocida, por
    lo que la hora y el día son relativos a esa referencia y no al reloj local.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_out = X.copy()
        X_out["hora"] = (X_out[config.TIEMPO] // 3600) % 24
        X_out["dia_semana"] = (X_out[config.TIEMPO] // 86400) % 7
        return X_out


class TransformadorLogMonto(BaseEstimator, TransformerMixin):
    """Aplica log1p a las columnas de monto para reducir el impacto de los valores extremos."""

    def __init__(self, columnas=(config.MONTO, "monto_24h", "monto_prom_previo")):
        self.columnas = columnas

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_out = X.copy()
        for col in self.columnas:
            if col in X_out.columns:
                X_out[col] = np.log1p(X_out[col])
        return X_out


class EliminadorColumnas(BaseEstimator, TransformerMixin):
    """Elimina columnas específicas del DataFrame."""

    def __init__(self, columnas):
        self.columnas = columnas

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop(columns=[c for c in self.columnas if c in X.columns])


class ExclusorNulos(BaseEstimator, TransformerMixin):
    """Excluye columnas con porcentaje de nulos superior al umbral.

    El umbral se aprende en fit() sobre el conjunto de entrenamiento únicamente.
    Las columnas de `conservar` no se evalúan: en las variables SQL un nulo no es
    un dato perdido sino una tarjeta sin historial suficiente, y se imputa con un
    valor neutro.
    """

    def __init__(self, umbral=config.UMBRAL_NULOS, conservar=()):
        self.umbral = umbral
        self.conservar = conservar

    def fit(self, X, y=None):
        nulos = X.isnull().mean()
        self.cols_conservar_ = [c for c in X.columns if nulos[c] <= self.umbral or c in self.conservar]
        return self

    def transform(self, X):
        return X[self.cols_conservar_]


class ImputadorAML(BaseEstimator, TransformerMixin):
    """Imputa cada familia de variables según su naturaleza.

    - Variables id_ y D numéricas: -1, porque la ausencia puede ser informativa y
      0 es un valor válido en las D (misma fecha).
    - Variables C: 0, porque son conteos y un nulo equivale a cero ocurrencias.
    - Variables SQL sin historial: valores neutros (RELLENO_FEATURES_SQL).
    - Resto de numéricas: mediana del entrenamiento.
    - Categóricas: moda del entrenamiento.
    """

    def fit(self, X, y=None):
        numericas = X.select_dtypes(include=[np.number])
        categoricas = X.select_dtypes(exclude=[np.number])
        self.medianas_ = numericas.median()
        self.modas_ = categoricas.mode().iloc[0] if not categoricas.empty else pd.Series(dtype="object")
        return self

    def transform(self, X):
        X_out = X.copy()
        numericas = set(X_out.select_dtypes(include=[np.number]).columns)

        centinela = [c for c in numericas if PATRON_ID.match(c) or PATRON_D.match(c)]
        X_out[centinela] = X_out[centinela].fillna(-1)

        conteos = [c for c in numericas if PATRON_C.match(c)]
        X_out[conteos] = X_out[conteos].fillna(0)

        for col, valor in RELLENO_FEATURES_SQL.items():
            if col in X_out.columns:
                X_out[col] = X_out[col].fillna(valor)

        X_out = X_out.fillna(self.medianas_.reindex(X_out.columns).dropna().to_dict())
        X_out = X_out.fillna(self.modas_.reindex(X_out.columns).dropna().to_dict())
        return X_out


def preparar(X: pd.DataFrame) -> Pipeline:
    """Pasos previos al escalado (temporales, log, exclusión de nulos e imputación)."""
    return Pipeline([
        ("temporales", CreadorFeaturesTemporales()),
        ("log_monto", TransformadorLogMonto()),
        ("eliminar_dt", EliminadorColumnas([config.TIEMPO])),
        ("excluir_nulos", ExclusorNulos(conservar=tuple(FEATURES_SQL))),
        ("imputador", ImputadorAML()),
    ])


def construir_pipeline(X_entrenamiento: pd.DataFrame) -> Pipeline:
    """Pipeline completo: preparación más escalado robusto y codificación ordinal.

    Las listas de columnas numéricas y categóricas se obtienen aplicando la
    preparación al entrenamiento, por lo que nunca se mira validación ni prueba.
    """
    muestra = preparar(X_entrenamiento).fit_transform(X_entrenamiento)
    num_cols = muestra.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = muestra.select_dtypes(exclude=[np.number]).columns.tolist()

    preprocesador = ColumnTransformer(
        transformers=[
            ("num", RobustScaler(), num_cols),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
            ]), cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    pasos = preparar(X_entrenamiento).steps + [("preprocesador", preprocesador)]
    return Pipeline(pasos)


def nombres_salida(pipeline: Pipeline) -> list[str]:
    """Nombres de las columnas que entrega el pipeline ya ajustado."""
    return list(pipeline.named_steps["preprocesador"].get_feature_names_out())
