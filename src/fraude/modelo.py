"""Modelos, métricas, optimización con Optuna, umbral de decisión y persistencia."""
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score,
                             roc_auc_score)

from fraude import config


def metricas(y, proba) -> dict:
    """ROC-AUC, PR-AUC (precisión promedio) y GINI de un vector de probabilidades."""
    roc = roc_auc_score(y, proba)
    return {
        "ROC-AUC": round(roc, 4),
        "PR-AUC": round(average_precision_score(y, proba), 4),
        "GINI": round((2 * roc - 1) * 100, 2),
    }


def razon_desbalance(y) -> float:
    """Legítimas por cada fraude; se usa como scale_pos_weight."""
    y = np.asarray(y)
    return float((y == 0).sum() / (y == 1).sum())


def modelos_base(ratio: float, semilla: int = config.SEMILLA) -> dict:
    """Los cuatro modelos de la comparación inicial, todos con manejo del desbalance."""
    return {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=semilla),
        "Random Forest": RandomForestClassifier(
            class_weight="balanced", n_estimators=100,
            random_state=semilla, n_jobs=-1),
        "XGBoost": xgb.XGBClassifier(
            scale_pos_weight=ratio, n_estimators=200, random_state=semilla,
            n_jobs=-1, verbosity=0),
        # is_unbalance es el nombre correcto del parámetro en LightGBM
        "LightGBM": lgb.LGBMClassifier(
            is_unbalance=True, n_estimators=200, random_state=semilla,
            n_jobs=-1, verbose=-1),
    }


def comparar(modelos: dict, X_ent, y_ent, X_val, y_val) -> tuple[pd.DataFrame, dict]:
    """Entrena cada modelo y lo evalúa en validación; devuelve la tabla y los modelos ajustados."""
    filas, ajustados = [], {}
    for nombre, modelo in modelos.items():
        X_e, X_v = X_ent, X_val
        if isinstance(modelo, LogisticRegression):
            # lbfgs pierde precisión y se detiene antes de converger con float32
            X_e, X_v = np.asarray(X_ent, dtype=np.float64), np.asarray(X_val, dtype=np.float64)
        modelo.fit(X_e, y_ent)
        proba = modelo.predict_proba(X_v)[:, 1]
        filas.append({"Modelo": nombre, **metricas(y_val, proba)})
        ajustados[nombre] = modelo
    tabla = pd.DataFrame(filas).sort_values("PR-AUC", ascending=False).reset_index(drop=True)
    return tabla, ajustados


def espacio_lightgbm(trial: optuna.Trial) -> dict:
    """Espacio de búsqueda de hiperparámetros para LightGBM."""
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 20, 100),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
    }


def lightgbm(params: dict, semilla: int = config.SEMILLA) -> lgb.LGBMClassifier:
    """LightGBM con los parámetros dados, manejo del desbalance y submuestreo activo.

    subsample solo tiene efecto en LightGBM si subsample_freq es mayor que cero.
    """
    return lgb.LGBMClassifier(
        **params, is_unbalance=True, subsample_freq=1,
        random_state=semilla, n_jobs=-1, verbose=-1,
    )


def optimizar_lightgbm(X_ent, y_ent, X_val, y_val, n_trials: int = 20,
                       semilla: int = config.SEMILLA) -> optuna.Study:
    """Optimización bayesiana (TPE) que maximiza el PR-AUC en validación, nunca en prueba."""
    def objetivo(trial):
        modelo = lightgbm(espacio_lightgbm(trial), semilla)
        modelo.fit(X_ent, y_ent)
        return average_precision_score(y_val, modelo.predict_proba(X_val)[:, 1])

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    estudio = optuna.create_study(direction="maximize",
                                  sampler=optuna.samplers.TPESampler(seed=semilla))
    estudio.optimize(objetivo, n_trials=n_trials)
    return estudio


def umbral_f1(y, proba) -> dict:
    """Umbral que maximiza el F1 sobre la curva precisión-recall."""
    precision, recall, umbrales = precision_recall_curve(y, proba)
    f1 = 2 * precision[:-1] * recall[:-1] / np.clip(precision[:-1] + recall[:-1], 1e-12, None)
    i = int(np.argmax(f1))
    return {"umbral": float(umbrales[i]), "precision": float(precision[i]),
            "recall": float(recall[i]), "f1": float(f1[i])}


def evaluar_umbral(y, proba, umbral: float) -> dict:
    """Matriz de confusión y métricas de clasificación con un umbral fijo."""
    pred = (proba >= umbral).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    return {
        "umbral": umbral,
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred),
        "f1": f1_score(y, pred),
        "tasa_alertas": float(pred.mean()),
        "vp": int(tp), "fp": int(fp), "fn": int(fn), "vn": int(tn),
    }


def captura_por_presupuesto(y, puntaje, presupuesto: float) -> dict:
    """Fraudes capturados si se revisa solo la fracción `presupuesto` de transacciones con mayor puntaje."""
    y = np.asarray(y)
    n = max(1, int(round(len(y) * presupuesto)))
    top = np.argsort(-np.asarray(puntaje))[:n]
    return {"alertas": n, "fraudes": int(y[top].sum()),
            "precision": float(y[top].mean()), "recall": float(y[top].sum() / y.sum())}


def guardar(objeto, nombre: str, carpeta: Path = config.MODELOS) -> Path:
    """Guarda un objeto con joblib en outputs/modelos."""
    carpeta.mkdir(parents=True, exist_ok=True)
    ruta = carpeta / nombre
    joblib.dump(objeto, ruta)
    return ruta


def cargar(nombre: str, carpeta: Path = config.MODELOS):
    """Carga un objeto guardado con guardar()."""
    return joblib.load(carpeta / nombre)
