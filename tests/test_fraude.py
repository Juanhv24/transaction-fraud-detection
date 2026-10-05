"""Pruebas del paquete fraude sobre una base SQLite pequeña construida en memoria."""
import sqlite3

import numpy as np
import pandas as pd
import pytest

from fraude import autoencoder as ae
from fraude import base_datos as bd
from fraude import datos, modelo, pipeline


@pytest.fixture
def conn():
    """Base mínima: una tarjeta con cinco transacciones, otra con una y dos registros de identidad."""
    conexion = sqlite3.connect(":memory:")
    trans = pd.DataFrame({
        "TransactionID": [1, 2, 3, 4, 5, 6],
        "isFraud": [0, 0, 0, 1, 0, 0],
        "TransactionDT": [86400, 86400 + 30, 86400 + 4000, 86400 + 90000, 86400 + 90001, 86400 + 50],
        "TransactionAmt": [10.0, 20.0, 30.0, 100.0, 40.0, 5.0],
        "card1": [111, 111, 111, 111, 111, 222],
        "card2": [1.0, 1.0, 1.0, 1.0, 1.0, 2.0],
        "card3": [150.0] * 6,
        "card4": ["visa"] * 5 + ["mastercard"],
        "card5": [226.0] * 6,
        "card6": ["debit"] * 6,
        "addr1": [300.0] * 6,
        # D1 crece con los días, de modo que (día - D1) es constante para la tarjeta 111
        "D1": [0.0, 0.0, 0.0, 1.0, 1.0, 0.0],
        "M4": ["M0", None, "M2", None, None, "M0"],
    })
    ident = pd.DataFrame({"TransactionID": [1, 4], "id_01": [-5.0, None], "DeviceType": ["mobile", None]})
    trans.to_sql("transactions", conexion, index=False)
    ident.to_sql("identity", conexion, index=False)
    datos.crear_features(conexion)
    yield conexion
    conexion.close()


def features(conn):
    return pd.read_sql_query("SELECT * FROM features_tarjeta ORDER BY TransactionID", conn).set_index("TransactionID")


def test_features_solo_usan_transacciones_previas(conn):
    f = features(conn)
    assert f.loc[[1, 2, 3, 4, 5], "tx_previas"].tolist() == [0, 1, 2, 3, 4]
    assert f.loc[6, "tx_previas"] == 0
    # La ventana de 24 h cubre [t - 86400, t - 1]: excluye la transacción actual,
    # lo posterior y lo ocurrido hace más de un día (la 4 ya no ve a la 1 ni a la 2)
    assert f.loc[[1, 2, 3, 4, 5], "tx_24h"].tolist() == [0, 1, 2, 1, 2]
    assert f.loc[3, "tx_1h"] == 0 and f.loc[2, "tx_1h"] == 1
    assert f.loc[2, "seg_desde_anterior"] == 30 and pd.isna(f.loc[1, "seg_desde_anterior"])
    assert f.loc[3, "monto_24h"] == pytest.approx(30.0)


def test_z_score_exige_tres_previas_y_usa_solo_el_pasado(conn):
    f = features(conn)
    assert f.loc[[1, 2, 3], "z_monto_previo"].isna().all()
    previos = np.array([10.0, 20.0, 30.0])
    esperado = (100.0 - previos.mean()) / previos.std()
    assert f.loc[4, "z_monto_previo"] == pytest.approx(esperado)
    assert f.loc[4, "ratio_monto_previo"] == pytest.approx(100.0 / 20.0)


def test_tiene_identidad(conn):
    f = features(conn)
    assert f["tiene_identidad"].to_dict() == {1: 1, 2: 0, 3: 0, 4: 1, 5: 0, 6: 0}


def test_tasa_nulos_cuenta_identidad_ausente_como_nula(conn):
    nulos = datos.tasa_nulos(conn)
    assert nulos["id_01"] == pytest.approx(5 / 6)
    assert nulos["M4"] == pytest.approx(3 / 6)
    assert nulos["TransactionAmt"] == 0
    hasta = datos.tasa_nulos(conn, hasta_dt=86400 + 60)
    assert hasta["M4"] == pytest.approx(1 / 3)


def test_cortes_y_particion_cronologica(conn):
    cortes = datos.cortes_temporales(conn, proporciones=(0.5, 0.25, 0.25))
    df = datos.cargar_datos(conn, ["TransactionAmt"])
    ent, val, pru = datos.particion_cronologica(df, cortes)
    assert len(ent) + len(val) + len(pru) == len(df)
    assert ent["TransactionDT"].max() < val["TransactionDT"].min()
    assert val["TransactionDT"].max() < pru["TransactionDT"].min()


def test_carga_conserva_texto_nulo_y_reduce_flotantes(conn):
    df = bd.leer_por_lotes("SELECT TransactionAmt, M4 FROM transactions ORDER BY TransactionID", conn, tamano_lote=2)
    assert df["TransactionAmt"].dtype == np.float32
    assert df["M4"].isna().sum() == 3 and df["M4"].iloc[0] == "M0"


def test_imputador_por_familia():
    X = pd.DataFrame({
        "D1": [np.nan, 2.0], "C1": [np.nan, 3.0], "id_02": [np.nan, 1.0], "V1": [np.nan, 4.0],
        "DeviceType": [None, "mobile"], "z_monto_previo": [np.nan, 1.5],
        "seg_desde_anterior": [np.nan, 60.0], "ratio_monto_previo": [np.nan, 2.0],
    })
    salida = pipeline.ImputadorAML().fit(X).transform(X)
    assert salida.loc[0, "D1"] == -1 and salida.loc[0, "id_02"] == -1
    assert salida.loc[0, "C1"] == 0 and salida.loc[0, "V1"] == 4.0
    assert salida.loc[0, "DeviceType"] == "mobile"
    assert salida.loc[0, "z_monto_previo"] == 0 and salida.loc[0, "ratio_monto_previo"] == 1
    assert salida.loc[0, "seg_desde_anterior"] == -1


def test_exclusor_conserva_variables_sql():
    X = pd.DataFrame({"V1": [np.nan, np.nan, 1.0], "z_monto_previo": [np.nan, np.nan, 1.0], "C1": [1, 2, 3]})
    salida = pipeline.ExclusorNulos(conservar=("z_monto_previo",)).fit(X).transform(X)
    assert list(salida.columns) == ["z_monto_previo", "C1"]


def test_umbral_y_presupuesto():
    y = np.array([0, 0, 0, 1, 1, 0, 1, 0])
    p = np.array([0.1, 0.2, 0.3, 0.9, 0.8, 0.4, 0.35, 0.05])
    elegido = modelo.umbral_f1(y, p)
    assert 0 < elegido["f1"] <= 1
    r = modelo.evaluar_umbral(y, p, 0.8)
    assert (r["vp"], r["fp"], r["fn"]) == (2, 0, 1)
    captura = modelo.captura_por_presupuesto(y, p, 0.25)
    assert captura == {"alertas": 2, "fraudes": 2, "precision": 1.0, "recall": pytest.approx(2 / 3)}


def test_lightgbm_usa_parametros_validos():
    params = modelo.lightgbm({"n_estimators": 10}).get_params()
    assert params["is_unbalance"] is True and params["subsample_freq"] == 1


def test_autoencoder_aprende_y_puntua():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(512, 8)).astype(np.float32)
    escalador = ae.Escalador().fit(X)
    Z = escalador.transform(X)
    red, historial = ae.entrenar(Z, Z[:64], epocas=5, tamano_lote=64)
    assert historial[-1]["entrenamiento"] < historial[0]["entrenamiento"]
    errores = ae.error_reconstruccion(red, Z)
    assert errores.shape == (512,) and np.all(errores >= 0)
