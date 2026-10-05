"""Variables de comportamiento en SQL, cortes temporales y carga del dataset de modelado.

La parte pesada del trabajo (uniones, ventanas por tarjeta, tasas de nulos y
cortes temporales) se resuelve dentro de SQLite; Python solo recibe las columnas
que necesita, ya reducidas a float32.
"""
import sqlite3

import pandas as pd

from fraude import config
from fraude.base_datos import columnas, existe_tabla, leer_por_lotes, leer_sql

# Variables que crea src/fraude/sql/features_tarjeta.sql
FEATURES_SQL = [
    "tiene_identidad",
    "tx_previas",
    "seg_desde_anterior",
    "tx_1h",
    "tx_24h",
    "monto_24h",
    "monto_prom_previo",
    "z_monto_previo",
    "ratio_monto_previo",
]


def crear_features(conn: sqlite3.Connection, forzar: bool = False) -> int:
    """Crea la tabla features_tarjeta con las ventanas por tarjeta y devuelve su número de filas."""
    if forzar or not existe_tabla(conn, config.TABLA_FEATURES):
        conn.executescript(leer_sql("features_tarjeta"))
        conn.commit()
    return conn.execute(f"SELECT COUNT(*) FROM {config.TABLA_FEATURES}").fetchone()[0]


def cortes_temporales(conn: sqlite3.Connection, proporciones=config.PROPORCIONES) -> tuple[int, int]:
    """Valores de TransactionDT que separan entrenamiento, validación y prueba.

    Los cortes se ubican por posición en el orden temporal, de modo que cada
    partición tiene la proporción pedida de transacciones y todas las de
    validación y prueba son posteriores a las de entrenamiento.
    """
    total = conn.execute(f"SELECT COUNT(*) FROM {config.TABLA_TRANSACCIONES}").fetchone()[0]
    posiciones = (int(total * proporciones[0]), int(total * (proporciones[0] + proporciones[1])))
    cortes = []
    for pos in posiciones:
        cortes.append(conn.execute(
            f"SELECT {config.TIEMPO} FROM {config.TABLA_TRANSACCIONES} "
            f"ORDER BY {config.TIEMPO}, {config.ID} LIMIT 1 OFFSET ?", (pos,)
        ).fetchone()[0])
    return cortes[0], cortes[1]


def tasa_nulos(conn: sqlite3.Connection, hasta_dt: int | None = None) -> pd.Series:
    """Proporción de nulos de cada columna de transacciones e identidad, calculada en SQL.

    Las columnas de identidad se evalúan sobre la unión LEFT JOIN, de modo que una
    transacción sin registro de identidad cuenta como nula, igual que en el
    DataFrame unificado. Con hasta_dt se usa solo el periodo de entrenamiento.
    """
    trans = [c for c in columnas(conn, config.TABLA_TRANSACCIONES) if c != config.ID]
    ident = [c for c in columnas(conn, config.TABLA_IDENTIDAD) if c != config.ID]
    expresiones = [f'AVG(t."{c}" IS NULL) AS "{c}"' for c in trans]
    expresiones += [f'AVG(i."{c}" IS NULL) AS "{c}"' for c in ident]
    filtro = f"WHERE t.{config.TIEMPO} < {int(hasta_dt)}" if hasta_dt is not None else ""
    sql = (
        f"SELECT {', '.join(expresiones)} "
        f"FROM {config.TABLA_TRANSACCIONES} t "
        f"LEFT JOIN {config.TABLA_IDENTIDAD} i ON i.{config.ID} = t.{config.ID} {filtro}"
    )
    return pd.read_sql_query(sql, conn).iloc[0].astype(float).sort_values(ascending=False)


def columnas_utiles(conn: sqlite3.Connection, hasta_dt: int, umbral: float = config.UMBRAL_NULOS) -> dict:
    """Columnas con proporción de nulos menor o igual al umbral en el periodo de entrenamiento."""
    nulos = tasa_nulos(conn, hasta_dt)
    conservar = set(nulos[nulos <= umbral].index)
    trans = [c for c in columnas(conn, config.TABLA_TRANSACCIONES) if c in conservar]
    ident = [c for c in columnas(conn, config.TABLA_IDENTIDAD) if c in conservar]
    return {"transacciones": trans, "identidad": ident, "excluidas": sorted(set(nulos.index) - conservar)}


def cargar_datos(
    conn: sqlite3.Connection,
    columnas_transacciones: list[str] | None = None,
    columnas_identidad: list[str] | None = None,
    con_features: bool = True,
) -> pd.DataFrame:
    """Une transacciones, identidad y features SQL en un DataFrame ordenado por tiempo.

    TransactionID, isFraud y TransactionDT siempre se incluyen. Con None en
    columnas_transacciones se traen todas las columnas de la tabla.
    """
    base = [config.ID, config.OBJETIVO, config.TIEMPO]
    if columnas_transacciones is None:
        columnas_transacciones = columnas(conn, config.TABLA_TRANSACCIONES)
    trans = base + [c for c in columnas_transacciones if c not in base]
    seleccion = [f't."{c}"' for c in trans]
    seleccion += [f'i."{c}"' for c in (columnas_identidad or [])]
    union = f"LEFT JOIN {config.TABLA_IDENTIDAD} i ON i.{config.ID} = t.{config.ID} " if columnas_identidad else ""
    if con_features:
        seleccion += [f"f.{c}" for c in FEATURES_SQL]
        union += f"LEFT JOIN {config.TABLA_FEATURES} f ON f.{config.ID} = t.{config.ID} "
    sql = (
        f"SELECT {', '.join(seleccion)} FROM {config.TABLA_TRANSACCIONES} t {union}"
        f"ORDER BY t.{config.TIEMPO}, t.{config.ID}"
    )
    return leer_por_lotes(sql, conn)


def particion_cronologica(df: pd.DataFrame, cortes: tuple[int, int]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Divide un DataFrame en entrenamiento, validación y prueba según los cortes de TransactionDT."""
    dt = df[config.TIEMPO]
    entrenamiento = df[dt < cortes[0]]
    validacion = df[(dt >= cortes[0]) & (dt < cortes[1])]
    prueba = df[dt >= cortes[1]]
    return entrenamiento, validacion, prueba
