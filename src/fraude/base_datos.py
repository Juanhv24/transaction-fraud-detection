"""Creación de la base SQLite, conexión y consultas.

Todos los notebooks se conectan a la base a través de este módulo, de modo que
la ruta se define una sola vez en `config.py` y no depende del equipo.
"""
import sqlite3
from contextlib import closing
from pathlib import Path

import numpy as np
import pandas as pd

from fraude import config


def conectar(ruta: Path = config.BASE) -> sqlite3.Connection:
    """Abre la base SQLite del proyecto y falla con un mensaje claro si no existe."""
    ruta = Path(ruta)
    if not ruta.exists():
        raise FileNotFoundError(
            f"No se encontró la base {ruta}. La base se crea con el notebook 00 (o con "
            f"fraude.base_datos.crear_base()) a partir de los CSV de {config.URL_DATOS} ubicados en data/."
        )
    return sqlite3.connect(ruta)


def leer_sql(nombre: str) -> str:
    """Lee un script de la carpeta src/fraude/sql."""
    return (config.SQL / f"{nombre}.sql").read_text(encoding="utf-8")


def consultar(sql: str, conn: sqlite3.Connection | None = None, params=None) -> pd.DataFrame:
    """Ejecuta una consulta y devuelve un DataFrame; abre y cierra la conexión si no se pasa una."""
    if conn is not None:
        return pd.read_sql_query(sql, conn, params=params)
    with closing(conectar()) as propia:
        return pd.read_sql_query(sql, propia, params=params)


def existe_tabla(conn: sqlite3.Connection, tabla: str) -> bool:
    fila = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (tabla,)
    ).fetchone()
    return fila is not None


def columnas(conn: sqlite3.Connection, tabla: str) -> list[str]:
    """Nombres de las columnas de una tabla, en el orden en que se crearon."""
    return [fila[1] for fila in conn.execute(f"PRAGMA table_info({tabla})")]


def crear_indices(conn: sqlite3.Connection) -> None:
    """Índices para las uniones por TransactionID y los filtros por tiempo."""
    conn.executescript(f"""
        CREATE UNIQUE INDEX IF NOT EXISTS ix_transactions_id
            ON {config.TABLA_TRANSACCIONES} ({config.ID});
        CREATE INDEX IF NOT EXISTS ix_transactions_dt
            ON {config.TABLA_TRANSACCIONES} ({config.TIEMPO});
        CREATE UNIQUE INDEX IF NOT EXISTS ix_identity_id
            ON {config.TABLA_IDENTIDAD} ({config.ID});
    """)
    conn.commit()


def crear_base(ruta: Path = config.BASE, forzar: bool = False, tamano_lote: int = 50_000) -> str:
    """Convierte los CSV del IEEE-CIS en la base SQLite del proyecto.

    Los CSV se leen por lotes para no cargar los 683 MB de transacciones en
    memoria. Si la base ya existe con ambas tablas, solo se asegura que tenga
    los índices, a menos que se pida reconstruirla con forzar=True.
    """
    ruta = Path(ruta)
    if ruta.exists() and not forzar:
        with closing(sqlite3.connect(ruta)) as conn:
            if all(existe_tabla(conn, t) for t in (config.TABLA_TRANSACCIONES, config.TABLA_IDENTIDAD)):
                crear_indices(conn)
                return "existente"

    faltantes = [p.name for p in (config.CSV_TRANSACCIONES, config.CSV_IDENTIDAD) if not p.exists()]
    if faltantes:
        raise FileNotFoundError(
            f"Faltan {', '.join(faltantes)} en {config.DATA}. Se descargan de {config.URL_DATOS}"
        )

    if ruta.exists():
        ruta.unlink()
    with closing(sqlite3.connect(ruta)) as conn:
        for csv, tabla in ((config.CSV_TRANSACCIONES, config.TABLA_TRANSACCIONES),
                           (config.CSV_IDENTIDAD, config.TABLA_IDENTIDAD)):
            for i, lote in enumerate(pd.read_csv(csv, chunksize=tamano_lote)):
                lote.to_sql(tabla, conn, if_exists="replace" if i == 0 else "append", index=False)
        crear_indices(conn)
    return "creada"


def resumen_tablas(conn: sqlite3.Connection) -> pd.DataFrame:
    """Filas y columnas de cada tabla de la base."""
    tablas = [f[0] for f in conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name")]
    filas = [
        {
            "tabla": t,
            "filas": conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0],
            "columnas": len(columnas(conn, t)),
        }
        for t in tablas
    ]
    return pd.DataFrame(filas)


def compactar(df: pd.DataFrame) -> pd.DataFrame:
    """Reduce la memoria de un lote: float64 a float32 y texto a categoría.

    Pasar el texto a categoría libera los miles de cadenas que crea la lectura
    fila a fila; si se conservaran, quedarían dispersas entre los bloques de
    memoria de Python e impedirían liberarlos al terminar cada lote.
    """
    tipos = {col: np.float32 for col in df.select_dtypes(include=["float64"]).columns}
    tipos |= {col: "category" for col in df.select_dtypes(exclude=["number"]).columns}
    return df.astype(tipos)


def leer_por_lotes(sql: str, conn: sqlite3.Connection, params=None, tamano_lote: int = 20_000) -> pd.DataFrame:
    """Lee una consulta grande por lotes, compacta cada lote y los une.

    Las columnas de texto vuelven a ser texto al final; las categorías de cada
    lote solo sirven para no acumular cadenas repetidas durante la lectura.
    """
    lotes = pd.read_sql_query(sql, conn, params=params, chunksize=tamano_lote)
    df = pd.concat([compactar(lote) for lote in lotes], ignore_index=True)
    texto = df.select_dtypes(exclude=["number"]).columns
    return df.astype({col: "object" for col in texto})
