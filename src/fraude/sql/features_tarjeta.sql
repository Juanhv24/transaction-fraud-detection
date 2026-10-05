-- =============================================================================
-- Variables de comportamiento por tarjeta
-- Cada variable se calcula solo con las transacciones ANTERIORES de la misma
-- tarjeta (marcos de ventana que terminan en 1 PRECEDING), de modo que el valor
-- que recibe una transacción es el que se conocería en el momento de autorizarla.
-- Ninguna variable usa isFraud, por lo que no hay fuga de la etiqueta.
-- =============================================================================
DROP TABLE IF EXISTS features_tarjeta;

CREATE TABLE features_tarjeta AS
WITH base AS (
    SELECT
        t.TransactionID,
        t.TransactionDT,
        t.TransactionAmt,
        -- El dataset no trae un identificador de tarjeta. Se aproxima con las
        -- variables de la tarjeta (card1 a card6), la dirección de facturación
        -- (addr1) y el día en que empezó la relación con la tarjeta, que se
        -- obtiene restando D1 (días desde el inicio) al día de la transacción.
        t.card1 || '-' || IFNULL(t.card2, 'na') || '-' || IFNULL(t.card3, 'na') || '-' ||
        IFNULL(t.card5, 'na') || '-' || IFNULL(t.card4, 'na') || '-' ||
        IFNULL(t.card6, 'na') || '-' || IFNULL(CAST(t.addr1 AS INTEGER), 'na') || '-' ||
        IFNULL(CAST(t.TransactionDT / 86400 - t.D1 AS INTEGER), 'na') AS tarjeta,
        CASE WHEN i.TransactionID IS NULL THEN 0 ELSE 1 END    AS tiene_identidad
    FROM transactions t
    LEFT JOIN identity i ON i.TransactionID = t.TransactionID
),
ventanas AS (
    SELECT
        TransactionID,
        tarjeta,
        TransactionAmt,
        tiene_identidad,
        COUNT(*)                            OVER previas AS tx_previas,
        AVG(TransactionAmt)                 OVER previas AS monto_prom_previo,
        AVG(TransactionAmt * TransactionAmt) OVER previas AS monto_cuad_previo,
        TransactionDT - LAG(TransactionDT)  OVER orden   AS seg_desde_anterior,
        COUNT(*)                            OVER ult_1h  AS tx_1h,
        COUNT(*)                            OVER ult_24h AS tx_24h,
        SUM(TransactionAmt)                 OVER ult_24h AS monto_24h
    FROM base
    WINDOW
        orden   AS (PARTITION BY tarjeta ORDER BY TransactionDT, TransactionID),
        previas AS (PARTITION BY tarjeta ORDER BY TransactionDT, TransactionID
                    ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING),
        ult_1h  AS (PARTITION BY tarjeta ORDER BY TransactionDT
                    RANGE BETWEEN 3600 PRECEDING AND 1 PRECEDING),
        ult_24h AS (PARTITION BY tarjeta ORDER BY TransactionDT
                    RANGE BETWEEN 86400 PRECEDING AND 1 PRECEDING)
)
SELECT
    TransactionID,
    tarjeta,
    tiene_identidad,
    tx_previas,
    seg_desde_anterior,
    tx_1h,
    tx_24h,
    IFNULL(monto_24h, 0)                                         AS monto_24h,
    monto_prom_previo,
    -- z-score del monto frente al historial previo de la tarjeta; se exige un
    -- mínimo de tres transacciones previas para que la desviación sea estable.
    CASE
        WHEN tx_previas >= 3
         AND monto_cuad_previo - monto_prom_previo * monto_prom_previo > 0
        THEN (TransactionAmt - monto_prom_previo)
             / SQRT(monto_cuad_previo - monto_prom_previo * monto_prom_previo)
    END                                                          AS z_monto_previo,
    TransactionAmt / monto_prom_previo                           AS ratio_monto_previo
FROM ventanas;

CREATE UNIQUE INDEX IF NOT EXISTS ix_features_tarjeta_id ON features_tarjeta (TransactionID);
