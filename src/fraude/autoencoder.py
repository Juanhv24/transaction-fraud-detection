"""Autoencoder en PyTorch para detección no supervisada de anomalías.

El autoencoder se entrena solo con transacciones legítimas, de modo que aprende
a reconstruir el comportamiento normal; una transacción que reconstruye mal
(error alto) se aleja de ese comportamiento y se considera anómala.
"""
from dataclasses import dataclass, field

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn

from fraude import config

LIMITE = 5.0  # recorte de las entradas estandarizadas para que unos pocos extremos no dominen la pérdida


class Autoencoder(nn.Module):
    """Codificador entrada → 64 → 16 y decodificador simétrico 16 → 64 → entrada."""

    def __init__(self, n_entradas: int, oculta: int = 64, latente: int = 16):
        super().__init__()
        self.codificador = nn.Sequential(
            nn.Linear(n_entradas, oculta), nn.ReLU(),
            nn.Linear(oculta, latente), nn.ReLU(),
        )
        self.decodificador = nn.Sequential(
            nn.Linear(latente, oculta), nn.ReLU(),
            nn.Linear(oculta, n_entradas),
        )

    def forward(self, x):
        return self.decodificador(self.codificador(x))


@dataclass
class Escalador:
    """Estandarización ajustada con las legítimas de entrenamiento, más recorte a ±LIMITE."""
    escalador: StandardScaler = field(default_factory=StandardScaler)

    def fit(self, X):
        self.escalador.fit(X)
        return self

    def transform(self, X) -> np.ndarray:
        Z = self.escalador.transform(X)
        return np.clip(np.nan_to_num(Z), -LIMITE, LIMITE).astype(np.float32)


def entrenar(X_ent: np.ndarray, X_val: np.ndarray, epocas: int = 15, tamano_lote: int = 1024,
             tasa_aprendizaje: float = 1e-3, semilla: int = config.SEMILLA) -> tuple[Autoencoder, list[dict]]:
    """Entrena el autoencoder con MSE y Adam; X_val (también legítimas) solo se usa para vigilar la pérdida."""
    torch.manual_seed(semilla)
    generador = torch.Generator().manual_seed(semilla)
    modelo = Autoencoder(X_ent.shape[1])
    optimizador = torch.optim.Adam(modelo.parameters(), lr=tasa_aprendizaje)
    perdida = nn.MSELoss()
    datos = torch.utils.data.DataLoader(
        torch.from_numpy(X_ent), batch_size=tamano_lote, shuffle=True, generator=generador)
    x_val = torch.from_numpy(X_val)

    historial = []
    for epoca in range(1, epocas + 1):
        modelo.train()
        total = 0.0
        for lote in datos:
            optimizador.zero_grad()
            error = perdida(modelo(lote), lote)
            error.backward()
            optimizador.step()
            total += error.item() * len(lote)
        modelo.eval()
        with torch.no_grad():
            val = perdida(modelo(x_val), x_val).item()
        historial.append({"epoca": epoca, "entrenamiento": total / len(X_ent), "validacion": val})
    return modelo, historial


def error_reconstruccion(modelo: Autoencoder, X: np.ndarray, tamano_lote: int = 8192) -> np.ndarray:
    """Error cuadrático medio de reconstrucción de cada fila (puntaje de anomalía)."""
    modelo.eval()
    errores = []
    with torch.no_grad():
        for i in range(0, len(X), tamano_lote):
            x = torch.from_numpy(X[i:i + tamano_lote])
            errores.append(((modelo(x) - x) ** 2).mean(dim=1).numpy())
    return np.concatenate(errores)
