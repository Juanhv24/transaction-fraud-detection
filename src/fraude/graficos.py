"""Estilo y colores compartidos por las gráficas de los notebooks."""
import matplotlib.pyplot as plt
import seaborn as sns

from fraude import config

AZUL = "#378ADD"      # legítimas, barras generales
ROJO = "#E24B4A"      # fraude, umbrales y referencias
VERDE = "#1D9E75"
OCRE = "#BA7517"
GRIS = "#8A8F98"
PALETA = [AZUL, VERDE, ROJO, OCRE]


def estilo() -> None:
    """Tema común de seaborn y matplotlib."""
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({"figure.dpi": 100, "savefig.dpi": 150, "axes.titleweight": "bold"})


def guardar(fig, nombre: str) -> None:
    """Guarda la figura en outputs/graficas con el nombre dado."""
    config.GRAFICAS.mkdir(parents=True, exist_ok=True)
    fig.savefig(config.GRAFICAS / f"{nombre}.png", bbox_inches="tight")
