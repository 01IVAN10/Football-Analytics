"""Спільний стиль графіків: кольори, заголовок, підпис, збереження у файл.

Навіщо окремий модуль: радар, карти на полі, а згодом Streamlit мають виглядати
як одна система. Колір "основного гравця" міняємо в одному місці, а не в кожному файлі.
"""
import io
import re

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from src.paths import FIGURES, PROJECT_ROOT
from src.percentiles import normalize_name


# --- палітра ---
# Синій і помаранчевий — пара, яку розрізняють і люди з дальтонізмом
# (перевірено валідатором палітр: ΔE 24.7 при протанопії, поріг — 8).
# Текст завжди нейтральний: колір несе лише "чиє це / яка категорія".
BG = "#FFFFFF"
TEXT = "#1F1F1F"
MUTED = "#52514E"
LINES = "#D6D6D6"       # лінії поля, межі кілець радару
RING_FILL = "#F2F2F2"   # заливка кілець радару
CONTEXT = "#C9C9C6"     # "фонові" дані: інші паси тощо
BLUE = "#2A78D6"        # основний гравець / основна категорія
ORANGE = "#EB6834"      # гравець для порівняння / акцент (голи, ключові паси)

# Послідовна шкала для теплових карт: один тон, від кольору фону до темно-синього.
# Нуль зливається з полем, тож "порожні" зони не привертають уваги.
HEAT_CMAP = LinearSegmentedColormap.from_list(
    "heat_blue", [BG, "#CDE2FB", "#86B6EF", "#3987E5", "#256ABF", "#104281"])


def player_subtitle(row) -> str:
    """'England · Center Forward · 635 min' — підзаголовок під іменем гравця."""
    return f"{row['team']} · {row['main_position']} · {row['minutes']:.0f} min"


def draw_header(ax, left: tuple[str, str], right: tuple[str, str],
                left_color: str = TEXT, right_color: str = TEXT,
                left_size: int = 22, right_size: int = 16) -> None:
    """Заголовок у дві колонки: (назва, підзаголовок) зліва і справа."""
    for (title, sub), x, ha, color, size in [
        (left, 0.01, "left", left_color, left_size),
        (right, 0.99, "right", right_color, right_size),
    ]:
        ax.text(x, 0.68, title, fontsize=size, fontweight="bold", color=color, ha=ha, va="center")
        ax.text(x, 0.22, sub, fontsize=12, color=MUTED, ha=ha, va="center")


def draw_endnote(ax, lines: list[str]) -> None:
    """Підпис знизу: як читати графік + джерело даних."""
    ax.text(0.01, 0.9, "\n".join(lines), fontsize=9, color=MUTED, ha="left", va="top")
    ax.text(0.99, 0.9, "Data: StatsBomb Open Data", fontsize=9, color=MUTED, ha="right", va="top")


def slugify(name: str) -> str:
    """'Kylian Mbappé Lottin' -> 'kylian_mbappe_lottin' (для імені файлу)."""
    return re.sub(r"[^a-z0-9]+", "_", normalize_name(name)).strip("_")


def _savefig(fig: plt.Figure, target) -> None:
    """Однакові параметри збереження для файлу і для веб-застосунку.

    target — шлях до файлу або буфер у пам'яті (BytesIO): savefig приймає обидва.
    """
    fig.savefig(target, format="png", dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)   # звільняємо пам'ять: у циклі по гравцях фігури накопичуються


def save_figure(fig: plt.Figure, name: str) -> str:
    """Зберігає PNG у reports/figures і закриває фігуру."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / f"{name}.png"
    _savefig(fig, path)
    return str(path.relative_to(PROJECT_ROOT))


def figure_to_png(fig: plt.Figure) -> bytes:
    """PNG у пам'яті, без файлу на диску — для Streamlit.

    Байти легко кешувати (st.cache_data) і показати через st.image.
    Сам об'єкт Figure кешувати погано: він важкий і його не можна
    безпечно ділити між кількома користувачами застосунку.
    """
    buffer = io.BytesIO()
    _savefig(fig, buffer)
    return buffer.getvalue()
