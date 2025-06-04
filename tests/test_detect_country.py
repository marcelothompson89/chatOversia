import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from chatbot import QueryProcessor

import pytest

@pytest.mark.parametrize("query,expected", [
    ("Alertas de México", "México"),
    ("alertas de mexico", "México"),
    ("Alerta en Perú", "Perú"),
    ("Novedades peru", "Perú"),
    ("Normativa EE.UU.", "Estados Unidos"),
    ("leyes eeuu", "Estados Unidos"),
    ("analisis usa", "Estados Unidos"),
    ("reglas eua", "Estados Unidos"),
    ("sin pais", None),
])
def test_detect_country(query, expected):
    assert QueryProcessor.detect_country(query) == expected
