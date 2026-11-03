"""Policy manual retrieval.

The manual is short (10 sections, roughly 3k tokens), so instead of embeddings
we split it by its own numbered headings and let the model pick the section
from a table of contents. Retrieval is deterministic and every fetch is
visible in the trace, which makes "did the agent check the policy?" auditable.
"""

from __future__ import annotations

import re
from pathlib import Path

from .data import DEFAULT_DATA_DIR

# Stable keys the model uses in the consultar_politica tool, mapped to the
# section number in the manual.
SECTION_KEYS = {
    "sobre_e_contato": 1,
    "horario": 2,
    "pagamento": 3,
    "trocas_e_devolucoes": 4,
    "frete_e_entregas": 5,
    "promocoes": 6,
    "atendimento_e_escopo": 7,
    "garantia": 8,
    "privacidade": 9,
    "disposicoes_finais": 10,
}

SECTION_HINTS = {
    "sobre_e_contato": "história, o que a loja vende e não vende, endereço, telefone, e-mail, CNPJ",
    "horario": "horário de funcionamento da loja e do WhatsApp",
    "pagamento": "formas de pagamento, desconto PIX, parcelamento e parcela mínima",
    "trocas_e_devolucoes": "arrependimento, troca por defeito, troca por preferência, itens não elegíveis",
    "frete_e_entregas": "frete grátis, taxas, prazos, Correios e Jadlog, rastreamento",
    "promocoes": "calendário de campanhas, regras de cumulatividade e transparência",
    "atendimento_e_escopo": "tom de voz, escopo (só instrumentos), situações especiais, reclamações",
    "garantia": "garantia legal de 90 dias, garantia do fabricante, o que não cobre",
    "privacidade": "LGPD, uso de dados pessoais, exclusão de dados",
    "disposicoes_finais": "uso do manual e quando consultar a gerência",
}


def load_sections(path: Path | str = DEFAULT_DATA_DIR / "politicas_da_loja.md") -> dict[str, str]:
    text = Path(path).read_text(encoding="utf-8")
    chunks = re.split(r"^## ", text, flags=re.MULTILINE)[1:]
    by_number: dict[int, str] = {}
    for chunk in chunks:
        match = re.match(r"(\d+)\.", chunk)
        if match:
            by_number[int(match.group(1))] = "## " + chunk.strip()
    return {key: by_number[num] for key, num in SECTION_KEYS.items() if num in by_number}


def table_of_contents() -> str:
    return "\n".join(f"- {key}: {hint}" for key, hint in SECTION_HINTS.items())
