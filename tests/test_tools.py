"""Deterministic tests for the tool layer (no LLM, no API key needed)."""

import json
import sys
from datetime import datetime
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from emporio_agent.data import load_data  # noqa: E402
from emporio_agent.policies import load_sections  # noqa: E402
from emporio_agent.tools import STORE_TZ, StoreTools  # noqa: E402

DATA = load_data()
SECTIONS = load_sections()


def make_tools(now: datetime, tmp_path: Path) -> StoreTools:
    return StoreTools(DATA, SECTIONS, now_fn=lambda: now, escalation_log=tmp_path / "esc.jsonl")


@pytest.fixture
def tools(tmp_path):
    # Tuesday, 10:00 in Campo Grande
    return make_tools(datetime(2026, 3, 24, 10, 0, tzinfo=STORE_TZ), tmp_path)


# ----------------------------------------------------------------- catalog
def test_violoes_ate_1000_only_sellable_and_within_budget(tools):
    result = tools.buscar_produtos(categoria="Violões", preco_max=1000)
    names = [p["nome"] for p in result["produtos"]]
    assert names, "expected guitars under R$ 1000"
    assert all(p["disponibilidade"] == "disponível" for p in result["produtos"])
    omitted = {p["nome"] for p in result["indisponiveis_omitidos"]}
    assert "Giannini GF-3D Dreadnought Sunburst" in omitted  # active but zero stock
    assert "Shelby SN-7C 7 Cordas Nylon Natural" in omitted  # discontinued


def test_partial_search_result_says_so(tools):
    # 12 sellable guitars up to R$ 1000, 8 shown by default: the agent must be told the list is partial
    result = tools.buscar_produtos(categoria="Violões", preco_max=1000)
    assert result["total_disponiveis"] == 12
    assert len(result["produtos"]) == 8
    assert result["lista_parcial"] is True
    assert "4" in result["observacao"]
    full = tools.buscar_produtos(categoria="Violões", preco_max=1000, limite=20)
    assert len(full["produtos"]) == 12 and "lista_parcial" not in full


def test_plural_and_accents_in_free_text(tools):
    result = tools.buscar_produtos(termo="violões nylon", limite=20)
    assert result["total_encontrados"] > 0
    for p in result["produtos"]:
        assert p["categoria"] == "Violões"  # "violões" became a category filter
        assert DATA.product_by_id(p["id"])["specs"]["strings"] == "nylon"


def test_numeric_tokens_are_kept(tools):
    result = tools.buscar_produtos(termo="violão 7 cordas", limite=20)
    assert result["produtos"] and all("7 Cordas" in p["nome"] for p in result["produtos"])


def test_takamine_gd20_price_and_expired_promo(tools):
    result = tools.detalhes_produto("Takamine GD20")
    assert result["encontrado"] is True
    assert result["preco_tabela"] == "R$ 2.199,00"
    assert "promocao_ativa" not in result
    assert result["promocoes_encerradas"] == ["Oferta Especial"]
    assert result["preco_no_pix"].startswith("R$ 2.089,05")


def test_active_promo_is_not_combined_with_pix(tools):
    result = tools.detalhes_produto("Taylor 110e")
    assert result["promocao_ativa"]["desconto_percentual"] == 8
    assert result["preco_no_pix"].startswith(result["promocao_ativa"]["preco_promocional"])


def test_coming_soon_is_not_sellable(tools):
    result = tools.detalhes_produto("130")
    assert result["disponibilidade"].startswith("em breve")
    sim = tools.simular_pagamento([{"produto_id": 130}])
    assert "erro" in sim


def test_ambiguous_name_asks_for_confirmation(tools):
    result = tools.detalhes_produto("Martin D-28")
    assert result["encontrado"] is False
    assert set(result["possiveis_produtos"]) >= {"Martin D-28 Dreadnought Natural", "Martin D-28E Elétrico Aço Natural"}


# ----------------------------------------------------------------- payment
def test_installments_respect_minimum_values(tools):
    # Kalani KAL-700T: 359.00 with 10% promo -> 323.10. 4x = 80.78 (>= R$80), 5x = 64.62 (< R$80)
    sim = tools.simular_pagamento([{"produto_id": 127}])
    assert sim["total"] == "R$ 323,10"
    assert sim["pix"] == "R$ 323,10"
    assert sim["cartao_credito"] == "até 4x de R$ 80,78 sem juros"


def test_installments_for_expensive_item(tools):
    sim = tools.simular_pagamento([{"produto_id": 95}])  # 2199.00
    assert sim["cartao_credito"] == "até 12x de R$ 183,25 sem juros"
    assert sim["combinar_formas_de_pagamento"] is True


def test_pix_price_rounds_half_up_to_the_cent(tools):
    # 599.90 * 0.95 = 569.905 exactly; float round() gave 569.90 (binary 569.90499...)
    assert tools.detalhes_produto("81")["preco_no_pix"].startswith("R$ 569,91")
    sim = tools.simular_pagamento([{"produto_id": 81, "quantidade": 2}])
    assert sim["pix"] == "R$ 1.139,82"
    listed = {p["id"]: p["preco_no_pix"] for p in tools.buscar_produtos(categoria="Violões", preco_max=1000)["produtos"]}
    assert listed[83].startswith("R$ 408,41")  # 429.90 * 0.95 = 408.405


# ----------------------------------------------------------------- orders
def test_order_requires_matching_contact(tools):
    denied = tools.consultar_pedido(8, "someone@else.com")
    assert denied["encontrado"] is False
    missing = tools.consultar_pedido(999, "anacarol.ferreira@coldmail.com")
    assert missing == denied  # no way to tell "wrong contact" from "no such order"


def test_order_by_phone_and_email(tools):
    by_email = tools.consultar_pedido(8, "AnaCarol.Ferreira@coldmail.com")
    by_phone = tools.consultar_pedido("8", "67 99654-7821")
    assert by_email["status"] == "enviado" == by_phone["status"]
    assert by_email["codigo_rastreio"] == "BRJL5544332BR"


def test_late_shipped_order_raises_alert(tmp_path):
    on_time = make_tools(datetime(2026, 2, 20, 10, 0, tzinfo=STORE_TZ), tmp_path)
    assert "alerta" not in on_time.consultar_pedido(8, "anacarol.ferreira@coldmail.com")
    late = make_tools(datetime(2026, 3, 24, 10, 0, tzinfo=STORE_TZ), tmp_path)
    result = late.consultar_pedido(8, "anacarol.ferreira@coldmail.com")
    assert result["previsao_entrega"] == "2026-02-28"
    assert "alerta" in result


def test_delivered_order_reports_days_since_delivery_estimate(tmp_path):
    # Order 4: delivered, estimate 2025-12-22. A customer claiming "received two days ago"
    # must not be told the 7-day return window still applies without the team checking.
    recent = make_tools(datetime(2025, 12, 25, 10, 0, tzinfo=STORE_TZ), tmp_path)
    fresh = recent.consultar_pedido(4, "lucas.mendes@jmail.com")
    assert fresh["dias_desde_a_previsao_de_entrega"] == 3
    assert "alerta" not in fresh
    later = make_tools(datetime(2026, 11, 4, 10, 0, tzinfo=STORE_TZ), tmp_path)
    old = later.consultar_pedido(4, "lucas.mendes@jmail.com")
    assert old["dias_desde_a_previsao_de_entrega"] == 317
    assert "7 dias" in old["alerta"]


def test_cancelled_order_shows_reason(tools):
    result = tools.consultar_pedido(20, "thiago.barbosa@jmail.com")
    assert result["status"] == "cancelado"
    assert "Endereço" in result["observacao"]


# ----------------------------------------------------------------- store hours
def test_store_open_on_weekday_morning(tools):
    assert tools.status_da_loja()["aberta_agora"] is True


def test_store_status_includes_weekly_hours(tools):
    # "Vocês abrem sábado?" may be routed here instead of the policy: the answer must still be in the result
    hours = tools.status_da_loja()["horario_semanal"]
    assert hours["sábado"] == "09:00 às 13:00"
    assert hours["segunda a sexta"] == "09:00 às 18:00"
    assert hours["domingo e feriados"] == "fechado"


def test_store_closed_saturday_afternoon(tmp_path):
    t = make_tools(datetime(2026, 3, 28, 14, 0, tzinfo=STORE_TZ), tmp_path)
    status = t.status_da_loja()
    assert status["aberta_agora"] is False
    assert status["proxima_abertura"].startswith("segunda-feira 30/03")


def test_store_closed_on_holiday(tmp_path):
    t = make_tools(datetime(2026, 11, 2, 10, 0, tzinfo=STORE_TZ), tmp_path)
    status = t.status_da_loja()
    assert status["aberta_agora"] is False
    assert "Finados" in status["motivo_fechada"]


# ----------------------------------------------------------------- policies and escalation
def test_policy_sections_available(tools):
    assert "7 (sete) dias corridos" in tools.consultar_politica("trocas_e_devolucoes")["conteudo"]
    assert "erro" in tools.consultar_politica("nao_existe")


def test_escalation_is_logged(tools, tmp_path):
    out = tools.encaminhar_para_atendente("reclamacao", "Cliente relata atraso", "(67) 99999-0000")
    assert out["protocolo"].startswith("EM-20260324-")
    logged = (tmp_path / "esc.jsonl").read_text(encoding="utf-8")
    assert out["protocolo"] in logged


def test_same_case_is_not_escalated_twice(tools, tmp_path):
    first = tools.encaminhar_para_atendente("devolucao_ou_troca", "Pedido 4, arrependimento", "lucas.mendes@jmail.com")
    again = tools.encaminhar_para_atendente("devolucao_ou_troca", "Pedido 4, devolução", "lucas.mendes@jmail.com")
    assert again["protocolo"] == first["protocolo"] and again["ja_registrado"] is True
    other = tools.encaminhar_para_atendente("reclamacao", "Atendimento demorado", "lucas.mendes@jmail.com")
    assert other["protocolo"] != first["protocolo"]
    assert len((tmp_path / "esc.jsonl").read_text(encoding="utf-8").splitlines()) == 2


def test_dispatch_returns_json_errors_instead_of_raising(tools):
    out = json.loads(tools.call("detalhes_produto", {}))
    assert "erro" in out
    out = json.loads(tools.call("rm_rf", {}))
    assert "erro" in out
