"""Tools the agent can call.

Design rule: the LLM decides *what* to look up and *how* to phrase the answer;
everything that must be exact (prices, stock, discounts, installment math,
order status, opening hours) is computed here, in plain Python, and returned
as JSON. This keeps the model from inventing numbers, which the store manual
explicitly forbids (section 7.1, "Informações precisas").
"""

from __future__ import annotations

import difflib
import json
import random
import string
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any, Callable
from zoneinfo import ZoneInfo

from .data import (
    ORDER_STATUS_PT,
    PRODUCT_STATUS_PT,
    StoreData,
    discounted_price,
    digits_only,
    normalize_text,
    payment_label,
)
from .policies import SECTION_KEYS

STORE_TZ = ZoneInfo("America/Campo_Grande")
PIX_DISCOUNT_PERCENT = 5
COMBINED_PAYMENT_MIN = 2000.00

# Section 3.1 of the manual. "Até 3x ... sem valor mínimo de parcela (exceto
# abaixo de R$ 50,00)" is read as: minimum installment of R$ 50 for 1x-3x.
INSTALLMENT_RULES = [(3, 50.00), (6, 80.00), (12, 100.00)]

# Fixed-date national holidays. Movable ones (Carnaval, Sexta-feira Santa,
# Corpus Christi) and municipal holidays are not covered: see README limitations.
FIXED_HOLIDAYS = {
    (1, 1): "Confraternização Universal",
    (4, 21): "Tiradentes",
    (5, 1): "Dia do Trabalho",
    (9, 7): "Independência do Brasil",
    (10, 12): "Nossa Senhora Aparecida",
    (11, 2): "Finados",
    (11, 15): "Proclamação da República",
    (11, 20): "Dia da Consciência Negra",
    (12, 25): "Natal",
}

OPENING_HOURS = {  # weekday() -> (open, close); Sunday closed
    0: (time(9), time(18)),
    1: (time(9), time(18)),
    2: (time(9), time(18)),
    3: (time(9), time(18)),
    4: (time(9), time(18)),
    5: (time(9), time(13)),
}

WEEKDAYS_PT = ["segunda-feira", "terça-feira", "quarta-feira", "quinta-feira", "sexta-feira", "sábado", "domingo"]


def brl(value: float) -> str:
    """1234.5 -> 'R$ 1.234,50'"""
    text = f"{value:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return f"R$ {text}"


def _token_in(token: str, text: str) -> bool:
    """Very small Portuguese plural handling: violões -> violao, guitarras -> guitarra."""
    variants = {token, token.rstrip("s")}
    if token.endswith("oes"):
        variants.add(token[:-3] + "ao")
    if token.endswith("es"):
        variants.add(token[:-2])
    words = text.split()
    if token.isdigit():
        return token in words
    return any(v in words or any(w.startswith(v) for w in words) for v in variants if len(v) >= 2)


class StoreTools:
    def __init__(
        self,
        data: StoreData,
        policy_sections: dict[str, str],
        now_fn: Callable[[], datetime] | None = None,
        escalation_log: Path | str = Path("logs") / "encaminhamentos.jsonl",
    ):
        self.data = data
        self.sections = policy_sections
        self.now_fn = now_fn or (lambda: datetime.now(STORE_TZ))
        self.escalation_log = Path(escalation_log)

    # ------------------------------------------------------------------ catalog
    def _product_view(self, row: dict, detailed: bool = False) -> dict:
        price = float(row["price_brl"])
        promo_price = row.get("promo_price")
        has_promo = promo_price is not None and promo_price == promo_price  # not NaN
        if row["status"] != "active":
            availability = PRODUCT_STATUS_PT.get(row["status"], row["status"])
        elif row["stock_quantity"] > 0:
            availability = "disponível"
        else:
            availability = "esgotado no momento"
        view = {
            "id": int(row["product_id"]),
            "nome": row["name"],
            "categoria": row["category_name"],
            "preco_tabela": brl(price),
            "disponibilidade": availability,
            "unidades_em_estoque": int(row["stock_quantity"]) if row["status"] == "active" else 0,
        }
        if has_promo:
            view["promocao_ativa"] = {
                "nome": row["promo_label"],
                "desconto_percentual": int(row["promo_percent"]),
                "preco_promocional": brl(float(promo_price)),
            }
            view["preco_no_pix"] = brl(float(promo_price)) + " (PIX não acumula com promoção)"
        else:
            view["preco_no_pix"] = brl(discounted_price(price, PIX_DISCOUNT_PERCENT)) + " (5% de desconto no PIX)"
        if detailed:
            view["descricao"] = row["description"]
            view["especificacoes"] = row["specs"]
            ended = self.data.promotions[
                (self.data.promotions["product_id"] == row["product_id"]) & (self.data.promotions["is_active"] == 0)
            ]["description"].tolist()
            if ended:
                view["promocoes_encerradas"] = sorted(set(ended))
        return view

    def buscar_produtos(
        self,
        termo: str | None = None,
        categoria: str | None = None,
        preco_min: float | None = None,
        preco_max: float | None = None,
        apenas_disponiveis: bool = True,
        limite: int = 8,
    ) -> dict:
        df = self.data.products.copy()
        if categoria:
            cat = normalize_text(categoria)
            df = df[df["category_name"].apply(normalize_text).apply(lambda c: _token_in(cat, c) or cat in c)]
        if termo:
            tokens = [t for t in normalize_text(termo).split() if len(t) >= 2 or t.isdigit()]
            # A word that names a category ("violões", "guitarra") becomes a category
            # filter, so descriptions that merely mention it ("uma ponte entre o
            # ukulele e o violão") do not leak into the results.
            category_names = df["category_name"].apply(normalize_text)
            for token in list(tokens):
                hits = category_names.apply(lambda c: _token_in(token, c.split()[0] if c else ""))
                if hits.any():
                    df = df[hits]
                    category_names = category_names[hits]
                    tokens.remove(token)
            if tokens:
                df = df[df["search_text"].apply(lambda text: all(_token_in(t, text) for t in tokens))]
        # Price filters use what the customer actually pays (promo price when there is one).
        df = df.assign(_eff=df["promo_price"].fillna(df["price_brl"]))
        if preco_min is not None:
            df = df[df["_eff"] >= float(preco_min)]
        if preco_max is not None:
            df = df[df["_eff"] <= float(preco_max)]
        total_matches = len(df)
        unavailable = df[~df["sellable"]]
        if apenas_disponiveis:
            df = df[df["sellable"]]
        df = df.sort_values("_eff")
        limite = max(1, min(int(limite), 20))
        result = {
            "total_encontrados": total_matches,
            "total_disponiveis": int(df["sellable"].sum()),
            "produtos": [self._product_view(r) for r in df.head(limite).to_dict("records")],
        }
        if len(df) > limite:
            result["lista_parcial"] = True
            result["observacao"] = (
                f"Mostrando os {limite} mais baratos de {len(df)}. Há mais {len(df) - limite} opção(ões) "
                "que não estão nesta lista: diga ao cliente que são algumas das opções e ofereça mostrar as demais "
                "(chame de novo com limite maior)."
            )
        if apenas_disponiveis and len(unavailable):
            result["indisponiveis_omitidos"] = [
                {"nome": r["name"], "situacao": self._product_view(r)["disponibilidade"]}
                for r in unavailable.to_dict("records")
            ]
        if not result["produtos"] and termo:
            result["sugestoes_por_nome"] = self._fuzzy_names(termo)
        return result

    def _fuzzy_names(self, query: str, n: int = 3) -> list[str]:
        names = self.data.products["name"].tolist()
        norm = {normalize_text(x): x for x in names}
        hits = difflib.get_close_matches(normalize_text(query), list(norm), n=n, cutoff=0.35)
        return [norm[h] for h in hits]

    def _resolve_product(self, produto: str | int) -> tuple[dict | None, list[str]]:
        text = str(produto).strip()
        if text.isdigit():
            row = self.data.product_by_id(int(text))
            if row:
                return row, []
        query = normalize_text(text)
        tokens = query.split()
        products = self.data.products
        contains = products[products["name_norm"].apply(lambda n: all(_token_in(t, n) for t in tokens))]
        if len(contains) == 1:
            return contains.iloc[0].to_dict(), []
        if len(contains) > 1:
            return None, contains["name"].tolist()[:6]
        candidates = self._fuzzy_names(text, n=4)
        if len(candidates) == 1:
            row = products[products["name"] == candidates[0]].iloc[0].to_dict()
            return row, []
        return None, candidates

    def detalhes_produto(self, produto: str) -> dict:
        row, candidates = self._resolve_product(produto)
        if row is None:
            if candidates:
                return {"encontrado": False, "possiveis_produtos": candidates,
                        "instrucao": "Peça ao cliente para confirmar qual destes."}
            return {"encontrado": False, "mensagem": "Produto não encontrado no catálogo."}
        return {"encontrado": True, **self._product_view(row, detailed=True)}

    def listar_promocoes_ativas(self) -> dict:
        df = self.data.products[self.data.products["promo_percent"].notna()]
        return {"promocoes_ativas": [self._product_view(r) for r in df.to_dict("records")]}

    # ------------------------------------------------------------------ payment
    @staticmethod
    def _max_installments(total: float) -> tuple[int, float]:
        best = 1
        for n in range(1, 13):
            min_installment = next(m for limit, m in INSTALLMENT_RULES if n <= limit)
            if total / n >= min_installment:
                best = n
        return best, round(total / best, 2)

    def simular_pagamento(self, itens: list[dict]) -> dict:
        lines, total, total_pix = [], 0.0, 0.0
        for item in itens:
            row = self.data.product_by_id(int(item["produto_id"]))
            qty = int(item.get("quantidade", 1))
            if row is None:
                return {"erro": f"Produto {item['produto_id']} não existe."}
            if not row["sellable"]:
                return {"erro": f"{row['name']} não está disponível para venda ({self._product_view(row)['disponibilidade']})."}
            if qty > row["stock_quantity"]:
                return {"erro": f"Só há {row['stock_quantity']} unidade(s) de {row['name']} em estoque."}
            promo = row.get("promo_price")
            has_promo = promo is not None and promo == promo
            unit = float(promo) if has_promo else float(row["price_brl"])
            unit_pix = unit if has_promo else discounted_price(unit, PIX_DISCOUNT_PERCENT)
            total += unit * qty
            total_pix += unit_pix * qty
            lines.append({"produto": row["name"], "quantidade": qty, "preco_unitario": brl(unit),
                          "em_promocao": bool(has_promo)})
        n, value = self._max_installments(total)
        return {
            "itens": lines,
            "total": brl(total),
            "pix": brl(round(total_pix, 2)),
            "debito_ou_boleto": brl(total) + " (à vista, boleto compensa em até 3 dias úteis)",
            "cartao_credito": f"até {n}x de {brl(value)} sem juros",
            "combinar_formas_de_pagamento": total > COMBINED_PAYMENT_MIN,
            "regra_aplicada": "Parcela mínima: R$ 50 até 3x, R$ 80 de 4x a 6x, R$ 100 de 7x a 12x. "
                              "PIX dá 5% só em itens sem promoção.",
        }

    # ------------------------------------------------------------------ orders
    def consultar_pedido(self, numero_pedido: int | str, email_ou_telefone: str) -> dict:
        denied = {
            "encontrado": False,
            "mensagem": "Não foi possível localizar um pedido com esse número e esse contato. "
                        "Peça ao cliente para conferir o número do pedido e o e-mail ou telefone usado na compra.",
        }
        number = digits_only(str(numero_pedido))
        if not number:
            return denied
        order = self.data.order_by_id(int(number))
        if order is None:
            return denied  # same answer as a mismatch, so nobody can probe which orders exist
        customer = self.data.customer_by_id(order["customer_id"])
        given = (email_ou_telefone or "").strip().lower()
        given_digits = digits_only(given)
        email_ok = "@" in given and given == customer["email_norm"]
        phone_ok = len(given_digits) >= 8 and customer["phone_digits"].endswith(given_digits[-8:])
        if not (email_ok or phone_ok):
            return denied

        today = self.now_fn().date()
        result = {
            "encontrado": True,
            "pedido": int(order["order_id"]),
            "primeiro_nome_cliente": customer["name"].split()[0],
            "data_do_pedido": order["order_date"],
            "status": ORDER_STATUS_PT.get(order["status"], order["status"]),
            "itens": self.data.items_of_order(order["order_id"]),
            "total": brl(float(order["total_brl"])),
            "forma_de_pagamento": payment_label(order["payment_method"]),
        }
        if order.get("tracking_code"):
            result["codigo_rastreio"] = order["tracking_code"]
        if order.get("estimated_delivery"):
            result["previsao_entrega"] = order["estimated_delivery"]
            if order["status"] == "shipped" and date.fromisoformat(order["estimated_delivery"]) < today:
                result["alerta"] = "Previsão de entrega já passou e o pedido ainda consta como enviado: " \
                                   "ofereça encaminhar para a equipe verificar com a transportadora."
            if order["status"] == "delivered":
                # The data has no actual receipt date, only the estimate. Python does the
                # date math so the model does not confirm a return window on the customer's word.
                days = (today - date.fromisoformat(order["estimated_delivery"])).days
                result["dias_desde_a_previsao_de_entrega"] = days
                if days > 7:
                    result["alerta"] = (
                        f"Se o cliente pedir devolução por arrependimento (7 dias) ou troca, NÃO diga que está no prazo: "
                        f"a previsão de entrega foi há {days} dias e o sistema não registra a data real de recebimento. "
                        "Explique isso, chame encaminhar_para_atendente agora (o contato usado na verificação serve) "
                        "e diga que a equipe vai confirmar a data de recebimento."
                    )
        if order.get("notes"):
            result["observacao"] = order["notes"]
        return result

    # ------------------------------------------------------------------ store hours
    def status_da_loja(self) -> dict:
        now = self.now_fn().astimezone(STORE_TZ)

        def closed_reason(day: date) -> str | None:
            if (day.month, day.day) in FIXED_HOLIDAYS:
                return f"feriado ({FIXED_HOLIDAYS[(day.month, day.day)]})"
            if day.weekday() not in OPENING_HOURS:
                return "domingo"
            return None

        reason = closed_reason(now.date())
        hours = OPENING_HOURS.get(now.weekday())
        is_open = reason is None and hours is not None and hours[0] <= now.time() < hours[1]

        next_open = None
        for offset in range(0, 8):
            day = now.date() + timedelta(days=offset)
            if closed_reason(day):
                continue
            open_t, close_t = OPENING_HOURS[day.weekday()]
            if offset == 0 and now.time() >= open_t:
                continue
            next_open = f"{WEEKDAYS_PT[day.weekday()]} {day.strftime('%d/%m')} às {open_t.strftime('%H:%M')}"
            break

        result = {
            "agora_em_campo_grande": now.strftime("%d/%m/%Y %H:%M") + f" ({WEEKDAYS_PT[now.weekday()]})",
            "aberta_agora": is_open,
        }
        if is_open:
            result["fecha_hoje_as"] = hours[1].strftime("%H:%M")
        else:
            if reason:
                result["motivo_fechada"] = reason
            result["proxima_abertura"] = next_open
        return result

    # ------------------------------------------------------------------ policies
    def consultar_politica(self, secao: str) -> dict:
        if secao not in self.sections:
            return {"erro": f"Seção inválida. Use uma de: {', '.join(self.sections)}"}
        return {"secao": secao, "conteudo": self.sections[secao]}

    # ------------------------------------------------------------------ escalation
    def encaminhar_para_atendente(self, motivo: str, resumo: str, contato_cliente: str | None = None) -> dict:
        now = self.now_fn()
        protocol = "EM-" + now.strftime("%Y%m%d") + "-" + "".join(random.choices(string.digits, k=4))
        record = {"protocolo": protocol, "criado_em": now.isoformat(), "motivo": motivo,
                  "resumo": resumo, "contato_cliente": contato_cliente}
        self.escalation_log.parent.mkdir(parents=True, exist_ok=True)
        with self.escalation_log.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        return {"protocolo": protocol, "prazo_retorno": "até 24 horas úteis",
                "registrado": True}

    # ------------------------------------------------------------------ dispatch
    def call(self, name: str, arguments: dict[str, Any]) -> str:
        handler = getattr(self, name, None)
        if name not in TOOL_NAMES or handler is None:
            return json.dumps({"erro": f"Ferramenta desconhecida: {name}"}, ensure_ascii=False)
        try:
            return json.dumps(handler(**arguments), ensure_ascii=False, default=str)
        except Exception as exc:  # the model gets the error and can recover
            return json.dumps({"erro": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False)


TOOL_SCHEMAS: list[dict] = [
    {
        "name": "buscar_produtos",
        "description": "Busca instrumentos no catálogo por termo livre, categoria e faixa de preço. "
                       "Use para perguntas do tipo 'quais violões até R$ 1000', 'tem ukulele tenor?'. "
                       "Os preços filtrados consideram o preço promocional quando houver.",
        "input_schema": {
            "type": "object",
            "properties": {
                "termo": {"type": "string", "description": "Palavras-chave: marca, modelo, tipo (ex.: 'nylon', 'elétrico', 'Yamaha')."},
                "categoria": {"type": "string", "description": "Nome da categoria (ex.: 'Violões', 'Guitarras', 'Ukuleles')."},
                "preco_min": {"type": "number"},
                "preco_max": {"type": "number"},
                "apenas_disponiveis": {"type": "boolean", "description": "Padrão true. Use false só se o cliente quiser ver itens esgotados."},
                "limite": {"type": "integer", "description": "Máximo de itens retornados (padrão 8)."},
            },
        },
    },
    {
        "name": "detalhes_produto",
        "description": "Preço, estoque, promoção vigente, promoções já encerradas e especificações de UM produto, "
                       "pelo nome (mesmo aproximado) ou id. Use antes de informar preço ou disponibilidade de um item.",
        "input_schema": {
            "type": "object",
            "properties": {"produto": {"type": "string", "description": "Nome ou id do produto, ex.: 'Takamine GD20'."}},
            "required": ["produto"],
        },
    },
    {
        "name": "listar_promocoes_ativas",
        "description": "Lista os produtos com promoção ativa hoje.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "simular_pagamento",
        "description": "Calcula o total, o valor no PIX e o parcelamento máximo sem juros para um ou mais produtos, "
                       "aplicando as regras de promoção e parcela mínima. Use sempre que o cliente perguntar "
                       "sobre parcelas, PIX ou valor total.",
        "input_schema": {
            "type": "object",
            "properties": {
                "itens": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"produto_id": {"type": "integer"}, "quantidade": {"type": "integer"}},
                        "required": ["produto_id"],
                    },
                }
            },
            "required": ["itens"],
        },
    },
    {
        "name": "consultar_pedido",
        "description": "Status de um pedido. Exige o número do pedido E o e-mail ou telefone usado na compra "
                       "(verificação de identidade, LGPD). Se o cliente não informou os dois, peça antes de chamar.",
        "input_schema": {
            "type": "object",
            "properties": {
                "numero_pedido": {"type": "string"},
                "email_ou_telefone": {"type": "string"},
            },
            "required": ["numero_pedido", "email_ou_telefone"],
        },
    },
    {
        "name": "status_da_loja",
        "description": "Data e hora atuais em Campo Grande, se a loja está aberta agora e a próxima abertura. "
                       "Use para 'vocês estão abertos?', 'atendem hoje?' e para saber se é fora do expediente.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "consultar_politica",
        "description": "Retorna uma seção do manual de políticas da loja. Use para regras (trocas, devolução, garantia, "
                       "frete, pagamento, horário, contato, escopo). Nunca responda regras de memória.",
        "input_schema": {
            "type": "object",
            "properties": {"secao": {"type": "string", "enum": list(SECTION_KEYS)}},
            "required": ["secao"],
        },
    },
    {
        "name": "encaminhar_para_atendente",
        "description": "Registra o caso para a equipe humana e gera protocolo (retorno em até 24h úteis). Use para "
                       "reclamações, defeitos, devoluções a efetivar, pedidos atrasados, dúvidas que o manual não "
                       "resolve, ou quando o cliente pedir um humano.",
        "input_schema": {
            "type": "object",
            "properties": {
                "motivo": {"type": "string", "enum": ["reclamacao", "devolucao_ou_troca", "pedido_atrasado",
                                                       "duvida_nao_coberta", "pedido_do_cliente"]},
                "resumo": {"type": "string", "description": "Resumo objetivo do caso para o atendente."},
                "contato_cliente": {"type": "string"},
            },
            "required": ["motivo", "resumo"],
        },
    },
]

TOOL_NAMES = {t["name"] for t in TOOL_SCHEMAS}
