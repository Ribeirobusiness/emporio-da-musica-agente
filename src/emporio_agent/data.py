"""Data layer: loads the store CSVs once, cleans them and exposes typed lookups.

The agent never sees raw tables. It only reaches the data through the tool
functions in ``tools.py``, which read from the ``StoreData`` object built here.
"""

from __future__ import annotations

import json
import re
from decimal import ROUND_HALF_UP, Decimal
import unicodedata
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import pandas as pd

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def discounted_price(price: float, percent: float) -> float:
    """Price after a percentage discount, rounded half-up to the cent on the exact
    decimal value. Plain round() works on the binary float and drifts a cent:
    round(599.90 * 0.95, 2) == 569.90, while 569.905 rounds to 569.91."""
    value = Decimal(str(price)) * (Decimal(100) - Decimal(str(percent))) / Decimal(100)
    return float(value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))

# Customer-facing labels (the store talks to customers in Portuguese)
PRODUCT_STATUS_PT = {
    "active": "ativo",
    "discontinued": "descontinuado",
    "coming_soon": "em breve (pré-lançamento)",
}

ORDER_STATUS_PT = {
    "pending": "aguardando confirmação de pagamento",
    "confirmed": "pagamento confirmado, em separação",
    "shipped": "enviado",
    "delivered": "entregue",
    "cancelled": "cancelado",
}

PAYMENT_PT = {
    "pix": "PIX",
    "debit": "cartão de débito",
    "boleto": "boleto bancário",
}




def normalize_text(text: str) -> str:
    """Lowercase, strip accents and collapse spaces, for accent-insensitive search."""
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^a-z0-9]+", " ", text.lower())
    return text.strip()


def digits_only(text: str) -> str:
    return re.sub(r"\D", "", text or "")


def payment_label(code: str) -> str:
    """'credit_6x' -> 'cartão de crédito em 6x'."""
    if not isinstance(code, str):
        return "não informado"
    match = re.fullmatch(r"credit_(\d+)x", code)
    if match:
        return f"cartão de crédito em {match.group(1)}x"
    return PAYMENT_PT.get(code, code)


def _clean_optional(value):
    """Turn pandas NaN into None so tool outputs serialize as null."""
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    if isinstance(value, str) and not value.strip():
        return None
    return value


@dataclass
class StoreData:
    products: pd.DataFrame
    categories: pd.DataFrame
    customers: pd.DataFrame
    orders: pd.DataFrame
    order_items: pd.DataFrame
    promotions: pd.DataFrame

    # ---------- products ----------
    def product_by_id(self, product_id: int) -> dict | None:
        rows = self.products[self.products["product_id"] == int(product_id)]
        return None if rows.empty else rows.iloc[0].to_dict()

    # ---------- customers / orders ----------
    def order_by_id(self, order_id: int) -> dict | None:
        rows = self.orders[self.orders["order_id"] == int(order_id)]
        if rows.empty:
            return None
        return {k: _clean_optional(v) for k, v in rows.iloc[0].to_dict().items()}

    def customer_by_id(self, customer_id: int) -> dict | None:
        rows = self.customers[self.customers["customer_id"] == int(customer_id)]
        return None if rows.empty else rows.iloc[0].to_dict()

    def items_of_order(self, order_id: int) -> list[dict]:
        items = self.order_items[self.order_items["order_id"] == int(order_id)]
        result = []
        for _, item in items.iterrows():
            product = self.product_by_id(item["product_id"])
            result.append(
                {
                    "produto": product["name"] if product else f"produto #{item['product_id']}",
                    "quantidade": int(item["quantity"]),
                }
            )
        return result


def _load_products(data_dir: Path, categories: pd.DataFrame, promotions: pd.DataFrame) -> pd.DataFrame:
    products = pd.read_csv(data_dir / "products.csv")
    products["name"] = products["name"].str.strip()
    products["specs"] = products["specs"].apply(lambda s: json.loads(s) if isinstance(s, str) and s else {})
    products = products.merge(
        categories[["category_id", "name"]].rename(columns={"name": "category_name"}),
        on="category_id",
        how="left",
    )

    # Only promotions flagged active count. If a product ever had more than one
    # active promotion we keep the largest discount (policy: promotions do not stack).
    active = (
        promotions[promotions["is_active"] == 1]
        .sort_values("discount_percent", ascending=False)
        .drop_duplicates("product_id")
        [["product_id", "discount_percent", "description"]]
        .rename(columns={"discount_percent": "promo_percent", "description": "promo_label"})
    )
    products = products.merge(active, on="product_id", how="left")
    products["promo_price"] = products.apply(
        lambda r: discounted_price(r["price_brl"], r["promo_percent"])
        if pd.notna(r["promo_percent"])
        else None,
        axis=1,
    )

    # Sellable = active status and stock on hand. 'coming_soon' is never sellable,
    # even if the table shows units in stock.
    products["sellable"] = (products["status"] == "active") & (products["stock_quantity"] > 0)

    specs_text = products["specs"].apply(lambda d: " ".join(str(v) for v in d.values()))
    products["search_text"] = (
        products["name"] + " " + products["description"].fillna("") + " "
        + products["category_name"].fillna("") + " " + specs_text
    ).apply(normalize_text)
    products["name_norm"] = products["name"].apply(normalize_text)
    return products.sort_values("product_id").reset_index(drop=True)


def load_data(data_dir: Path | str = DEFAULT_DATA_DIR) -> StoreData:
    data_dir = Path(data_dir)
    categories = pd.read_csv(data_dir / "categories.csv")
    promotions = pd.read_csv(data_dir / "promotions.csv")
    customers = pd.read_csv(data_dir / "customers.csv")
    orders = pd.read_csv(data_dir / "orders.csv", dtype={"tracking_code": "string", "notes": "string"})
    order_items = pd.read_csv(data_dir / "order_items.csv")

    customers["email_norm"] = customers["email"].str.strip().str.lower()
    customers["phone_digits"] = customers["phone"].apply(digits_only)

    products = _load_products(data_dir, categories, promotions)
    return StoreData(products, categories, customers, orders, order_items, promotions)


def audit(data: StoreData) -> list[str]:
    """Data-quality checks. Returns human-readable findings (used by scripts/data_audit.py)."""
    findings: list[str] = []
    prices = data.products.set_index("product_id")["price_brl"]
    status = data.products.set_index("product_id")["status"]
    created = data.products.set_index("product_id")["created_at"]

    for _, order in data.orders.iterrows():
        items = data.order_items[data.order_items["order_id"] == order["order_id"]]
        expected = sum(prices.get(r.product_id, 0) * r.quantity for r in items.itertuples())
        if abs(expected - order["total_brl"]) > 0.01:
            findings.append(
                f"Order {order['order_id']}: total_brl={order['total_brl']} but items at list price add up to {expected:.2f}"
            )
        for r in items.itertuples():
            if status.get(r.product_id) == "coming_soon":
                findings.append(
                    f"Order {order['order_id']} ({order['status']}) contains product {r.product_id}, which is 'coming_soon'"
                )
            if str(created.get(r.product_id, "")) > str(order["order_date"]):
                findings.append(
                    f"Order {order['order_id']} dated {order['order_date']} contains product {r.product_id} created on {created.get(r.product_id)}"
                )

    dup_emails = data.customers[data.customers["email_norm"].duplicated(keep=False)]
    for email, group in dup_emails.groupby("email_norm"):
        findings.append(f"Customers {list(group['customer_id'])} share the same e-mail {email}")

    promo_ids = data.promotions["product_id"]
    orphan = set(promo_ids) - set(data.products["product_id"])
    if orphan:
        findings.append(f"Promotions reference unknown products: {sorted(orphan)}")

    out_of_stock_active = data.products[(data.products["status"] == "active") & (data.products["stock_quantity"] == 0)]
    for r in out_of_stock_active.itertuples():
        findings.append(f"Product {r.product_id} ({r.name}) is active but has zero stock")

    no_products = set(data.categories["category_id"]) - set(data.products["category_id"])
    for cid in sorted(no_products):
        name = data.categories.loc[data.categories["category_id"] == cid, "name"].iloc[0]
        findings.append(f"Category {cid} ({name}) has no products in the catalog")

    return findings
