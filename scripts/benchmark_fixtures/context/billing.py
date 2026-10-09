"""Invoice calculation used by checkout routes."""

from pricing import apply_discount


def invoice_total(prices, discount):
    """Calculate the discounted invoice total."""
    return apply_discount(sum(prices), discount)
