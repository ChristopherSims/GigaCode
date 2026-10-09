"""Shared discount policy for invoice pricing."""


def apply_discount(amount, discount):
    """Apply a fractional discount to an amount."""
    return amount * (1 - discount)
