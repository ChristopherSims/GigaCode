"""Checkout entry points and administrative authorization."""

from auth import AuthService, can_administer
from billing import invoice_total


def checkout(token, prices, discount):
    user = AuthService().authenticate(token)
    if user is None:
        raise PermissionError("Authentication required")
    return {"user": user, "total": invoice_total(prices, discount)}


def admin_allowed(role):
    return can_administer(role)
