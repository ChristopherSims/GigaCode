"""Token authentication and role authorization for the checkout API."""


class AuthService:
    def authenticate(self, token):
        """Return the user associated with a known token."""
        return {"valid": "alice"}.get(token)


def can_administer(role):
    """Return whether a role has administrator privileges."""
    return role == "admin"
