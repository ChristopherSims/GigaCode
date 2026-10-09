from auth import AuthService, can_administer


def test_known_token():
    assert AuthService().authenticate("valid") == "alice"


def test_role_policy():
    assert can_administer("admin")
    assert not can_administer("viewer")
