from routes import checkout


def test_checkout():
    assert checkout("valid", [10, 20], 0)["total"] == 30
