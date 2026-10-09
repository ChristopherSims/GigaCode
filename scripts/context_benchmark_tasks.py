"""Five fixed edits exercising source-backed navigation and context discovery."""

TASKS_CONTEXT = [
    {
        "id": "context_symbol_validation",
        "prompt": "In AuthService.authenticate, reject an empty token with ValueError before lookup. Preserve valid and unknown token behavior. Change only that method.",
        "check": {"file": "auth.py", "symbol": "authenticate"},
    },
    {
        "id": "context_callers_rounding",
        "prompt": "Trace the invoice total helper used by the checkout endpoint. Round its final total to two decimal places so checkout returns currency precision. Change only that helper, not its callers.",
        "check": {"file": "billing.py", "symbol": "invoice_total"},
    },
    {
        "id": "context_dependency_discount",
        "prompt": "Follow the pricing dependency used by invoice totals. Reject a discount outside 0..1 inclusive with ValueError, preserving valid calculations. Change only the discount helper.",
        "check": {"file": "pricing.py", "symbol": "apply_discount"},
    },
    {
        "id": "context_summary_doctest",
        "prompt": "Locate the public helper in the authentication module that reports whether a role can administer, and inspect its consumers/tests. Add a runnable doctest showing the admin role returns True. Change only its docstring.",
        "check": {"file": "auth.py", "symbol": "can_administer"},
    },
    {
        "id": "context_config_alignment",
        "prompt": "Inspect the service port declared in deployment values and the Python API base URL helper used by the frontend settings. Make the helper use the declared port instead of its outdated hardcoded port. Change only that Python helper; leave deployment and frontend files unchanged.",
        "check": {"file": "settings.py", "symbol": "api_base_url"},
    },
]

CONTEXT_BEHAVIOR = {
    "context_symbol_validation": """
f = ns['AuthService']().authenticate
assert f('valid') == 'alice'
assert f('unknown') is None
try: f('')
except ValueError: pass
else: raise AssertionError('empty token accepted')
""",
    "context_callers_rounding": """
f = ns['invoice_total']
assert f([0.1, 0.2], 0) == 0.3
assert f([19.99, 5.55], 0.15) == 21.71
assert f([], 0) == 0
""",
    "context_dependency_discount": """
f = ns['apply_discount']
assert f(100, 0) == 100 and f(100, 1) == 0
assert f(100, .25) == 75
for discount in (-.01, 1.01):
    try: f(100, discount)
    except ValueError: pass
    else: raise AssertionError('invalid discount accepted')
""",
    "context_config_alignment": """
assert ns['api_base_url']() == 'http://localhost:8080'
assert ns['api_base_url']('example.com') == 'http://example.com:8080'
""",
}
