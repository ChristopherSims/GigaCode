"""Multi-language context navigation must be deterministic and evidence-backed."""

import json

import pytest

from gigacode.chunker import _get_grammar
from gigacode.navigation_index import NavigationIndex, parse_file


@pytest.fixture
def index(tmp_path):
    return NavigationIndex(tmp_path)


def test_python_symbols_scopes_import_aliases_and_calls(index):
    index.update({
        "app/billing.py": '"""Billing operations."""\n__all__ = ["create_invoice"]\n'
                          "def create_invoice(user):\n    return user\n",
        "app/routes.py": "from .billing import create_invoice as make_invoice\n"
                         "class Routes:\n"
                         "    def create(self, user):\n        return make_invoice(user)\n"
                         "    def forward(self, user):\n        return self.create(user)\n",
    })
    invoice = index.select("create_invoice", "app/billing.py")[0]
    create = index.select("Routes.create")[0]
    assert invoice.exported and create.parent == "app/routes.py::Routes"
    callees = index.relations(create, "callees")["items"]
    assert callees[0]["resolution"] == "resolved"
    assert callees[0]["symbols"][0]["id"] == invoice.id
    callers = index.relations(invoice, "callers")["items"]
    assert callers[0]["symbol"] == "Routes.create" and callers[0]["call_line"] == 4
    assert index.relations(create, "callers")["items"][0]["symbol"] == "Routes.forward"
    assert index.dependency_rows("app/routes.py")[0]["files"] == ["app/billing.py"]


def test_duplicate_and_qualified_lookup_never_guesses(index):
    index.update({"one.py": "def same():\n    pass\n", "two.py": "def same():\n    pass\n"})
    assert len(index.select("same")) == 2
    assert index.select("Unknown.same") == []
    assert index.select("one.same")[0].file == "one.py"


def test_nested_calls_belong_to_nested_symbol_and_shadowing_is_unresolved(index):
    index.update({"code.py": "def target():\n    pass\n"
                  "def outer():\n    def inner():\n        return target()\n    return inner()\n"
                  "def shadow(target):\n    return target()\n"})
    outer = index.select("outer")[0]
    inner = index.select("outer.inner")[0]
    assert [c["target"] for c in outer.calls] == ["inner"]
    assert [c["target"] for c in inner.calls] == ["target"]
    assert index.relations(index.select("shadow")[0], "callees")["items"][0]["resolution"] == "unresolved"
    assert {c["symbol"] for c in index.relations(index.select("target")[0], "callers")["items"]} == {"outer.inner"}


def test_scoped_import_does_not_leak_into_another_function(index):
    index.update({
        "helper.py": "def foo():\n    pass\n",
        "code.py": "def one():\n    from helper import foo\n    foo()\n"
                   "def two():\n    foo()\n",
    })
    assert index.relations(index.select("one")[0], "callees")["items"][0]["resolution"] == "resolved"
    assert index.relations(index.select("two")[0], "callees")["items"][0]["resolution"] == "unresolved"


@pytest.mark.parametrize("language", ["typescript", "tsx"])
def test_typescript_grammar_entrypoints_load(language):
    assert _get_grammar(language) is not None


@pytest.mark.parametrize("extension", ["js", "ts"])
def test_js_ts_exported_symbols_and_alias_imports(index, extension):
    index.update({
        f"billing.{extension}": "export function createInvoice(user) { return user; }\n",
        f"routes.{extension}": 'import {createInvoice as make} from "./billing";\n'
                               "export const submit = (user) => make(user);\n"
                               "export class Routes { run(user) { return make(user); } }\n",
    })
    submit = index.select("submit")[0]
    invoice = index.select("createInvoice")[0]
    assert submit.exported and invoice.exported
    assert index.select("Routes.run")[0].parent == f"routes.{extension}::Routes"
    result = index.relations(submit, "callees")["items"][0]
    assert result["resolution"] == "resolved" and result["symbols"][0]["id"] == invoice.id
    assert index.dependency_rows(f"routes.{extension}")[0]["files"] == [f"billing.{extension}"]


def test_tsx_arrow_component_and_interface_children(index):
    index.update({"ui.tsx": "export interface Auth { login(): string; }\n"
                  'export const Login = () => <button>Login</button>;\n'})
    assert index.select("Login")[0].symbol_type == "function"
    assert index.select("Auth.login")[0].parent == "ui.tsx::Auth"
    assert not index.files["ui.tsx"].diagnostics


def test_cpp_includes_namespace_and_overloads(index):
    index.update({
        "api.hpp": "namespace auth { int login(int x); int login(double x); }\n",
        "main.cpp": '#include "api.hpp"\nint run() { return auth::login(1); }\n',
    })
    overloads = index.select("auth.login", "api.hpp")
    assert len(overloads) == 2 and overloads[0].id != overloads[1].id
    result = index.relations(index.select("run")[0], "callees")["items"][0]
    assert result["resolution"] == "ambiguous" and len(result["symbols"]) == 2
    assert index.dependency_rows("main.cpp")[0]["files"] == ["api.hpp"]


def test_yaml_keys_aliases_and_no_scalar_value_leak(index):
    index.update({"values.yaml": "auth:\n  provider: jwt\n  password: PRIVATE_SENTINEL\n"
                  "replicaCount: 2\ncopy: &shared\n  nested: 1\nagain: *shared\n"})
    assert index.select("auth.provider", "values.yaml")[0].parent == "values.yaml::auth"
    alias = index.select("again")[0]
    assert alias.line_end >= alias.line_start
    assert "PRIVATE_SENTINEL" not in json.dumps(index.summary("values.yaml"))


def test_helm_definitions_values_templates_and_chart_dependencies(index):
    index.update({
        "chart/values.yaml": "auth:\n  enabled: true\n",
        "chart/templates/_helpers.tpl": '{{- define "app.name" -}}\nhello\n{{- end -}}\n',
        "chart/templates/deployment.yaml": '{{ include "app.name" . }}\nvalue: {{ .Values.auth.enabled }}\n',
        "chart/Chart.yaml": "name: app\ndependencies:\n  - name: dependency\n    repository: file://../dependency\n",
        "dependency/Chart.yaml": "name: dependency\n",
    })
    helper = index.select("app.name")[0]
    assert helper.symbol_type == "template" and helper.line_end == 3
    edges = index.dependency_rows("chart/templates/deployment.yaml")
    assert edges[0]["files"] == ["chart/templates/_helpers.tpl"]
    assert edges[1]["files"] == ["chart/values.yaml"]
    assert index.dependency_rows("chart/Chart.yaml")[0]["files"] == ["dependency/Chart.yaml"]


def test_docker_stages_and_compose_dependencies(index):
    index.update({
        "Dockerfile": "FROM python:3.12 AS build\nRUN echo hi\nFROM build AS final\n"
                      "COPY --from=build /app /app\n",
        "compose.yaml": "services:\n  api:\n    depends_on:\n      - db\n  db:\n    image: postgres\n",
    })
    assert index.select("build")[0].line_end == 2
    edges = index.dependency_rows("Dockerfile")
    assert edges[0]["resolution"] == "external"
    assert edges[1]["files"] == ["Dockerfile"]
    assert edges[2]["kind"] == "docker_stage"
    assert index.dependency_rows("compose.yaml")[0]["resolution"] == "resolved"


def test_api_clients_resolve_to_declared_routes_and_urls_are_sanitized(index):
    index.update({
        "api.py": "from fastapi import APIRouter\nrouter = APIRouter(prefix='/auth')\n"
                  "@router.get('/login')\ndef login():\n    return True\n",
        "ui.ts": "export function login() { return fetch('/auth/login'); }\n"
                 "function external() { return fetch('https://user:password@example.com/token?secret=hidden'); }\n",
    })
    edges = index.dependency_rows("ui.ts")
    assert edges[0]["files"] == ["api.py"] and edges[0]["kind"] == "api"
    assert edges[1]["target"] == "https://example.com"
    assert "password" not in json.dumps(edges) and "hidden" not in json.dumps(edges)


def test_incremental_update_and_deletion_refresh(index):
    index.update({"one.py": "def one():\n    pass\n", "two.py": "def two():\n    pass\n"})
    count = index.parse_count
    revision = index.revision
    index.update(dict(index.sources))
    assert index.parse_count == count and index.revision == revision
    index.update({"one.py": "def renamed():\n    pass\n", "two.py": index.sources["two.py"]})
    assert index.parse_count == count + 1 and index.revision != revision
    assert index.select("one") == [] and index.select("renamed")
    index.update({"one.py": index.sources["one.py"]})
    assert index.select("two") == []


def test_parse_errors_are_explicit_and_do_not_echo_values():
    parsed = parse_file("bad.yaml", "secret: [PRIVATE_SENTINEL\n")
    assert parsed.diagnostics and "PRIVATE_SENTINEL" not in json.dumps(parsed.diagnostics)
    assert parse_file("bad.py", "def missing(:").diagnostics


def test_refresh_ignores_caches_and_outside_symlinks(tmp_path):
    (tmp_path / "code.py").write_text("def one():\n    pass\n")
    cache = tmp_path / ".ai"
    cache.mkdir()
    (cache / "summary.py").write_text("def should_not_index():\n    pass\n")
    index = NavigationIndex(tmp_path)
    index.refresh()
    assert ".ai/summary.py" not in index.files
    assert not index.select("should_not_index")


def test_relation_pagination_retains_full_total(index):
    functions = "def target():\n    pass\n" + "".join(
        f"def caller_{number}():\n    target()\n" for number in range(105)
    )
    index.update({"code.py": functions})
    result = index.relations(index.select("target")[0], "callers", 10, 100)
    assert result["total"] == 105 and len(result["items"]) == 5
    assert result["next_offset"] is None


def test_oversized_pending_overlay_does_not_return_stale_disk_symbols(tmp_path, monkeypatch):
    import gigacode.navigation_index as navigation

    (tmp_path / "code.py").write_text("def old():\n    pass\n")
    monkeypatch.setattr(navigation, "MAX_FILE_BYTES", 50)
    index = NavigationIndex(tmp_path)
    index.refresh({"code.py": ["# " + "x" * 100]})
    assert index.select("old") == [] and index.diagnostics


def test_summary_cache_freshness_and_task_context_are_bounded(index, tmp_path):
    index.update({
        "auth.py": '"""JWT authentication."""\nfrom fastapi import APIRouter\n'
                   "class AuthService:\n    def login(self):\n        return True\n",
        "tests/test_auth.py": "from auth import AuthService\n",
    })
    cache = tmp_path / "cache"
    index.save_summaries(cache)
    original = (cache / "navigation_summaries.json").read_text()
    context = index.task_context("Add OAuth authentication to the API", 2)
    assert context["likely_files"][0]["file"] == "auth.py"
    assert "tests/test_auth.py" in context["relevant_tests"]
    assert context["architecture_evidence"]["items"][0]["technology"] == "fastapi"
    assert context["existing_authentication"]["items"]
    index.update({"auth.py": "def oauth():\n    pass\n"})
    index.save_summaries(cache)
    assert (cache / "navigation_summaries.json").read_text() != original
