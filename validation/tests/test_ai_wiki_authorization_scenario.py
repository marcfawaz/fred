# Copyright Thales 2026
#
# Licensed under the Apache License, Version 2.0 (the "License").

"""Offline guards for the AUTHZ-WIKI-07D live AI Wiki authorization scenario."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

from factory_config import TEST_TEAM, USERS


REPO_ROOT = Path(__file__).resolve().parents[2]
SCENARIO_PATH = REPO_ROOT / "validation" / "scenarios" / "test_ai_wiki_authorization.py"
CONTRACT_PATH = REPO_ROOT / "validation" / "ai_wiki_authz_campaign_contract.json"
AI_WIKI_REPO_CONTRACT_PATH = REPO_ROOT.parent / "fred-knowledge-wiki" / "scripts" / "ai_wiki_authz_campaign_contract.json"


def _load_scenario() -> ModuleType:
    spec = importlib.util.spec_from_file_location("ai_wiki_authorization_live_scenario_module", SCENARIO_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


scenario = _load_scenario()
contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def test_live_scenario_references_documented_personas() -> None:
    assert scenario.DOCUMENTED_PERSONAS == (
        "marc",
        "bob",
        "elena",
        "zoe",
        "priya",
        "alice",
        "gabriel",
        "oscar",
    )


def test_live_scenario_uses_fredlab() -> None:
    assert TEST_TEAM == "fredlab"
    assert contract["team_name"] == "fredlab"


def test_live_scenario_uses_demo_provisioning_fixture_without_provisioning() -> None:
    assert USERS["marc"].relations_in("fredlab") == frozenset({"team_admin"})
    assert USERS["bob"].relations_in("fredlab") == frozenset({"team_editor"})
    assert USERS["elena"].relations_in("fredlab") == frozenset({"team_analyst"})
    assert USERS["zoe"].relations_in("fredlab") == frozenset({"team_member"})
    assert USERS["priya"].relations_in("fredlab") == frozenset({"team_admin", "team_editor", "team_analyst"})
    assert USERS["alice"].is_platform_admin
    assert not USERS["alice"].relations_in("fredlab")
    assert USERS["gabriel"].is_platform_observer
    assert not USERS["gabriel"].relations_in("fredlab")
    assert not USERS["oscar"].relations_in("fredlab")


def test_live_scenario_contains_all_nine_ai_wiki_capability_expectations() -> None:
    assert set(scenario.AI_WIKI_CAPABILITIES) == {
        "can_read_wikis",
        "can_contribute_wikis",
        "can_review_wiki_changes",
        "can_manage_wiki_schema",
        "can_manage_wiki_lifecycle",
        "can_manage_wiki_governance",
        "can_use_wiki_review_assistant",
        "can_run_wiki_guarded_auto_apply",
        "can_run_wiki_autonomous_apply",
    }
    assert set(scenario.EXPECTED_AI_WIKI_CAPABILITIES["priya"]) == set(scenario.AI_WIKI_CAPABILITIES)
    assert "can_contribute_wikis" not in scenario.EXPECTED_AI_WIKI_CAPABILITIES["marc"]


def test_live_scenario_is_opt_in_not_an_offline_network_test() -> None:
    marker = scenario.pytestmark
    kwargs = marker.mark.kwargs if hasattr(marker, "mark") else marker.kwargs

    assert kwargs["reason"].startswith("AI Wiki authorization campaign is live/openfga-only")
    assert scenario.RUN_AI_WIKI_AUTHZ_LIVE_ENV in kwargs["reason"]
    assert "AUTHORIZATION_MODE=openfga" in kwargs["reason"]


def test_live_scenario_never_writes_direct_capability_tuples() -> None:
    source = SCENARIO_PATH.read_text(encoding="utf-8")

    forbidden = (
        "write_tuple",
        "write_tuples",
        "/stores/{store_id}/write",
        "/stores/{store_id}/tuple",
        "OpenFGAClient",
        "can_contribute_wikis@",
    )
    offenders = [needle for needle in forbidden if needle in source]
    assert not offenders


def test_canonical_endpoint_contract_covers_every_capability_with_correct_routes() -> None:
    by_capability = {endpoint["capability"]: endpoint for endpoint in contract["endpoints"]}

    assert contract["version"] == "AUTHZ-WIKI-07D"
    assert set(by_capability) == set(contract["capabilities"])
    assert by_capability["can_manage_wiki_schema"]["path"] == "/wiki/v1/wikis/{wiki_id}/schema-proposals"
    assert by_capability["can_manage_wiki_schema"]["method"] == "GET"
    assert by_capability["can_contribute_wikis"]["path"] == "/wiki/v1/teams/{team_id}/permissions/can_contribute_wikis"
    assert by_capability["can_contribute_wikis"]["method"] == "GET"
    assert by_capability["can_contribute_wikis"]["mutating"] is False
    assert by_capability["can_manage_wiki_lifecycle"]["path"] == "/wiki/v1/wikis/{wiki_id}"
    assert by_capability["can_manage_wiki_lifecycle"]["method"] == "PATCH"
    assert all(endpoint["path"] != "/wiki/v1/wikis/{wiki_id}/schema" for endpoint in contract["endpoints"])
    assert all(endpoint["key"] != "lifecycle_create" for endpoint in contract["endpoints"])
    assert all("accepted_authorized_statuses" in endpoint for endpoint in contract["endpoints"])
    assert all("accepted_denied_statuses" in endpoint for endpoint in contract["endpoints"])
    assert all("shadow_legacy_expected_allowed" in endpoint for endpoint in contract["endpoints"])


def test_live_scenario_uses_the_canonical_contract() -> None:
    assert scenario.CONTRACT == contract
    assert tuple(endpoint["key"] for endpoint in scenario.REPRESENTATIVE_ENDPOINTS) == tuple(
        endpoint["key"] for endpoint in contract["endpoints"]
    )


def test_ai_wiki_and_fred_contract_copies_are_synchronized_when_repo_is_present() -> None:
    assert AI_WIKI_REPO_CONTRACT_PATH.is_file(), (
        f"AI Wiki contract not found at {AI_WIKI_REPO_CONTRACT_PATH}; "
        "set up the sibling fred-knowledge-wiki checkout before changing this contract."
    )

    ai_wiki_contract = json.loads(AI_WIKI_REPO_CONTRACT_PATH.read_text(encoding="utf-8"))
    assert ai_wiki_contract == contract


def test_probe_status_evaluator_rejects_ambiguous_denials() -> None:
    endpoint = next(item for item in contract["endpoints"] if item["key"] == "read_list")

    assert scenario.probe_status_passes(endpoint, expected_allowed=False, status=403)
    assert scenario.probe_status_passes(endpoint, expected_allowed=False, status=404)
    assert not scenario.probe_status_passes(endpoint, expected_allowed=False, status=409)
    assert not scenario.probe_status_passes(endpoint, expected_allowed=True, status=422)
    assert not scenario.probe_status_passes(endpoint, expected_allowed=False, status=422)
    assert not scenario.probe_status_passes(endpoint, expected_allowed=False, status=503)
    assert not scenario.probe_status_passes(endpoint, expected_allowed=True, status=401)


def test_probe_status_evaluator_allows_documented_business_state_only_for_allowed_probe() -> None:
    endpoint = {
        **next(item for item in contract["endpoints"] if item["key"] == "read_list"),
        "accepted_business_statuses": [409],
    }

    assert scenario.probe_status_passes(endpoint, expected_allowed=True, status=409)
    assert not scenario.probe_status_passes(endpoint, expected_allowed=False, status=409)


def test_review_automation_probes_accept_only_authorized_business_409() -> None:
    endpoints = {item["key"]: item for item in contract["endpoints"]}

    assert endpoints["review_assistant"]["body"] == {}
    assert "dry_run" not in endpoints["review_assistant"]["body"]
    for key in ("review_assistant", "guarded_auto_apply", "autonomous_apply"):
        endpoint = endpoints[key]
        assert endpoint["accepted_authorized_statuses"] == [200]
        assert endpoint["accepted_business_statuses"] == [409]
        assert 409 not in endpoint["accepted_denied_statuses"]
        assert scenario.probe_status_passes(endpoint, expected_allowed=True, status=409)
        assert not scenario.probe_status_passes(endpoint, expected_allowed=False, status=409)
        assert not scenario.probe_status_passes(endpoint, expected_allowed=True, status=422)
        assert not scenario.probe_status_passes(endpoint, expected_allowed=False, status=422)
        assert not scenario.probe_status_passes(endpoint, expected_allowed=True, status=401)
        assert not scenario.probe_status_passes(endpoint, expected_allowed=True, status=503)

    assert endpoints["guarded_auto_apply"]["body"]["dry_run"] is True
    assert endpoints["autonomous_apply"]["body"]["dry_run"] is True
