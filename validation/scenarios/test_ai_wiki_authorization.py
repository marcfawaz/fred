# Copyright Thales 2026
#
# Licensed under the Apache License, Version 2.0 (the "License").

"""AUTHZ-WIKI-07D live AI Wiki authorization persona campaign.

This scenario is deliberately opt-in because it calls the live AI Wiki service
and creates/deletes a disposable team wiki. It validates backend enforcement
against the same Fred demo-provisioning personas verified by the rest of the
validation suite; it never seeds direct per-capability OpenFGA tuples.
"""

from __future__ import annotations

import base64
import json
import os
import uuid
from collections.abc import Callable, Iterator
from pathlib import Path

import httpx
import pytest

from factory_config import TEST_TEAM


RUN_AI_WIKI_AUTHZ_LIVE_ENV = "RUN_AI_WIKI_AUTHZ_LIVE"
WIKI_BASE_URL = os.getenv("WIKI_BASE_URL", "http://localhost:8030").rstrip("/")
CAMPAIGN_MARKER_PREFIX = "authz-campaign-"
SETUP_PERSONA = "priya"
AUTHORIZATION_MODE = os.getenv("AUTHORIZATION_MODE", "")
CONTRACT_PATH = Path(__file__).resolve().parents[1] / "ai_wiki_authz_campaign_contract.json"
CONTRACT = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))

AI_WIKI_CAPABILITIES = tuple(CONTRACT["capabilities"])
EXPECTED_AI_WIKI_CAPABILITIES: dict[str, tuple[str, ...]] = {
    persona: tuple(capabilities) for persona, capabilities in CONTRACT["personas"].items()
}
DOCUMENTED_PERSONAS = tuple(EXPECTED_AI_WIKI_CAPABILITIES)
REPRESENTATIVE_ENDPOINTS = tuple(CONTRACT["endpoints"])

pytestmark = pytest.mark.skipif(
    os.getenv(RUN_AI_WIKI_AUTHZ_LIVE_ENV) != "1" or AUTHORIZATION_MODE != "openfga",
    reason=f"AI Wiki authorization campaign is live/openfga-only; set {RUN_AI_WIKI_AUTHZ_LIVE_ENV}=1 and AUTHORIZATION_MODE=openfga",
)


def _jwt_sub(token: str) -> str:
    payload = token.split(".")[1]
    payload += "=" * (-len(payload) % 4)
    decoded = json.loads(base64.urlsafe_b64decode(payload.encode("ascii")).decode("utf-8"))
    sub = decoded.get("sub")
    assert isinstance(sub, str) and sub
    return sub


def _wiki_client(username: str, token_for: Callable[[str], str]) -> httpx.Client:
    return httpx.Client(
        base_url=WIKI_BASE_URL,
        headers={"Authorization": f"Bearer {token_for(username)}"},
        timeout=20.0,
    )


def _request(client: httpx.Client, method: str, path: str, body: dict | None = None) -> httpx.Response:
    return client.request(method, path, json=body)


def _render_body(body: dict | None, context: dict[str, str]) -> dict | None:
    if body is None:
        return None
    return {
        key: value.format(**context) if isinstance(value, str) else value
        for key, value in body.items()
    }


def probe_status_passes(endpoint: dict, *, expected_allowed: bool, status: int) -> bool:
    authorized = set(endpoint.get("accepted_authorized_statuses", CONTRACT["defaults"]["accepted_authorized_statuses"]))
    authorized.update(endpoint.get("accepted_business_statuses", CONTRACT["defaults"]["accepted_business_statuses"]))
    denied = set(endpoint.get("accepted_denied_statuses", CONTRACT["defaults"]["accepted_denied_statuses"]))
    if status in (401, 422, 503):
        return False
    if expected_allowed:
        return status in authorized
    return status in denied


@pytest.fixture(scope="module")
def fredlab_id(cp) -> str:
    resp = cp("alice").get("/teams/all")
    resp.raise_for_status()
    item = next((t for t in resp.json() if t.get("name") == TEST_TEAM), None)
    assert item is not None, f"{TEST_TEAM!r} not found in /teams/all"
    return str(item.get("id") or item.get("team_id") or item["name"])


@pytest.fixture(scope="module")
def disposable_wiki(fredlab_id: str, token_for) -> Iterator[dict[str, str]]:
    marker = f"{CAMPAIGN_MARKER_PREFIX}{uuid.uuid4().hex[:10]}"
    client = _wiki_client(SETUP_PERSONA, token_for)
    wiki_id: str | None = None
    try:
        create = client.post("/wiki/v1/wikis", json={"name": marker, "scope_type": "team", "team_id": fredlab_id})
        assert create.status_code == 201, f"create disposable wiki failed: {create.status_code} {create.text[:300]}"
        wiki_id = str(create.json()["id"])
        yield {"id": wiki_id, "name": marker, "marker": marker, "team_id": fredlab_id}
    finally:
        if wiki_id is not None:
            assert marker.startswith(CAMPAIGN_MARKER_PREFIX)
            delete = client.delete(f"/wiki/v1/wikis/{wiki_id}")
            assert delete.status_code in (200, 202, 204, 404), (
                f"cleanup of disposable wiki {wiki_id} failed: {delete.status_code} {delete.text[:300]}"
            )
            verify = client.get(f"/wiki/v1/wikis?scope_type=team&team_id={fredlab_id}")
            assert verify.status_code == 200, f"cleanup verification failed: {verify.status_code} {verify.text[:300]}"
            leftovers = [
                item
                for item in verify.json()
                if isinstance(item, dict) and str(item.get("name") or "").startswith(marker)
            ]
            assert not leftovers, f"cleanup left campaign team wikis behind: {leftovers}"
        client.close()


def test_permissions_endpoint_matches_exact_ai_wiki_persona_matrix(fredlab_id: str, token_for) -> None:
    for username, expected_caps in EXPECTED_AI_WIKI_CAPABILITIES.items():
        with _wiki_client(username, token_for) as client:
            resp = client.get(f"/wiki/v1/teams/{fredlab_id}/permissions")
        if expected_caps:
            assert resp.status_code == 200, f"{username}: {resp.status_code} {resp.text[:300]}"
            actual = set(resp.json().get("permissions") or []) & set(AI_WIKI_CAPABILITIES)
            assert actual == set(expected_caps), f"{username}: expected {sorted(expected_caps)}, got {sorted(actual)}"
        else:
            if resp.status_code == 200:
                leaked = set(resp.json().get("permissions") or []) & set(AI_WIKI_CAPABILITIES)
                assert not leaked, f"{username} leaked AI Wiki capabilities on {TEST_TEAM}: {sorted(leaked)}"
            else:
                assert resp.status_code in (403, 404)


def test_persona_tokens_are_keyed_by_jwt_subject_not_username(token_for) -> None:
    subjects = {username: _jwt_sub(token_for(username)) for username in DOCUMENTED_PERSONAS}

    assert len(set(subjects.values())) == len(DOCUMENTED_PERSONAS)
    assert all(subject != username for username, subject in subjects.items())


def test_representative_endpoint_matrix_matches_ai_wiki_capabilities(disposable_wiki: dict[str, str], token_for) -> None:
    context = {
        "team_id": disposable_wiki["team_id"],
        "wiki_id": disposable_wiki["id"],
        "marker": disposable_wiki["marker"],
    }
    for endpoint in REPRESENTATIVE_ENDPOINTS:
        for username in DOCUMENTED_PERSONAS:
            expected_allowed = username in endpoint["allowed"]
            path = endpoint["path"].format(**context)
            body = _render_body(endpoint.get("body"), context)
            with _wiki_client(username, token_for) as client:
                resp = _request(client, endpoint["method"], path, body)
            assert probe_status_passes(endpoint, expected_allowed=expected_allowed, status=resp.status_code), (
                f"{username} expected_allowed={expected_allowed} for {endpoint['key']} but got "
                f"{resp.status_code}: {resp.text[:300]}"
            )
