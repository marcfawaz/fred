# Copyright Thales 2026
#
# Licensed under the Apache License, Version 2.0 (the "License").

"""Offline guards for the validation report grouping UX."""

from __future__ import annotations

import generate_report


def test_ai_wiki_authorization_scenario_has_readable_report_group() -> None:
    assert (
        generate_report.GROUP_LABELS["test_ai_wiki_authorization.py"]
        == "AI Wiki authorization (AUTHZ-WIKI-07D)"
    )
    assert "AI Wiki authorization (AUTHZ-WIKI-07D)" in generate_report.GROUP_ORDER
