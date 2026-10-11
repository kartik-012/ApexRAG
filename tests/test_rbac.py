"""
Unit and Integration Tests for Multi-Tenant Role-Based Access Control (RBAC).
"""

from src.guardrails.rbac import (
    UserContext,
    ClearanceLevel,
    can_access_chunk,
    filter_authorized_chunks,
)
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)


def test_tenant_isolation():
    user_eng = UserContext(user_id="u1", tenant_id="engineering", roles=["developer"])
    user_fin = UserContext(user_id="u2", tenant_id="finance", roles=["analyst"])
    user_super = UserContext(user_id="u3", tenant_id="public", roles=["superadmin"])

    chunk_eng = {"tenant_id": "engineering", "clearance_level": "public"}
    chunk_fin = {"tenant_id": "finance", "clearance_level": "public"}
    chunk_public = {"tenant_id": "public", "clearance_level": "public"}

    # Eng user can access Eng and Public, but not Finance
    assert can_access_chunk(chunk_eng, user_eng) is True
    assert can_access_chunk(chunk_public, user_eng) is True
    assert can_access_chunk(chunk_fin, user_eng) is False

    # Finance user can access Finance and Public, but not Eng
    assert can_access_chunk(chunk_fin, user_fin) is True
    assert can_access_chunk(chunk_eng, user_fin) is False

    # Superadmin can access any tenant
    assert can_access_chunk(chunk_fin, user_super) is True
    assert can_access_chunk(chunk_eng, user_super) is True


def test_clearance_hierarchy():
    user_public = UserContext(clearance_level=ClearanceLevel.PUBLIC)
    user_internal = UserContext(clearance_level=ClearanceLevel.INTERNAL)
    user_confidential = UserContext(clearance_level=ClearanceLevel.CONFIDENTIAL)

    chunk_confidential = {"clearance_level": "confidential"}
    chunk_internal = {"clearance_level": "internal"}
    chunk_public = {"clearance_level": "public"}

    # Public user only sees public
    assert can_access_chunk(chunk_public, user_public) is True
    assert can_access_chunk(chunk_internal, user_public) is False
    assert can_access_chunk(chunk_confidential, user_public) is False

    # Internal user sees public + internal, but not confidential
    assert can_access_chunk(chunk_public, user_internal) is True
    assert can_access_chunk(chunk_internal, user_internal) is True
    assert can_access_chunk(chunk_confidential, user_internal) is False

    # Confidential user sees all three
    assert can_access_chunk(chunk_confidential, user_confidential) is True
    assert can_access_chunk(chunk_internal, user_confidential) is True
    assert can_access_chunk(chunk_public, user_confidential) is True


def test_role_based_acl():
    user_dev = UserContext(roles=["developer"])
    user_admin = UserContext(roles=["admin"])
    user_intern = UserContext(roles=["intern"])

    chunk_dev_only = {"allowed_roles": ["developer", "tech_lead"]}

    assert can_access_chunk(chunk_dev_only, user_dev) is True
    assert can_access_chunk(chunk_dev_only, user_admin) is True  # Admin bypass
    assert can_access_chunk(chunk_dev_only, user_intern) is False


def test_filter_authorized_chunks():
    user = UserContext(tenant_id="engineering", clearance_level=ClearanceLevel.INTERNAL)
    metadatas = {
        "c1": {"tenant_id": "engineering", "clearance_level": "internal"},
        "c2": {"tenant_id": "finance", "clearance_level": "internal"},
        "c3": {"tenant_id": "engineering", "clearance_level": "confidential"},
        "c4": {"tenant_id": "public", "clearance_level": "public"},
    }
    chunk_ids = ["c1", "c2", "c3", "c4"]
    authorized = filter_authorized_chunks(chunk_ids, metadatas, user)
    assert authorized == ["c1", "c4"]


def test_api_query_with_rbac_telemetry():
    res = client.post(
        "/query",
        json={
            "question": "How to manage state in React?",
            "user_id": "alice",
            "tenant_id": "engineering",
            "roles": ["developer"],
            "clearance_level": "internal",
        },
    )
    assert res.status_code == 200
    data = res.json()
    assert "rbac_telemetry" in data
    assert data["rbac_telemetry"]["user_id"] == "alice"
    assert data["rbac_telemetry"]["tenant_id"] == "engineering"
    assert data["rbac_telemetry"]["authorized_count"] >= 1
