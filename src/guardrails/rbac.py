"""
Multi-Tenant Role-Based Access Control (RBAC) & Document Security Pre-Filtering.

Enterprise Features:
  - Document-level Access Control Lists (ACL)
  - Multi-tenant tenant isolation ('tenant_id')
  - Role hierarchy ('public', 'developer', 'admin', 'superadmin')
  - Clearance level grading ('public', 'internal', 'confidential', 'restricted')
  - Pre-retrieval filtering before vector & BM25 ranking (prevents data leakage)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set


class ClearanceLevel(str, Enum):
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"


CLEARANCE_RANK: Dict[ClearanceLevel, int] = {
    ClearanceLevel.PUBLIC: 1,
    ClearanceLevel.INTERNAL: 2,
    ClearanceLevel.CONFIDENTIAL: 3,
    ClearanceLevel.RESTRICTED: 4,
}


@dataclass
class UserContext:
    """Security credentials and identity of the requesting user/service."""
    user_id: str = "anonymous"
    tenant_id: str = "public"
    roles: List[str] = field(default_factory=lambda: ["public"])
    clearance_level: ClearanceLevel = ClearanceLevel.PUBLIC

    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "tenant_id": self.tenant_id,
            "roles": self.roles,
            "clearance_level": self.clearance_level.value,
        }

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "UserContext":
        if not data:
            return cls()
        clearance_str = data.get("clearance_level", "public").lower()
        try:
            clearance = ClearanceLevel(clearance_str)
        except ValueError:
            clearance = ClearanceLevel.PUBLIC
        return cls(
            user_id=data.get("user_id", "anonymous"),
            tenant_id=data.get("tenant_id", "public"),
            roles=data.get("roles", ["public"]),
            clearance_level=clearance,
        )


def can_access_chunk(chunk_metadata: Dict[str, Any], user: UserContext) -> bool:
    """
    Evaluates whether the requesting UserContext is authorized to access a document chunk.
    Enforces:
      1. Superadmin bypass
      2. Tenant isolation (tenant_id match or chunk is public)
      3. Clearance level hierarchy (user clearance >= chunk clearance)
      4. Role-based ACL (user has at least one required role, or chunk has no role restrictions)
    """
    if "superadmin" in user.roles:
        return True

    # 1. Tenant Check
    chunk_tenant = chunk_metadata.get("tenant_id", "public")
    if chunk_tenant != "public" and chunk_tenant != user.tenant_id:
        return False

    # 2. Clearance Check
    chunk_clearance_str = chunk_metadata.get("clearance_level", "public").lower()
    try:
        chunk_clearance = ClearanceLevel(chunk_clearance_str)
    except ValueError:
        chunk_clearance = ClearanceLevel.PUBLIC

    user_rank = CLEARANCE_RANK.get(user.clearance_level, 1)
    chunk_rank = CLEARANCE_RANK.get(chunk_clearance, 1)
    if user_rank < chunk_rank:
        return False

    # 3. Role-Based ACL Check
    allowed_roles = chunk_metadata.get("allowed_roles")
    if allowed_roles:
        if isinstance(allowed_roles, str):
            allowed_roles = [allowed_roles]
        user_role_set: Set[str] = set(user.roles)
        if "admin" not in user_role_set and not any(r in user_role_set for r in allowed_roles):
            return False

    return True


def filter_authorized_chunks(
    chunk_ids: List[str],
    metadatas: Dict[str, Dict[str, Any]],
    user: UserContext,
) -> List[str]:
    """Filters chunk IDs returning only those the user is authorized to read."""
    authorized = []
    for cid in chunk_ids:
        meta = metadatas.get(cid, {})
        if can_access_chunk(meta, user):
            authorized.append(cid)
    return authorized
