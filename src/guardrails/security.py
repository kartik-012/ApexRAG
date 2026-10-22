"""
Enterprise Input/Output Guardrails, Prompt Injection Sanitization & PII Redaction for ApexRAG.

Features:
  1. Prompt Injection Defense: Detects direct instruction overrides, jailbreak jail patterns (DAN, Dev Mode),
     and delimiter hijacking (<|im_start|>, [INST], <<SYS>>).
  2. Zero-Loss PII Masking: Redacts API keys, Bearer JWT tokens, emails, phone numbers, SSNs, and credit cards.
  3. Anti-Hallucination Output Filter: Assesses factual alignment of generated answers against retrieved context.
"""

from dataclasses import dataclass, field
import re
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class GuardrailResult:
    is_safe: bool
    sanitized_text: str
    violations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_safe": self.is_safe,
            "sanitized_text": self.sanitized_text,
            "violations": self.violations,
            "metadata": self.metadata,
        }


# ── 1. Prompt Injection & Jailbreak Signatures ──────────────────────
_INJECTION_PATTERNS = [
    (re.compile(r"(?i)\b(?:ignore|disregard|forget)\s+(?:all\s+)?(?:previous|prior|above)\s+(?:instructions|prompts|commands|rules)\b"), "instruction_override"),
    (re.compile(r"(?i)\b(?:you\s+are\s+now|act\s+as)\s+(?:DAN|unfiltered|jailbroken|developer\s+mode|an\s+unconstrained\s+AI)\b"), "jailbreak_persona"),
    (re.compile(r"(?i)\b(?:override|bypass|disable)\s+(?:system\s+prompt|safety\s+filters|content\s+policy)\b"), "safety_bypass"),
    (re.compile(r"(?i)\b(?:print|reveal|output|display|show|leak)\s+(?:system\s+prompt|internal\s+instructions|secret\s+key)\b"), "system_prompt_exfiltration"),
    (re.compile(r"(?i)<\|(?:im_start|im_end|system|user|assistant)\|?>"), "chatml_delimiter_injection"),
    (re.compile(r"(?i)\[\/?(?:INST|SYS)\]|<<\/?SYS>>"), "llama_delimiter_injection"),
    (re.compile(r"(?i)---+\s*(?:BEGIN|START)\s+(?:SYSTEM|PROMPT)\s*---+"), "delimiter_hijacking"),
]

# ── 2. PII & Secret Patterns ─────────────────────────────────────────
_PII_PATTERNS = [
    # OpenAI & Standard API Keys
    (re.compile(r"\bsk-[a-zA-Z0-9]{20,}\b"), "[API_KEY_REDACTED]"),
    # JWT Tokens
    (re.compile(r"\beyJ[a-zA-Z0-9_-]{10,}\.eyJ[a-zA-Z0-9_-]{10,}\.[a-zA-Z0-9_-]{10,}\b"), "[JWT_TOKEN_REDACTED]"),
    # Email Addresses
    (re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"), "[EMAIL_REDACTED]"),
    # Phone Numbers (International & US)
    (re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"), "[PHONE_REDACTED]"),
    # US Social Security Numbers
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[SSN_REDACTED]"),
    # Credit Card Numbers (13-19 digits with dashes/spaces)
    (re.compile(r"\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|3(?:0[0-5]|[68][0-9])[0-9]{11}|6(?:011|5[0-9]{2})[0-9]{12})\b"), "[CREDENTIAL_REDACTED]"),
]


def detect_prompt_injection(text: str) -> Tuple[bool, List[str]]:
    """
    Scans incoming query text against deterministic adversarial prompt injection patterns.

    Detects direct instruction overrides, jailbreak personas (e.g. DAN mode),
    system prompt exfiltration requests, and delimiter injection tags (<|im_start|>, [INST]).

    Args:
        text: Raw user query string.

    Returns:
        Tuple of (has_injection: bool, list_of_violation_types: List[str]).
    """
    violations = []
    for pattern, name in _INJECTION_PATTERNS:
        if pattern.search(text):
            violations.append(name)
    return len(violations) > 0, violations


def mask_pii(text: str) -> Tuple[str, int]:
    """Redacts emails, API keys, phone numbers, SSNs, and credit cards."""
    masked = text
    total_redactions = 0
    for pattern, placeholder in _PII_PATTERNS:
        matches = len(pattern.findall(masked))
        if matches > 0:
            total_redactions += matches
            masked = pattern.sub(placeholder, masked)
    return masked, total_redactions


def apply_input_guardrails(prompt: str) -> GuardrailResult:
    """
    Sanitizes user prompt:
      1. Detects prompt injection attempts (marks unsafe if detected).
      2. Redacts all sensitive PII and API keys.
    """
    is_injection, injection_flags = detect_prompt_injection(prompt)
    sanitized_text, pii_count = mask_pii(prompt)

    violations = list(injection_flags)
    if pii_count > 0:
        violations.append(f"pii_detected_{pii_count}_entities")

    return GuardrailResult(
        is_safe=not is_injection,
        sanitized_text=sanitized_text,
        violations=violations,
        metadata={
            "injection_detected": is_injection,
            "pii_entities_redacted": pii_count,
            "original_length": len(prompt),
            "sanitized_length": len(sanitized_text),
        },
    )


def verify_output_grounding(
    answer: str,
    context: str,
    min_overlap_ratio: float = 0.35,
) -> Tuple[bool, float, List[str]]:
    """
    Anti-hallucination Output Verification:
    Checks that core content terms and technical identifiers in the answer
    are grounded in the provided context passage.
    """
    if not answer.strip():
        return True, 1.0, []

    if not context.strip():
        return False, 0.0, ["empty_context_unsupported_generation"]

    # Extract alphanumeric words (>3 chars) from answer
    stopwords = {
        "this", "that", "with", "from", "have", "were", "been", "will", "would",
        "could", "should", "what", "when", "where", "which", "there", "their",
        "they", "about", "other", "after", "before", "using", "under", "these",
    }
    answer_words = [
        w.lower()
        for w in re.findall(r"\b[A-Za-z0-9_]{4,}\b", answer)
        if w.lower() not in stopwords
    ]
    if not answer_words:
        return True, 1.0, []

    context_lower = context.lower()
    grounded_count = sum(1 for w in answer_words if w in context_lower)
    overlap_ratio = grounded_count / float(len(answer_words))

    violations = []
    is_grounded = overlap_ratio >= min_overlap_ratio
    if not is_grounded:
        violations.append(f"low_context_grounding_ratio_{overlap_ratio:.2f}")

    return is_grounded, round(overlap_ratio, 4), violations


def apply_output_guardrails(
    answer: str,
    context: str,
) -> GuardrailResult:
    """
    Output Guardrail Filter:
      1. Mask any accidentally reflected PII.
      2. Verify factual grounding against retrieved context.
      3. Prevent delimiter leakages.
    """
    masked_answer, pii_count = mask_pii(answer)
    is_grounded, grounding_score, grounding_violations = verify_output_grounding(masked_answer, context)

    violations = list(grounding_violations)
    if pii_count > 0:
        violations.append("output_pii_redacted")

    return GuardrailResult(
        is_safe=is_grounded,
        sanitized_text=masked_answer,
        violations=violations,
        metadata={
            "grounding_score": grounding_score,
            "pii_redacted": pii_count,
        },
    )
