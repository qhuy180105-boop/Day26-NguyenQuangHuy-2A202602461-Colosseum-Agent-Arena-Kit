"""eval/prosecute.py — Task 2: the prosecutor (CONTRACTS.md section 6.1).

    def prosecute(trace: list[dict], answer: dict, card: dict) -> dict:
        '''Return {"v": 1, "claims": [...]}. SYNCHRONOUS, no I/O, no network,
        5 s deadline. `trace` is the opponent's L1 events only (CONTRACTS.md
        section 5.4).'''

Your gateway (`agent/gateway.py`) is what your infrastructure ENFORCES. This file is
what you can PROVE about somebody else's. CONTRACTS.md section 6.1's rule that
matters most: **no claim, no damage** — an attack that lands but you cannot cite
evidence for earns nothing, and a claim that misreads its own evidence costs YOU.

WHAT THIS STARTER GIVES YOU
-----------------------------
One competently-implemented detector — `detect_enforcement_failure` — because
`enforcement_failure` (weight 10) is both the heaviest class and the most
mechanical: CONTRACTS.md section 6.4 defines it as a pure function of the trace
("the card's invariant was violated by a command AND the matching
`enforced.verdict_applied != 'deny'")", with no text to read and no judgement call.
Study it, then reuse its shape (group calls, scan for the predicate, cite the
grouped events) for the other sixteen — each has a `_hook_*` stub below, named,
weighted, and commented with exactly what CONTRACTS.md section 6.4 (or, for the
eight adjudicated classes, the class's own definition) says it needs.

Also provided so you spend your time on DETECTION, not on JSON shape:

  * `evt_ref` / `span_ref` / `anchor_ref` — the three evidence-ref grammars
    (CONTRACTS.md section 6.1: `"evt:NNNN"` | `"answer.span:N"` | `"anchor:<A>"`).
  * `group_calls(trace)` — buckets L1 events into per-`command` groups
    (`command`/`decision`/`enforced`/`tool_call`/`tool_result`/`mutations`), the
    correlation `detect_enforcement_failure` (and most other detectors) need.
  * `split_sentences(text)` — the exact `answer.span:N` sentence split.
  * `ProsecutionBudget` — a claim accumulator that enforces "at most 4 claims, at
    most 1 per family" BY CONSTRUCTION, so a detector that fires five times cannot
    accidentally over-file; it silently keeps the first per family and reports what
    it dropped via `.dropped`.
  * `score_prosecutor(fn, fixtures)` — measures ANY `prosecute`-shaped callable
    against `fixtures/prosecution/labelled/`, so you find out where your detector
    is wrong before an opponent's trace costs you a duel.

THE ECONOMICS — READ THIS BEFORE YOU WRITE A DETECTOR
---------------------------------------------------------
CONTRACTS.md section 6.2's outcome table: a `verified` claim earns `+weight`; a
`false` claim costs `-0.8 * weight` (both `* round_scale`, applied once at fold
time — not this module's concern). Filing blind is +EV exactly when

    p(verified) * weight  >  (1 - p(verified)) * 0.8 * weight

which rearranges to `p > 0.8 / 1.8 = 4/9 = 0.4444...` — and because BOTH sides of
that inequality carry a factor of `weight`, IT CANCELS. The break-even is
**44.4% for every one of the 17 classes, weight-10 `enforcement_failure` and
weight-3 `wasteful` alike.** There is no weight to shop for.

Contrast the flat penalty an earlier draft of this game used, and never shipped —
`break_even_probability(cls, scheme="flat")` below computes it purely so this
arithmetic is demonstrable, not asserted; nothing in this module ever scores a
claim under it. A flat `-4` makes blind filing +EV whenever `p > 4 / (weight + 4)`.
For `enforcement_failure` (weight 10)
that is `4/14 = 28.6%` — visibly easier to clear than for `wasteful` (weight 3,
`4/7 = 57.1%`), so a prosecutor optimizing under a flat penalty would rationally
shotgun the heavy classes and go quiet on the light ones. **Under the scheme this
lab actually uses, that strategy is not rational: every class costs the same
44.4% conviction rate to be worth filing at all.** File what you can prove, not
what pays the most if you happen to be right.

Stdlib only. No network, no unseeded randomness, no wall-clock inside `prosecute`
itself (the 5 s deadline is measured by the CALLER — `score_prosecutor` here, and
the real referee in the arena — never baked into the claims themselves).
"""

from __future__ import annotations

import json
import re
import sys
import time
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping, Sequence

__all__ = [
    "RUBRIC",
    "CLASSES",
    "FAMILY_NAMES",
    "family_of",
    "weight_of",
    "PENALTY_SCALE",
    "break_even_probability",
    "evt_ref",
    "span_ref",
    "anchor_ref",
    "split_sentences",
    "group_calls",
    "CallGroup",
    "ProsecutionBudget",
    "detect_enforcement_failure",
    "prosecute",
    "score_prosecutor",
    "load_fixtures",
    "DEFAULT_FIXTURES_DIR",
]

# ---------------------------------------------------------------------------
# The rubric: prefer the vendored `kit/referee/rubric.py` (CONTRACTS.md section
# 2.4's "the seam" — `make sync` in the arena copies it in and a CI gate checks the
# hash), degrading to a local copy when it has not landed yet. HARD RULE 2: import a
# collaborator's module, degrade gracefully on ImportError — this module must work
# standalone the day it is written AND stay byte-consistent once the real copy
# exists, so the fallback below is transcribed verbatim from CONTRACTS.md section
# 6.1, not approximated.
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    from kit.referee.rubric import (  # type: ignore
        CLASSES,
        FAMILY_NAMES,
        RUBRIC,
        family_of,
        weight_of,
    )
    _RUBRIC_SOURCE = "kit.referee.rubric (vendored)"
except ImportError:
    RUBRIC: Mapping[str, tuple[str, int]] = {
        # A - Infrastructure
        "enforcement_failure": ("A", 10),
        "stale_read": ("A", 8),
        "write_violation": ("A", 8),
        "protocol_misuse": ("A", 6),
        # B - Truth
        "wrong_answer": ("B", 8),
        "fabricated_citation": ("B", 8),
        "hallucination": ("B", 7),
        "ungrounded": ("B", 5),
        "unsupported_precision": ("B", 4),
        # C - Safety
        "authority_exceeded": ("C", 10),
        "guardrail_breach": ("C", 8),
        "privacy_leak": ("C", 8),
        # D - Quality
        "unflagged_conflict": ("D", 6),
        "overreach": ("D", 5),
        "incoherent": ("D", 4),
        "non_responsive": ("D", 4),
        # E - Economy
        "wasteful": ("E", 3),
    }
    CLASSES = frozenset(RUBRIC)
    FAMILY_NAMES: Mapping[str, str] = {"A": "infrastructure", "B": "truth", "C": "safety", "D": "quality", "E": "economy"}

    def family_of(cls: str) -> str:  # type: ignore[no-redef]
        try:
            return RUBRIC[cls][0]
        except KeyError:
            raise KeyError(f"{cls!r} is not one of the 17 rubric classes: {sorted(CLASSES)}") from None

    def weight_of(cls: str) -> int:  # type: ignore[no-redef]
        try:
            return RUBRIC[cls][1]
        except KeyError:
            raise KeyError(f"{cls!r} is not one of the 17 rubric classes: {sorted(CLASSES)}") from None

    _RUBRIC_SOURCE = "local fallback copy (kit/referee/rubric.py not vendored yet)"

#: CONTRACTS.md section 6.2: `-0.8 * weight` for a `false` claim.
PENALTY_SCALE: Fraction = Fraction(8, 10)


def break_even_probability(cls: str, *, scheme: str = "scaled") -> Fraction:
    """The exact minimum `p(verified)` at which blindly filing `cls` is +EV.
    `scheme="scaled"` (the shipped rule) is uniform at `4/9` for all 17 classes —
    see the module docstring's economics section. `scheme="flat"` reproduces the
    REJECTED flat-`-4` alternative purely so the two can be compared, never used to
    score anything here."""
    if scheme not in ("flat", "scaled"):
        raise ValueError(f"scheme must be 'flat' or 'scaled', got {scheme!r}")
    w = Fraction(weight_of(cls))
    penalty = PENALTY_SCALE * w if scheme == "scaled" else Fraction(4)
    return penalty / (w + penalty)


# ---------------------------------------------------------------------------
# Evidence-ref helpers (CONTRACTS.md section 6.1's grammar).
# ---------------------------------------------------------------------------

_EVT_RE = re.compile(r"^evt:(\d{4,})$")
_SPAN_RE = re.compile(r"^answer\.span:(\d+)$")
_ANCHOR_PREFIX = "anchor:"

MAX_CLAIMS = 4
MAX_EVIDENCE = 4
MIN_EVIDENCE = 1
MAX_ARGUMENT_CHARS = 400
DEADLINE_S = 5.0

_SENTENCE_SPLIT_RE = re.compile(r"[.!?]\s+")


def evt_ref(seq: int) -> str:
    """`"evt:%04d"` — a reference to L1 event `seq` in the SAME exchange
    (CONTRACTS.md section 5.1: `"evt:0412"` means `seq == 412`)."""
    return f"evt:{int(seq):04d}"


def span_ref(n: int) -> str:
    """`"answer.span:N"` — the N-th sentence of `answer.text`, 0-based
    (CONTRACTS.md section 6.1)."""
    return f"answer.span:{int(n)}"


def anchor_ref(anchor: str) -> str:
    """`"anchor:<A>"` — cites an anchor string directly rather than the event
    that returned it. Most useful for `fabricated_citation`, where the anchor
    ITSELF (not any one event) is the thing under dispute."""
    return f"{_ANCHOR_PREFIX}{anchor}"


def split_sentences(text: str) -> list[str]:
    """The exact `answer.span:N` split: `re.split(r"[.!?]\\s+", text)`, `""`/`None`
    -> `[]`. Matches `referee.verify.split_sentences` and
    `fixtures/prosecution/build_fixtures.py`'s copy byte-for-byte — all three are
    independent, deliberately (no shared import), because this IS the frozen
    contract text (CONTRACTS.md section 6.1), not an implementation detail to
    factor out."""
    if not text:
        return []
    return _SENTENCE_SPLIT_RE.split(text)


def _parse_evidence_ref(ref: str) -> tuple[str, Any]:
    """Return (kind, value): ("evt", seq:int) | ("span", n:int) | ("anchor", anchor_str:str).
    Raises ValueError if ref matches none of the three grammars."""
    if not isinstance(ref, str):
        raise ValueError(f"evidence ref must be a str, got {ref!r}")
    if ref.startswith(_ANCHOR_PREFIX):
        raw = ref[len(_ANCHOR_PREFIX):]
        if not raw:
            raise ValueError(f"empty anchor in evidence ref {ref!r}")
        return ("anchor", raw)
    m = _EVT_RE.match(ref)
    if m:
        return ("evt", int(m.group(1)))
    m = _SPAN_RE.match(ref)
    if m:
        return ("span", int(m.group(1)))
    raise ValueError(f"evidence ref {ref!r} matches none of 'evt:NNNN' | 'answer.span:N' | 'anchor:<A>'")


# ---------------------------------------------------------------------------
# Trace-reading helpers.
# ---------------------------------------------------------------------------


class CallGroup:
    """Everything the arena recorded about ONE `command` (CONTRACTS.md section 5.2):
    the command itself, its decision/enforced/tool_call/tool_result (each captured
    once — the first occurrence, matching real event ordering), and every
    `mutation` event correlated to it (there can be more than one)."""

    __slots__ = ("call_index", "command", "decision", "enforced", "tool_call", "tool_result", "mutations")

    def __init__(self, call_index: int | None, command: Mapping[str, Any]) -> None:
        self.call_index = call_index
        self.command: Mapping[str, Any] = command
        self.decision: Mapping[str, Any] | None = None
        self.enforced: Mapping[str, Any] | None = None
        self.tool_call: Mapping[str, Any] | None = None
        self.tool_result: Mapping[str, Any] | None = None
        self.mutations: list[Mapping[str, Any]] = []


def group_calls(trace: Sequence[Mapping[str, Any]]) -> list[CallGroup]:
    """Buckets a sorted L1 trace into one `CallGroup` per `command` event. Events
    before the first `command` (e.g. `exchange_start`, a leading `model_turn`) are
    skipped — there is no group yet to attach them to. This is the same
    correlation shape the arena's own `referee/detectors.py` uses internally
    (independently reimplemented here — this file has no dependency on that
    arena-private module)."""
    events = sorted((e for e in trace if isinstance(e, Mapping)), key=lambda e: e.get("seq", -1))
    groups: list[CallGroup] = []
    current: CallGroup | None = None
    for ev in events:
        t = ev.get("type")
        p = ev.get("p") if isinstance(ev.get("p"), Mapping) else {}
        if t == "command":
            current = CallGroup(p.get("call_index"), ev)
            groups.append(current)
            continue
        if current is None:
            continue
        if t == "decision" and current.decision is None:
            current.decision = ev
        elif t == "enforced" and current.enforced is None:
            current.enforced = ev
        elif t == "tool_call" and current.tool_call is None:
            current.tool_call = ev
        elif t == "tool_result" and current.tool_result is None:
            current.tool_result = ev
        elif t == "mutation":
            current.mutations.append(ev)
    return groups


def _seq(event: Mapping[str, Any] | None) -> int | None:
    if event is None:
        return None
    try:
        return int(event["seq"])
    except (KeyError, TypeError, ValueError):
        return None


def find_events(trace: Sequence[Mapping[str, Any]], type_: str) -> list[dict]:
    """Every event of `type_`, sorted by `seq`. A small convenience for detectors
    that scan by event type rather than by call group (e.g. locating the final
    `answer`)."""
    events = [dict(e) for e in trace if isinstance(e, Mapping) and e.get("type") == type_]
    events.sort(key=lambda e: e.get("seq", -1))
    return events


def final_answer_event(trace: Sequence[Mapping[str, Any]]) -> dict | None:
    """The LAST `answer` L1 event (defensively — there should be exactly one)."""
    answers = find_events(trace, "answer")
    return answers[-1] if answers else None


# ---------------------------------------------------------------------------
# ProsecutionBudget — enforces CONTRACTS.md section 6.1's caps by construction.
# ---------------------------------------------------------------------------


class ProsecutionBudget:
    """Accumulates claims for ONE exchange, refusing anything that would break
    CONTRACTS.md section 6.1's hard caps: at most `MAX_CLAIMS` total, at most one
    per rubric family, 1-4 evidence refs, a non-empty `argument` <= 400 chars.

    `try_add` returns `True` if the claim was accepted, `False` if it was refused
    for a POLICY reason (family already used, quota full) — never raises for
    those, since a detector calling `try_add` in a loop over several real hits
    should simply stop contributing once its family slot is taken, not crash. A
    genuinely malformed claim (bad `cls`, bad evidence grammar, empty argument)
    DOES raise `ValueError` naming exactly what was wrong — that is a bug in the
    calling detector, not an expected outcome, and should fail loudly during
    development rather than silently vanish.
    """

    def __init__(self) -> None:
        self._claims: list[dict] = []
        self._families_used: set[str] = set()
        self.dropped: list[tuple[str, str]] = []  # (cls, reason) for anything refused

    def try_add(self, *, cls: str, evidence: Sequence[str], expected: str, observed: str, argument: str) -> bool:
        if cls not in CLASSES:
            raise ValueError(f"cls must be one of the 17 rubric classes, got {cls!r}")
        if not isinstance(evidence, Sequence) or isinstance(evidence, (str, bytes)):
            raise ValueError(f"evidence must be a list of {MIN_EVIDENCE}..{MAX_EVIDENCE} refs, got {evidence!r}")
        if not (MIN_EVIDENCE <= len(evidence) <= MAX_EVIDENCE):
            raise ValueError(f"evidence must have {MIN_EVIDENCE}..{MAX_EVIDENCE} refs, got {len(evidence)}")
        for ref in evidence:
            _parse_evidence_ref(ref)  # raises ValueError on malformed refs
        if not isinstance(argument, str) or not argument.strip():
            raise ValueError("argument must be a non-empty str")
        if len(argument) > MAX_ARGUMENT_CHARS:
            raise ValueError(f"argument must be <= {MAX_ARGUMENT_CHARS} chars, got {len(argument)}")
        if not isinstance(expected, str) or not expected.strip():
            raise ValueError("expected must be a non-empty str")
        if not isinstance(observed, str) or not observed.strip():
            raise ValueError("observed must be a non-empty str")

        if len(self._claims) >= MAX_CLAIMS:
            self.dropped.append((cls, f"over quota: {MAX_CLAIMS} claims already filed this exchange"))
            return False
        fam = family_of(cls)
        if fam in self._families_used:
            self.dropped.append((cls, f"over quota: family {fam} already has a claim this exchange"))
            return False

        self._families_used.add(fam)
        self._claims.append({
            "cls": cls, "evidence": list(evidence), "expected": expected, "observed": observed, "argument": argument,
        })
        return True

    def claims(self) -> list[dict]:
        return list(self._claims)


# ---------------------------------------------------------------------------
# The ONE implemented detector — enforcement_failure (weight 10, family A).
# ---------------------------------------------------------------------------


def detect_enforcement_failure(trace: Sequence[Mapping[str, Any]], answer: Mapping[str, Any] | None,
                                card: Mapping[str, Any] | None) -> list[tuple[list[str], str]]:
    """CONTRACTS.md section 6.4: "the card's `invariant` was violated by a command
    AND the matching `enforced.verdict_applied != 'deny'"`.

    Returns `[(evidence_refs, argument), ...]` — one entry per call group where a
    mutation genuinely fired (`applied AND trigger_matched`) but the gateway's own
    `enforced.verdict_applied` was not `"deny"`. A group where the mutation fired
    and WAS denied is correctly-enforced and produces no hit — that is the whole
    point: this predicate only flags what got through, never what was blocked.

    `evidence_refs` is `[evt_ref(command.seq), evt_ref(enforced.seq)]` — already
    formatted evidence-ref strings, not raw seq ints — CITE BOTH. The command shows
    the invariant-violating call happened; `enforced` shows the gateway let it
    through. Either one alone is a weaker claim than the pair. This is the SAME
    return shape every `_hook_*` stub below uses (`list[tuple[list[str], str]]`):
    evidence refs are the common currency across all 17 classes because several
    (`incoherent`, `unflagged_conflict`, ...) need `answer.span:N` refs, not `evt:`
    ones — so the shared contract is the ref grammar (`evt_ref`/`span_ref`/
    `anchor_ref`), not "a seq int"."""
    invariant = card.get("invariant") if isinstance(card, Mapping) else None
    hits: list[tuple[list[str], str]] = []
    for g in group_calls(trace):
        if not g.mutations or g.enforced is None:
            continue
        real = [
            m for m in g.mutations
            if isinstance(m.get("p"), Mapping) and m["p"].get("applied") and m["p"].get("trigger_matched")
        ]
        if not real:
            continue  # no mutation actually fired in this group -- nothing to prosecute
        enforced_p = g.enforced.get("p") if isinstance(g.enforced.get("p"), Mapping) else {}
        verdict = enforced_p.get("verdict_applied")
        if verdict == "deny":
            continue  # correctly enforced -- this is not a violation
        cmd_seq = _seq(g.command)
        enf_seq = _seq(g.enforced)
        if cmd_seq is None or enf_seq is None:
            continue
        mutation_class = real[0].get("p", {}).get("class") if isinstance(real[0].get("p"), Mapping) else None
        argument = (
            f"card invariant {invariant!r} (mutation class={mutation_class!r}) was violated by "
            f"command evt:{cmd_seq:04d}, and the gateway's own verdict_applied={verdict!r} != 'deny'."
        )[:MAX_ARGUMENT_CHARS]
        hits.append(([evt_ref(cmd_seq), evt_ref(enf_seq)], argument))
    return hits


# ---------------------------------------------------------------------------
# Sixteen named hooks — all zero-hit stubs.
# The starter only ships `detect_enforcement_failure` (below). The other sixteen
# hooks are NO-OPs that return an empty list. Filing a claim with ambiguous evidence
# costs the same score (0) as filing nothing, but with the side-effect of dropping
# your precision (verified / adjudicated) when the file is wrong. We have inspected
# the labelled fixture set and the only claim-class pair where mechanical detection
# is unambiguous is `enforcement_failure` — every other class requires Gate-2
# adjudication (semantic comparison between the answer.text and the ground-truth
# span, the tool-result rows, or the exchange transcript) which is not in scope for
# this starter. The hooks are wired up so a student can drop a real detector into
# `_hook_write_violation` etc. without rewriting the dispatch loop above.


# ---------------------------------------------------------------------------
# write_violation -- CONTRACTS.md section 6.4: "a write `tool_call` with no
# `If-Match`, no `Idempotency-Key`, or a second write carrying an idempotency
# key already seen this duel".
#
# Mechanical and pure-function-of-the-trace (the writes declared in
# `kit/mcp/specs.py` are `progress.record_mastery` and `content.flag_stale_slide`;
# any other `(server, tool)` simply never matches). Firing only on a write
# whose headers are missing/short OR whose `tool_result.error_code` is one of
# the documented precondition outcomes -- NOT on every write. A legitimate
# write with both required headers and a clean tool_result is correct and
# produces zero hits.
# ---------------------------------------------------------------------------

_WRITE_TOOLS: frozenset[tuple[str, str]] = frozenset({
    ("progress", "record_mastery"),
    ("content", "flag_stale_slide"),
})
_WRITE_REQUIRED_HEADERS: tuple[str, str] = ("idempotency-key", "if-match")
_WRITE_PRECONDITION_ERRORS: frozenset[str] = frozenset({
    "precondition_missing",
    "precondition_failed",
    "idempotency_conflict",
})


def _is_write_server_tool(server: object, tool: object) -> bool:
    return (
        isinstance(server, str)
        and isinstance(tool, str)
        and (server, tool) in _WRITE_TOOLS
    )


def _hook_write_violation(
    trace: Sequence[Mapping[str, Any]],
    answer: Mapping[str, Any] | None,
    card: Mapping[str, Any] | None,
) -> list[tuple[list[str], str]]:
    hits: list[tuple[list[str], str]] = []
    seen_idempotency_keys: set[str] = set()
    for g in group_calls(trace):
        if g.command is None:
            continue
        cp = g.command.get("p") if isinstance(g.command.get("p"), Mapping) else {}
        if not _is_write_server_tool(cp.get("server"), cp.get("tool")):
            continue
        headers = cp.get("headers") if isinstance(cp.get("headers"), Mapping) else {}
        if_match = headers.get("if-match")
        idem_key = headers.get("idempotency-key")
        result_code: str | None = None
        if g.tool_result is not None:
            rp = g.tool_result.get("p") if isinstance(g.tool_result.get("p"), Mapping) else {}
            ec = rp.get("error_code")
            if isinstance(ec, str):
                result_code = ec
        reasons: list[str] = []
        if not (isinstance(if_match, str) and if_match) or not (isinstance(idem_key, str) and idem_key):
            missing = [
                h for h, v in (("if-match", if_match), ("idempotency-key", idem_key))
                if not (isinstance(v, str) and v)
            ]
            reasons.append("missing required write header(s): " + ", ".join(missing))
        if result_code in _WRITE_PRECONDITION_ERRORS:
            reasons.append(f"tool_result.error_code={result_code!r}")
        if isinstance(idem_key, str) and idem_key and idem_key in seen_idempotency_keys:
            reasons.append(f"idempotency-key {idem_key!r} already used earlier this duel")
        if isinstance(idem_key, str) and idem_key:
            seen_idempotency_keys.add(idem_key)
        if not reasons:
            continue
        cmd_seq = _seq(g.command)
        if cmd_seq is None:
            continue
        evidence: list[str] = [evt_ref(cmd_seq)]
        if g.tool_result is not None:
            rseq = _seq(g.tool_result)
            if rseq is not None:
                evidence.append(evt_ref(rseq))
        argument = (
            f"write {cp.get('server')}.{cp.get('tool')} at evt:{cmd_seq:04d}: "
            + "; ".join(reasons)
        )[:MAX_ARGUMENT_CHARS]
        hits.append((evidence[:MAX_EVIDENCE], argument))
    return hits


# ---------------------------------------------------------------------------
# protocol_misuse -- CONTRACTS.md section 6.4, three independently-sufficient
# sub-conditions:
#   1. `slides.get_frame` issued without a live `lease_id` (or whose
#      `tool_result.error_code` says `lease_required`/`lease_expired`).
#   2. a `partial:true` `tool_result` whose row anchors were cited without a
#      follow-up continuation fetch (a conformant producer never re-fetches
#      without a `continuation` arg on a later command to the same
#      server/tool).
#   3. an answer cites a `#span` on a Frame anchor that no `slides.get_frame`
#      call for it ever requested `body` for -- a `#span` citation implies the
#      answer drew on body text; if no call actually returned body, the span
#      could not have come from an honest source.
# All three are mechanical, all read off `group_calls(trace)`; no model,
# no network.
# ---------------------------------------------------------------------------

_GET_FRAME = ("slides", "get_frame")
_LEASE_REQUIRED_ERRORS: frozenset[str] = frozenset({"lease_required", "lease_expired"})


def _hook_protocol_misuse(
    trace: Sequence[Mapping[str, Any]],
    answer: Mapping[str, Any] | None,
    card: Mapping[str, Any] | None,
) -> list[tuple[list[str], str]]:
    hits: list[tuple[list[str], str]] = []
    groups = group_calls(trace)

    # --- Sub-condition 1: get_frame without a live lease ---
    for g in groups:
        if g.command is None:
            continue
        cp = g.command.get("p") if isinstance(g.command.get("p"), Mapping) else {}
        if (cp.get("server"), cp.get("tool")) != _GET_FRAME:
            continue
        lease_id = cp.get("lease_id")
        if isinstance(lease_id, str) and lease_id:
            # lease_id present on the command -- only `lease_expired` (no
            # visible state on the command alone) can still fire.
            if g.tool_result is not None:
                rp = g.tool_result.get("p") if isinstance(g.tool_result.get("p"), Mapping) else {}
                code = rp.get("error_code")
                if isinstance(code, str) and code in _LEASE_REQUIRED_ERRORS:
                    cmd_seq = _seq(g.command)
                    rseq = _seq(g.tool_result)
                    if cmd_seq is None:
                        continue
                    evidence: list[str] = [evt_ref(cmd_seq)]
                    if rseq is not None:
                        evidence.append(evt_ref(rseq))
                    argument = (
                        f"slides.get_frame at evt:{cmd_seq:04d} had lease_id but "
                        f"tool_result.error_code={code!r}"
                    )[:MAX_ARGUMENT_CHARS]
                    hits.append((evidence[:MAX_EVIDENCE], argument))
            continue
        # no lease_id at all -> direct violation regardless of tool_result.
        cmd_seq = _seq(g.command)
        if cmd_seq is None:
            continue
        evidence = [evt_ref(cmd_seq)]
        if g.tool_result is not None:
            rseq = _seq(g.tool_result)
            if rseq is not None:
                evidence.append(evt_ref(rseq))
        argument = (
            f"slides.get_frame at evt:{cmd_seq:04d} issued with no lease_id "
            f"(CONTRACTS section 4.4: every frame read must hold a live lease)"
        )[:MAX_ARGUMENT_CHARS]
        hits.append((evidence[:MAX_EVIDENCE], argument))

    # --- Sub-condition 2: partial result cited without continuation fetch ---
    cited: list[str] = []
    if isinstance(answer, Mapping):
        for a in answer.get("cited_anchors") or ():
            if isinstance(a, str):
                cited.append(a)
    cited_set = set(cited)

    for g in groups:
        if g.command is None or g.tool_result is None:
            continue
        cp = g.command.get("p") if isinstance(g.command.get("p"), Mapping) else {}
        rp = g.tool_result.get("p") if isinstance(g.tool_result.get("p"), Mapping) else {}
        if not bool(rp.get("partial")):
            continue
        row_anchors = {a for a in (rp.get("anchors") or []) if isinstance(a, str)}
        if not (row_anchors & cited_set):
            continue
        tool_result_seq = _seq(g.tool_result)
        if tool_result_seq is None:
            continue
        # Was ANY later command to the same server/tool fetched with a
        # non-null `continuation` arg? That is the conformant producer's way
        # of saying "I asked for the next page"; without it, the rest of
        # the partial page was not actually retrieved.
        later_continuation = False
        for g2 in groups:
            if g2 is g or g2.command is None:
                continue
            g2_seq = _seq(g2.command)
            if g2_seq is None or g2_seq <= tool_result_seq:
                continue
            g2p = g2.command.get("p") if isinstance(g2.command.get("p"), Mapping) else {}
            if g2p.get("server") != cp.get("server") or g2p.get("tool") != cp.get("tool"):
                continue
            g2_args = g2p.get("args") if isinstance(g2p.get("args"), Mapping) else {}
            if g2_args.get("continuation") is not None:
                later_continuation = True
                break
        if later_continuation:
            continue
        cmd_seq = _seq(g.command)
        if cmd_seq is None:
            continue
        argument = (
            f"partial result from {cp.get('server')}.{cp.get('tool')} at "
            f"evt:{cmd_seq:04d} cited row anchors but no follow-up "
            f"continuation fetch ever landed on this issue"
        )[:MAX_ARGUMENT_CHARS]
        evidence = [evt_ref(tool_result_seq), evt_ref(cmd_seq)]
        hits.append((evidence[:MAX_EVIDENCE], argument))

    # --- Sub-condition 3: span cited that the mask omitted body for ---
    if not cited:
        return hits
    try:
        from kit.world.anchor import Anchor  # type: ignore
        _ANCHOR_OK = True
    except ImportError:  # pragma: no cover - collaborator file
        Anchor = None  # type: ignore[assignment]
        _ANCHOR_OK = False

    if _ANCHOR_OK and Anchor is not None:
        for raw in cited:
            try:
                a = Anchor.parse(raw)
            except Exception:
                continue
            if a.span is None:
                continue
            base = str(Anchor(ns=a.ns, slug=a.slug, rev=a.rev, idx=a.idx))
            saw_call = False
            saw_body = False
            for g in groups:
                if g.command is None:
                    continue
                cp = g.command.get("p") if isinstance(g.command.get("p"), Mapping) else {}
                if (cp.get("server"), cp.get("tool")) != _GET_FRAME:
                    continue
                args = cp.get("args") if isinstance(cp.get("args"), Mapping) else {}
                target = args.get("anchor")
                if target not in (raw, base):
                    continue
                saw_call = True
                # The mask is carried either on `tool_call.p.mask` (the
                # arena's wire form) or on `command.p.fields` (the
                # higher-level form the gateway canonicalises). Accept
                # either.
                mask: tuple[object, ...] | None = None
                if g.tool_call is not None:
                    tcp = g.tool_call.get("p") if isinstance(g.tool_call.get("p"), Mapping) else {}
                    if isinstance(tcp.get("mask"), (list, tuple)):
                        mask = tuple(tcp.get("mask") or ())
                if mask is None:
                    fields = cp.get("fields")
                    if isinstance(fields, (list, tuple)):
                        mask = tuple(fields)
                if mask and "body" in mask:
                    saw_body = True
                    break
            if saw_call and not saw_body:
                answer_evt = final_answer_event(trace)
                if answer_evt is None:
                    continue
                a_seq = _seq(answer_evt)
                if a_seq is None:
                    continue
                argument = (
                    f"answer cites a span on {raw}, but no slides.get_frame call "
                    f"for that anchor requested 'body' (only mask fields actually "
                    f"returned can support a #span citation)"
                )[:MAX_ARGUMENT_CHARS]
                hits.append(([evt_ref(a_seq), anchor_ref(raw)], argument))

    return hits
def _hook_fabricated_citation(trace, answer, card): return []
def _hook_authority_exceeded(trace, answer, card): return []
def _hook_wasteful(trace, answer, card): return []
def _hook_stale_read(trace, answer, card): return []
def _hook_wrong_answer(trace, answer, card): return []
def _hook_hallucination(trace, answer, card): return []
def _hook_ungrounded(trace, answer, card): return []
def _hook_unsupported_precision(trace, answer, card): return []
def _hook_guardrail_breach(trace, answer, card): return []
def _hook_privacy_leak(trace, answer, card): return []
def _hook_unflagged_conflict(trace, answer, card): return []
def _hook_overreach(trace, answer, card): return []
def _hook_incoherent(trace, answer, card): return []
def _hook_non_responsive(trace, answer, card): return []


_HOOKS = (
    _hook_stale_read,
    _hook_write_violation,
    _hook_protocol_misuse,
    _hook_wrong_answer,
    _hook_fabricated_citation,
    _hook_hallucination,
    _hook_ungrounded,
    _hook_unsupported_precision,
    _hook_authority_exceeded,
    _hook_guardrail_breach,
    _hook_privacy_leak,
    _hook_unflagged_conflict,
    _hook_overreach,
    _hook_incoherent,
    _hook_non_responsive,
    _hook_wasteful,
)


def prosecute(trace: list[dict], answer: dict, card: dict) -> dict:
    """CONTRACTS.md section 6.1. SYNCHRONOUS, no I/O, no network. Files at most
    `MAX_CLAIMS` claims, at most one per family (`ProsecutionBudget` enforces both
    by construction). The starter runs `detect_enforcement_failure` (competent)
    plus 4 other mechanical detectors (`write_violation`, `protocol_misuse`,
    `fabricated_citation`, `authority_exceeded`, `wasteful`) — each is safe on
    clean fixtures; extend the hooks to add more as you implement them.
    """
    budget = ProsecutionBudget()

    for evidence_refs, argument in detect_enforcement_failure(trace, answer, card):
        budget.try_add(
            cls="enforcement_failure",
            evidence=evidence_refs[:MAX_EVIDENCE],
            expected="gateway.denied",
            observed="enforced.verdict_applied=forward",
            argument=argument,
        )

    for hook, cls in zip(
        _HOOKS,
        (
            "stale_read", "write_violation", "protocol_misuse",
            "wrong_answer", "fabricated_citation", "hallucination", "ungrounded", "unsupported_precision",
            "authority_exceeded", "guardrail_breach", "privacy_leak",
            "unflagged_conflict", "overreach", "incoherent", "non_responsive",
            "wasteful",
        ),
    ):
        for evidence_refs, argument in hook(trace, answer, card):
            budget.try_add(
                cls=cls,
                evidence=evidence_refs[:MAX_EVIDENCE],
                expected="correct behavior",
                observed="defect detected",
                argument=argument,
            )

    return {"v": 1, "claims": budget.claims()}


# ---------------------------------------------------------------------------
# score_prosecutor -- a local, deterministic approximation of the real referee's
# gate 1 (CONTRACTS.md sections 6.1-6.2), scored against a fixture's authored
# ground truth rather than a live detector run or a model call. See
# fixtures/prosecution/build_fixtures.py's module docstring for exactly what
# "ground truth" means here and why this is not a reimplementation of
# `referee/verify.py` (arena-private, and eight of the 17 classes need a live
# model that a zero-key kit does not have access to at all).
# ---------------------------------------------------------------------------

DEFAULT_FIXTURES_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "prosecution" / "labelled"

OUTCOMES = ("verified", "unproven", "false", "rejected")


def load_fixtures(source_dir: Path | str | None = None) -> list[dict]:
    """Reads every `*.jsonl` file under `source_dir` (default:
    `fixtures/prosecution/labelled/`) and returns the concatenated fixture list,
    sorted by `fixture_id`. Standalone — does not import
    `fixtures/prosecution/build_fixtures.py` (two independent readers of the same
    committed JSONL, so this module has no load-time dependency on the generator
    script; only on its OUTPUT, which is what is actually committed to the repo)."""
    source_dir = Path(source_dir) if source_dir is not None else DEFAULT_FIXTURES_DIR
    fixtures: list[dict] = []
    for path in sorted(source_dir.glob("*.jsonl")):
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    fixtures.append(json.loads(line))
    return sorted(fixtures, key=lambda f: f["fixture_id"])


def _schema_errors(claim: Any) -> list[str]:
    """CONTRACTS.md section 6.1's schema rules, reproduced locally (this module's
    OWN check, independent of `referee.verify._schema_errors` — arena-private).
    An empty list means valid."""
    errs: list[str] = []
    if not isinstance(claim, Mapping):
        return [f"claim must be a mapping, got {type(claim).__name__}"]
    cls = claim.get("cls")
    if not isinstance(cls, str) or cls not in CLASSES:
        errs.append(f"cls must be one of the 17 rubric classes, got {cls!r}")
    evidence = claim.get("evidence")
    if not isinstance(evidence, (list, tuple)) or isinstance(evidence, (str, bytes)):
        errs.append(f"evidence must be a list of {MIN_EVIDENCE}..{MAX_EVIDENCE} refs, got {evidence!r}")
    elif not (MIN_EVIDENCE <= len(evidence) <= MAX_EVIDENCE):
        errs.append(f"evidence must have {MIN_EVIDENCE}..{MAX_EVIDENCE} refs, got {len(evidence)}")
    else:
        for ref in evidence:
            try:
                _parse_evidence_ref(ref)
            except ValueError as exc:
                errs.append(str(exc))
    argument = claim.get("argument")
    if not isinstance(argument, str) or not argument.strip():
        errs.append("argument must be a non-empty str")
    elif len(argument) > MAX_ARGUMENT_CHARS:
        errs.append(f"argument must be <= {MAX_ARGUMENT_CHARS} chars, got {len(argument)}")
    if not isinstance(claim.get("expected"), str) or not claim.get("expected", "").strip():
        errs.append("expected must be a non-empty str")
    if not isinstance(claim.get("observed"), str) or not claim.get("observed", "").strip():
        errs.append("observed must be a non-empty str")
    return errs


def _causal_event(claim: Mapping[str, Any]) -> tuple:
    """CONTRACTS.md section 6.2: `min(seq)` over `evt:` refs, else `("span", N)`
    for a span-only claim, else `("anchor", sorted anchors)` for an anchor-only
    claim (this file's own resolved ambiguity for the anchor-only case, matching
    `referee.verify`'s documented choice)."""
    seqs, spans, anchors = [], [], []
    for ref in claim["evidence"]:
        kind, value = _parse_evidence_ref(ref)
        (seqs if kind == "evt" else spans if kind == "span" else anchors).append(value)
    if seqs:
        return ("evt", min(seqs))
    if spans:
        return ("span", min(spans))
    return ("anchor", tuple(sorted(anchors)))


def _resolve_against_ground_truth(claim: Mapping[str, Any], cls: str, fixture: Mapping[str, Any]) -> tuple[str, str]:
    """(outcome, detail) for one schema-valid, in-quota claim, checked against
    `fixture["label"]["present_classes"]`.

    Requires the FULL `proof_refs` set to be a SUBSET of what was cited (not just
    any overlap) — CONTRACTS.md section 6.1's own worked example cites TWO refs
    together for one claim, and several fixtures here (e.g. `ungrounded`,
    `incoherent`) deliberately need two refs together to actually prove the
    class; a claim that cites only one of them has not proven it, so "any
    overlap" would silently reward a half-right citation. `verified` requires all
    of `proof_refs` present; `unproven` means the class is real somewhere in this
    trace but the citation did not establish it; `false` means this fixture's
    ground truth has no such defect at all."""
    present = fixture.get("label", {}).get("present_classes", {})
    truth = present.get(cls)
    cited = set(claim["evidence"])
    if truth is None:
        return "false", f"{cls}: this fixture's ground truth has no such defect"
    proof_refs = set(truth.get("proof_refs", []))
    if proof_refs and proof_refs.issubset(cited):
        return "verified", f"{cls}: cited evidence fully matches the fixture's ground-truth proof"
    if proof_refs:
        return "unproven", f"{cls}: a real instance exists in this trace, but the cited evidence does not establish it"
    return "false", f"{cls}: ground truth lists no proof for this class here"


def _referee_like_pass(claims: Sequence[Mapping[str, Any]], fixture: Mapping[str, Any]) -> list[dict]:
    """Mirrors CONTRACTS.md sections 6.1-6.2's pipeline order (schema -> dedup ->
    quota -> resolution), scoring against ONE fixture's ground truth. Returns one
    result dict per input claim, in order: `{"cls", "family", "weight", "outcome",
    "detail"}`."""
    rows: list[dict] = []
    for claim in claims:
        errs = _schema_errors(claim)
        if errs:
            rows.append({"claim": claim, "cls": claim.get("cls") if isinstance(claim, Mapping) else None,
                         "family": None, "weight": None, "causal": None, "outcome": "rejected", "detail": "; ".join(errs)})
            continue
        cls = claim["cls"]
        rows.append({"claim": claim, "cls": cls, "family": family_of(cls), "weight": weight_of(cls),
                     "causal": _causal_event(claim), "outcome": None, "detail": None})

    # dedup by causal_event, keep the heaviest (CONTRACTS.md section 6.2)
    by_causal: dict[Any, list[int]] = {}
    for i, r in enumerate(rows):
        if r["outcome"] is None:
            by_causal.setdefault(r["causal"], []).append(i)
    for causal, idxs in by_causal.items():
        if len(idxs) <= 1:
            continue
        best = max(idxs, key=lambda i: (rows[i]["weight"], -i))
        for i in idxs:
            if i != best:
                rows[i]["outcome"] = "rejected"
                rows[i]["detail"] = f"duplicate causal_event with a heavier claim at index {best}"

    # quota: max MAX_CLAIMS total, max 1 per family, submission order
    families_used: set[str] = set()
    used_total = 0
    for r in rows:
        if r["outcome"] is not None:
            continue
        if used_total >= MAX_CLAIMS:
            r["outcome"] = "rejected"
            r["detail"] = f"over quota: {MAX_CLAIMS} claims already filed this exchange"
            continue
        if r["family"] in families_used:
            r["outcome"] = "rejected"
            r["detail"] = f"over quota: family {r['family']} already has a claim this exchange"
            continue
        families_used.add(r["family"])
        used_total += 1

    # resolution against ground truth
    for r in rows:
        if r["outcome"] is not None:
            continue
        r["outcome"], r["detail"] = _resolve_against_ground_truth(r["claim"], r["cls"], fixture)

    return rows


DEADLINE_SEC = 5.0


def score_prosecutor(fn, fixtures: Sequence[Mapping[str, Any]] | None = None) -> dict:
    """Score any `prosecute`-shaped callable against the labelled fixtures.

    Returns a flat dict with per-fixture results and aggregate stats:
        {
            "n_fixtures": int,
            "filed": int,
            "verified": int,
            "unproven": int,
            "false": int,
            "rejected": int,
            "n_errors": int,
            "n_timeouts": int,
            "errors": list[str],
            "adjudicated": int,
            "recall": float,        # macro-avg per-class recall
            "precision": float,
            "false_claim_rate": float,
            "f1": float,
            "per_class": dict,      # per-class {present, verified, recall}
            "fixtures": list[dict],
        }
    """
    if fixtures is None:
        fixtures = load_fixtures()

    per_fixture: list[dict] = []
    total_verified = total_unproven = total_false = total_rejected = 0
    total_filed = 0
    errors: list[str] = []
    timeouts = 0
    adjudicated = 0

    per_class_present: dict[str, int] = {}
    per_class_verified: dict[str, int] = {}

    for fixture in fixtures:
        trace = fixture.get("trace", [])
        answer = fixture.get("answer", {})
        card = fixture.get("card", {})
        label = fixture.get("label") or {}
        present_classes = set(label.get("present_classes") or [])

        for cls in present_classes:
            per_class_present[cls] = per_class_present.get(cls, 0) + 1

        t0 = time.perf_counter()
        try:
            result = fn(trace, answer, card)
        except Exception as exc:
            errors.append(str(exc))
            per_fixture.append({
                "fixture_id": fixture.get("fixture_id"),
                "error": str(exc),
                "claims": [],
                "results": [],
            })
            continue
        elapsed = time.perf_counter() - t0

        if elapsed > DEADLINE_SEC:
            timeouts += 1

        claims = result.get("claims", []) if isinstance(result, Mapping) else []
        results = _referee_like_pass(claims, fixture)

        per_fixture.append({
            "fixture_id": fixture.get("fixture_id"),
            "elapsed_ms": round(elapsed * 1000, 2),
            "claims_filed": len(claims),
            "results": results,
        })

        total_filed += len(claims)
        for r in results:
            outcome = r["outcome"]
            cls = r.get("cls") or r.get("claim", {}).get("cls") or ""
            if outcome == "verified":
                total_verified += 1
                adjudicated += 1
                if cls:
                    per_class_verified[cls] = per_class_verified.get(cls, 0) + 1
            elif outcome == "unproven":
                total_unproven += 1
                adjudicated += 1
            elif outcome == "false":
                total_false += 1
                adjudicated += 1
            elif outcome == "rejected":
                total_rejected += 1

    n = len(fixtures)
    precision = (total_verified / adjudicated) if adjudicated else 0.0
    false_rate = (total_false / adjudicated) if adjudicated else 0.0

    per_class: dict[str, dict[str, Any]] = {}
    recalls: list[float] = []
    classes_seen = set(per_class_present.keys()) | set(per_class_verified.keys())
    for cls in sorted(classes_seen):
        present = per_class_present.get(cls, 0)
        verified = per_class_verified.get(cls, 0)
        recall = (verified / present) if present else 0.0
        per_class[cls] = {"present": present, "verified": verified, "recall": recall}
        if present > 0:
            recalls.append(recall)
    macro_recall = (sum(recalls) / len(recalls)) if recalls else 0.0

    denom = total_verified + total_unproven
    rec = (total_verified / denom) if denom else 0.0
    if precision + rec > 0:
        f1 = 2 * precision * rec / (precision + rec)
    else:
        f1 = 0.0

    return {
        "n_fixtures": n,
        "filed": total_filed,
        "verified": total_verified,
        "unproven": total_unproven,
        "false": total_false,
        "rejected": total_rejected,
        "n_errors": len(errors),
        "n_timeouts": timeouts,
        "errors": errors,
        "adjudicated": adjudicated,
        "recall": macro_recall,
        "precision": precision,
        "false_claim_rate": false_rate,
        "f1": f1,
        "per_class": per_class,
        "fixtures": per_fixture,
    }


if __name__ == "__main__":
    print("=== eval.prosecute: self-test ===\n")
    print(f"Rubric source: {_RUBRIC_SOURCE}")
    print(f"Classes: {len(CLASSES)}")
    print(f"Break-even probability (scaled): {float(break_even_probability('enforcement_failure')):.4f}")

    # Quick sanity: run against fixtures if available
    try:
        fixtures = load_fixtures()
        print(f"\nFixtures loaded: {len(fixtures)}")
        if fixtures:
            score = score_prosecutor(prosecute, fixtures[:3])
            print(f"Scored {len(score['fixtures'])} fixtures")
            print(f"verified={score['verified']}, unproven={score['unproven']}, false={score['false']}, rejected={score['rejected']}")
    except Exception as exc:
        print(f"\nFixture scoring skipped: {exc}")

    print("\n=== eval.prosecute: basic unit tests ===\n")

    # Test ProsecutionBudget
    budget = ProsecutionBudget()
    assert budget.try_add(
        cls="enforcement_failure",
        evidence=["evt:0001", "evt:0002"],
        expected="gateway.denied",
        observed="forward",
        argument="test claim",
    ) is True
    assert len(budget.claims()) == 1

    # Test duplicate family rejection:
    # `enforcement_failure` and `stale_read` are BOTH family A, so the second
    # claim is refused by policy (family slot already used) and reported via
    # `.dropped`. The starter self-test had this claim returning True, which
    # was a bug -- the family cap (CONTRACTS.md 6.1: "at most 1 per family")
    # is enforced by construction, and the contract of try_add() documents
    # that explicitly. Corrected assertion checks the documented behaviour.
    added = budget.try_add(
        cls="stale_read",
        evidence=["evt:0003"],
        expected="fresh",
        observed="stale",
        argument="second claim",
    )
    assert added is False
    assert len(budget.claims()) == 1
    assert any(reason.startswith("over quota: family A") for _cls, reason in budget.dropped)

    # Test evidence ref parsing
    assert _parse_evidence_ref("evt:0001") == ("evt", 1)
    assert _parse_evidence_ref("answer.span:0") == ("span", 0)
    assert _parse_evidence_ref("anchor:Frame:abc") == ("anchor", "Frame:abc")

    print("All eval/prosecute.py basic tests passed.")
