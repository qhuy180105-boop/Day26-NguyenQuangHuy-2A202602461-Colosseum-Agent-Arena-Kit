"""agent/gateway.py — YOUR control plane. CONTRACTS.md section 4, exactly.

READ agent/README.md FIRST — it maps all five files in this directory to what
each is scored on. This file is the one CONTRACTS.md calls "the trusted
envelope's untrusted half": every single MCP / A2A / DISCOVER command your
agent's model wants to make passes through `Gateway.decide` before it is
allowed to happen.

WHY THERE IS NO `execute()` METHOD ON `GatewayContext` (read this before you
go looking for one — there isn't one, and that is not an oversight)
----------------------------------------------------------------------------
CONTRACTS.md section 4's trusted envelope, reproduced here because it is the
one diagram worth memorising:

    [ trusted ]   loop emits a raw action line
         v
    [ trusted ]   INTERCEPT + CANONICALISE -> Command        (kit/loop/agent.py)
         v
    [ UNTRUSTED ] Gateway.decide(cmd) -> Decision             <- THIS FILE
         v
    [ trusted ]   ENFORCE: honour the Decision, meter it,
                  apply the active mutation, execute the
                  ToolCall or refuse it                       (the arena)
         v
    [ trusted ]   RECORD the authoritative L1 event, then
                  RENDER the Observation                      (the arena)
         v
    [ trusted ]   the model sees the Observation

`decide()` returns a *decision*, never a *result*. You cannot reach a tool
server, a file, a socket, or a clock from in here — there is nothing to
call. Two things follow from that, and both matter more than they look:

  1. YOUR TRACE CANNOT BE FORGED. Every `command` / `decision` / `enforced`
     / `tool_call` / `tool_result` L1 event (CONTRACTS.md 5.2) is written by
     the arena, from what the arena itself actually did — never from
     anything you claimed happened. A student gateway that wanted to lie
     about having blocked an attack ("I totally denied that, trust me")
     simply has no channel to lie through: the only thing you ever hand
     back is this one small `Decision` value, and the arena is the one that
     turns it into history.
  2. NOBODY CAN ACCUSE YOU OF A CALL YOU DID NOT AUTHORISE, either. Because
     `decide()` is the ONLY door a command can walk through on its way to
     actually running, a prosecutor's `enforcement_failure` claim against
     you has exactly one thing to point at: the `Decision` you returned for
     that specific `cmd_id`. There is no ambiguity about "maybe the loop
     called the tool directly" — CONTRACTS.md 4.2 removed that path on
     purpose, and kit/loop/agent.py's own module docstring names the same
     invariant from the other side (the loop never imports this module,
     never sees a `Decision`, never executes anything itself).

The cost of that guarantee is that this file is PURE: synchronous, no I/O,
no threads, no `sleep`, 250 ms wall-clock deadline (RULES.md section 3).
Raising anything, returning something that is not a valid `Decision`, or
missing the deadline is treated by the arena as a DENIED command PLUS a 2
credit penalty PLUS an `integrity` event that hands the prosecutor a free
`enforcement_failure` — CONTRACTS.md 4.1's charging table, reproduced in
agent/README.md's own table. Getting this file to just plainly return valid
`Decision` values, every time, is worth more than getting it clever.

Stdlib only. No network, no randomness, no wall-clock reads, no sleeping —
none of that would even survive the kernel sandbox (CONTRACTS.md 12), but
the point is this file has no reason to want any of it in the first place.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, runtime_checkable

# kit.mcp.types is a collaborator's file (workspace hard rule 2: import it,
# degrade gracefully). It is present as of this writing and is core, stable
# infrastructure (CONTRACTS.md 3.1) — but this module must still not fail to
# IMPORT if a concurrent edit ever breaks it transiently. When it is
# unavailable, `Decision.call` type-checking is skipped (not enforced), and
# `Gateway.decide` falls back to a minimal local dict-shaped stand-in so the
# rest of this file — everything that does not need a *real* ToolCall — still
# runs.
try:
    from kit.mcp.types import ToolCall
    _TOOLCALL_AVAILABLE = True
except ImportError:  # pragma: no cover - collaborator file
    ToolCall = Any  # type: ignore[assignment, misc]
    _TOOLCALL_AVAILABLE = False

# kit.loop.agent is also a collaborator's file, used only by this module's
# own __main__ demo (to build real Commands the same way the arena's trusted
# canonicaliser would) — never by decide() itself, which never touches the
# loop. Degraded the same way.
try:
    from kit.loop.agent import canonicalise_action as _canonicalise_action
except ImportError:  # pragma: no cover - collaborator file
    _canonicalise_action = None

from agent.telemetry import RecordingGatewayContext, Telemetry

__all__ = [
    "COMMAND_KINDS",
    "DECISION_VERDICTS",
    "Command",
    "Decision",
    "GatewayContext",
    "Gateway",
]

# CONTRACTS.md 4.1: `Command.kind` — "mcp" | "a2a" | "discover". An "answer"
# action is NEVER a Command (kit/loop/agent.py's own module docstring: "an
# answer is not a tool call routed to a server, so it never becomes a
# Command at all") — it is handled entirely by the loop/arena and never
# reaches `Gateway.decide`.
COMMAND_KINDS: frozenset[str] = frozenset({"mcp", "a2a", "discover"})

# CONTRACTS.md 4.1: `Decision.verdict` — the closed three-member set.
DECISION_VERDICTS: frozenset[str] = frozenset({"forward", "deny", "rewrite"})


@dataclass(frozen=True, slots=True)
class Command:
    """CONTRACTS.md 4.1, field for field — "canonicalised by the arena
    BEFORE the student sees it". You never build one of these from your own
    agent's raw text; the arena's canonicaliser (kit/loop/agent.py's
    `canonicalise_action`, run inside the trusted envelope) already did that
    work and minted `cmd_id` by the time `decide()` sees it. The
    `from_action_dict` classmethod below exists only so this file's own demo
    (and your local tests, if you write any) can build a realistic `Command`
    without duplicating the arena's canonicalisation logic."""

    cmd_id: str
    kind: str  # "mcp" | "a2a" | "discover" — see COMMAND_KINDS
    raw: str
    server: str
    tool: str
    args: dict
    fields: tuple[str, ...]
    headers: dict
    lease_id: str | None
    call_index: int

    def __post_init__(self) -> None:
        if not isinstance(self.cmd_id, str) or not self.cmd_id:
            raise ValueError(f"Command.cmd_id must be a non-empty str, got {self.cmd_id!r}")
        if self.kind not in COMMAND_KINDS:
            raise ValueError(f"Command.kind must be one of {sorted(COMMAND_KINDS)}, got {self.kind!r}")
        if not isinstance(self.server, str) or not self.server:
            raise ValueError(f"Command.server must be a non-empty str, got {self.server!r}")
        if not isinstance(self.tool, str) or not self.tool:
            raise ValueError(f"Command.tool must be a non-empty str, got {self.tool!r}")
        if not isinstance(self.args, dict):
            raise ValueError(f"Command.args must be a dict, got {type(self.args).__name__}")
        if not isinstance(self.headers, dict):
            raise ValueError(f"Command.headers must be a dict, got {type(self.headers).__name__}")
        if (
            not isinstance(self.call_index, int)
            or isinstance(self.call_index, bool)
            or self.call_index < 0
        ):
            raise ValueError(f"Command.call_index must be a non-negative int, got {self.call_index!r}")

    @classmethod
    def from_action_dict(cls, action: Mapping[str, Any], *, cmd_id: str) -> "Command":
        """Build a `Command` from the dict shape `kit.loop.agent.canonicalise_action`
        returns (`kind, raw, server, tool, args, fields, headers, lease_id,
        call_index` — everything except the arena-minted `cmd_id`, supplied
        here as a keyword). Raises `ValueError` if `action["kind"] ==
        "answer"` — an answer is never a Command (see the module docstring).
        This is a convenience for tests/demos, not something the real arena
        calls: the trusted envelope mints `cmd_id` itself and constructs the
        real `Command` on its own side of the boundary."""
        kind = action.get("kind")
        if kind == "answer":
            raise ValueError(
                "an 'answer' action never becomes a Command (kit/loop/agent.py: "
                "\"an answer is not a tool call routed to a server\") — do not "
                "route it through Gateway.decide at all"
            )
        return cls(
            cmd_id=cmd_id,
            kind=kind,
            raw=action["raw"],
            server=action["server"],
            tool=action["tool"],
            args=dict(action.get("args", {})),
            fields=tuple(action.get("fields", ())),
            headers=dict(action.get("headers", {})),
            lease_id=action.get("lease_id"),
            call_index=action.get("call_index", 0),
        )

    def to_dict(self) -> dict:
        return {
            "cmd_id": self.cmd_id,
            "kind": self.kind,
            "raw": self.raw,
            "server": self.server,
            "tool": self.tool,
            "args": dict(self.args),
            "fields": list(self.fields),
            "headers": dict(self.headers),
            "lease_id": self.lease_id,
            "call_index": self.call_index,
        }


@dataclass(frozen=True, slots=True)
class Decision:
    """CONTRACTS.md 4.1, field for field.

    Validated strictly (`__post_init__`) because a *structurally* invalid
    `Decision` is charged exactly like a raised exception — CONTRACTS.md
    4.1's charging table: "malformed Decision (schema-invalid) -> 2 cr
    penalty, command denied." Failing loudly HERE, in your own process
    during development, is strictly better than discovering it live in a
    duel as an unexplained penalty.

    `verdict == "deny"` requires a non-empty `reason` (CONTRACTS.md 4.1:
    "required when verdict == 'deny'; shown in the combat log") and
    forbids `call` — a real denial has nothing left to carry out.
    `verdict` in `("forward", "rewrite")` requires `call` to be set — the
    arena executes exactly that `ToolCall`, nothing else, per the trusted
    envelope's whole point (see the module docstring)."""

    verdict: str  # "forward" | "deny" | "rewrite" — see DECISION_VERDICTS
    reason: str | None = None
    call: "ToolCall | None" = None
    quarantine: bool = False
    note: str | None = None

    def __post_init__(self) -> None:
        if self.verdict not in DECISION_VERDICTS:
            raise ValueError(
                f"Decision.verdict must be one of {sorted(DECISION_VERDICTS)}, got {self.verdict!r}"
            )
        if self.verdict == "deny":
            if not isinstance(self.reason, str) or not self.reason.strip():
                raise ValueError("Decision.verdict=='deny' requires a non-empty 'reason'")
            if self.call is not None:
                raise ValueError("Decision.verdict=='deny' must not carry a 'call' — there is nothing to run")
        else:  # forward | rewrite
            if self.call is None:
                raise ValueError(f"Decision.verdict=={self.verdict!r} requires 'call' to be set")
            if _TOOLCALL_AVAILABLE and not isinstance(self.call, ToolCall):
                raise ValueError(
                    f"Decision.call must be a kit.mcp.types.ToolCall instance, got {type(self.call).__name__}"
                )
        if not isinstance(self.quarantine, bool):
            raise ValueError(f"Decision.quarantine must be a bool, got {self.quarantine!r}")
        if self.note is not None and not isinstance(self.note, str):
            raise ValueError(f"Decision.note must be a str or None, got {self.note!r}")

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict,
            "reason": self.reason,
            "call": self.call.to_dict() if self.call is not None and hasattr(self.call, "to_dict") else self.call,
            "quarantine": self.quarantine,
            "note": self.note,
        }


@runtime_checkable
class GatewayContext(Protocol):
    """CONTRACTS.md 4.2 — "read-only, arena-provided". Note what this is
    NOT: unlike `Command`/`Decision` above, CONTRACTS.md writes this as a
    plain `class`, not a `@dataclass(frozen=True, ...)`. That is a real
    signal, not an inconsistency — a `GatewayContext` is a LIVE VIEW the
    arena owns and updates as the duel actually progresses (credits get
    spent, rounds advance, leases mint and expire), not a frozen snapshot
    you receive once. This file therefore models it as a structural
    `Protocol`: the arena hands your `Gateway.__init__` some object shaped
    like this (see `agent/telemetry.py`'s `RecordingGatewayContext` for a
    concrete example your own tests/demos can build), and you read
    `self.ctx.<field>` fresh every time `decide()` runs — NEVER cache a
    context field into a local variable across calls; if you need memory
    that persists across the duel (a cache, a budget tracker, drift notes),
    keep it on YOUR OWN `Gateway` instance attributes instead (CONTRACTS.md
    4.3: "Gateway is instantiated once per duel... persists across the 10
    rounds").

    There is deliberately no `execute` method here — see the module
    docstring's "WHY THERE IS NO execute() METHOD" section. `emit` is the
    ONE thing you can do that has any side effect at all, and it writes to
    `own_telemetry` only: your own side, never scored, never shown to the
    opponent (`agent/telemetry.py` wraps it)."""

    act: str  # "learner:sv-0417" — WHOM YOU SERVE. Authority derives from this.
    sub: str  # "agent:vlearn-tutor" — what you are. NOT authority.
    scopes: frozenset[str]  # {"wiki.read"}; writes need "wiki.write:progress" etc.
    credits: int  # remaining this duel
    round: int
    call_index: int
    leases: tuple[str, ...]  # live lease ids, arena-tracked
    history: tuple[Mapping[str, Any], ...]  # YOUR OWN prior (Command, Decision, outcome) triples this duel

    def emit(self, name: str, **payload: Any) -> None: ...


# ---------------------------------------------------------------------------
# Constants for write-tool enforcement and A2A server sets
# ---------------------------------------------------------------------------

# Tools that require If-Match + Idempotency-Key before they may execute
_WRITE_TOOLS: frozenset[tuple[str, str]] = frozenset({
    ("content", "flag_stale_slide"),
    ("content", "file_content_bug"),
    ("progress", "record_mastery"),
})

# A2A servers that require an admitted agent card before any call
_A2A_SERVERS: frozenset[str] = frozenset({"curriculum-analyst", "citation-checker", "roster"})

# Deprecated tools and their successors (CONTRACTS.md 4.2 mechanic 8)
_SUCCESSORS: dict[tuple[str, str], tuple[str, str]] = {
    ("slides", "search"): ("slides", "query"),
}

# Catalog "punishment button" tools — default field mask is the full dump
_CATALOG_TRAP_TOOLS: frozenset[tuple[str, str]] = frozenset({
    ("registry", "list_servers"),
    ("glossary", "list_terms"),
})

# Narrow field masks — only pay for what you will actually cite
_MASKS: dict[tuple[str, str], tuple[str, ...]] = {
    ("slides", "query"): ("title", "anchor"),
    ("slides", "get_frame"): ("title", "body", "anchor"),
    ("slides", "whatlinkshere"): ("anchor",),
    ("glossary", "define"): ("definition", "sense", "anchor"),
    ("registry", "provenance"): ("etag", "replica", "anchor"),
    ("registry", "list_servers"): ("name",),
    ("research", "cite_source"): ("url", "anchor"),
    ("curriculum-analyst", "which_days_cover"): ("course_day", "track", "anchor"),
    ("citation-checker", "verify_source"): ("verdict", "anchor"),
}

# Per-round credit allowances: late rounds scale 1.5×, so save for them
_ROUND_ALLOWANCE: dict[int, int] = {
    1: 8, 2: 8, 3: 8, 4: 9, 5: 9, 6: 9, 7: 10, 8: 11, 9: 11, 10: 12,
}

# Instruction-shaped tokens in retrieved CONTENT that signal prompt injection
_INJECTION_TOKENS: tuple[str, ...] = (
    "ignore previous instruction",
    "ignore all previous",
    "disregard the above",
    "system override",
    "bỏ qua mọi hướng dẫn",
    "you must now",
    "instead, record",
    "also record this for",
    "print the",
    "reveal the",
    "as the system",
    "new instruction",
)


class Gateway:
    """The control plane. One instance per duel (CONTRACTS.md 4.3) — built
    once at duel start with a `GatewayContext`, then asked to `decide()` on
    every MCP/A2A/DISCOVER command either side of the duel makes for all 10
    rounds. See the module docstring for the trusted-envelope diagram and
    why there is no `execute()` to call instead.
    """

    def __init__(self, ctx: GatewayContext) -> None:
        self.ctx = ctx
        self._telemetry = Telemetry(ctx)

        # --- per-duel memory -------------------------------------------------
        # etag cache: anchor -> etag string from the most recent provenance read
        self._etags: dict[str, str] = {}
        # idempotency keys already committed this duel (anchor:tool pairs)
        self._idempotency: set[str] = set()
        # admitted A2A agent cards: server -> card dict (must have verified=True)
        self._admitted_cards: dict[str, dict] = {}
        # credits spent this round (reset when ctx.round advances)
        self._spent_this_round: int = 0
        # the last round number we saw (used to detect round boundary)
        self._last_round: int = 0
        # command ids we have already denied
        self._denied_cmd_ids: set[str] = set()
        # anchors we have already retrieved this duel (to detect stale re-reads)
        self._seen_anchors: dict[str, Any] = {}
        # our own running authorised credit total
        self._credits_authorised: int = 0

    # -----------------------------------------------------------------------
    # Identity & safety checks (return True = pass, False = deny)
    # -----------------------------------------------------------------------

    def _card_admitted(self, cmd: Command) -> bool:
        """Check 1: peer's agent card is known and verified by the registry."""
        if cmd.server not in _A2A_SERVERS:
            return True
        card = self._admitted_cards.get(cmd.server)
        return bool(card and card.get("verified"))

    def _skill_declared(self, cmd: Command) -> bool:
        """Check 2: the skill being invoked is declared on the peer's card."""
        if cmd.server not in _A2A_SERVERS:
            return True
        card = self._admitted_cards.get(cmd.server) or {}
        skills = set(card.get("skills") or ())
        return cmd.tool in skills

    def _audience_matches(self, cmd: Command) -> bool:
        """Check 3: delegation token's aud matches the server being called."""
        aud = cmd.headers.get("aud") or cmd.headers.get("Aud")
        if aud is None:
            # A call with no aud is only acceptable for non-A2A servers
            return cmd.server not in _A2A_SERVERS
        return aud in (cmd.server, f"mcp:{cmd.server}", f"a2a:{cmd.server}")

    def _act_owns_target(self, cmd: Command) -> bool:
        """Check 4: any write/delegation target must belong to ctx.act.
        Authority derives from act (whom you serve), never from sub (what you are).
        """
        act = getattr(self.ctx, "act", None)
        if not act:
            return True
        for key in ("learner", "learner_id", "target", "subject"):
            target = cmd.args.get(key)
            if target and str(target) != str(act):
                return False
        return True

    def _routes_on_header(self, cmd: Command) -> bool:
        """Routing must come from the header, never from the request body.
        A route in cmd.args is a body-route — the header_spoof / replica_flip
        failure mode.
        """
        return not (
            cmd.args.get("route")
            or cmd.args.get("_route")
            or cmd.args.get("replica")
        )

    def _instruction_in_content(self, cmd: Command) -> bool:
        """Detect prompt-injection tokens inside retrieved content passed as args."""
        blob = " ".join(str(v) for v in cmd.args.values()).lower()
        return any(token in blob for token in _INJECTION_TOKENS)

    # -----------------------------------------------------------------------
    # Budget tracking
    # -----------------------------------------------------------------------

    def _within_budget(self, cmd: Command) -> bool:
        """True if we have room in this round's allowance."""
        rnd = getattr(self.ctx, "round", 0) or 0
        if rnd != self._last_round:
            # New round: reset per-round counter
            self._last_round = rnd
            self._spent_this_round = 0
        allowance = _ROUND_ALLOWANCE.get(rnd, 9)
        return self._spent_this_round < allowance

    # -----------------------------------------------------------------------
    # Field-mask enforcement
    # -----------------------------------------------------------------------

    def _safe_fields(self, cmd: Command) -> tuple[str, ...]:
        """Return a narrow, cost-controlled field mask.

        If the command asks for everything ('*') on a catalog-trap tool,
        rewrite to a narrow mask to avoid budget exhaustion.
        If the command specifies fields, honour them (they paid to ask for them).
        Otherwise apply our own narrow default mask.
        """
        server, tool = cmd.server, cmd.tool
        fields = cmd.fields

        # Catalog trap: rewrite wildcard/empty masks to narrow defaults
        if (server, tool) in _CATALOG_TRAP_TOOLS:
            if not fields or fields == ("*",):
                return _MASKS.get((server, tool), ("name",))

        # If caller specified fields, honour them (they know what they need)
        if fields:
            return fields

        # Apply our narrow default
        return _MASKS.get((server, tool), ("anchor",))

    # -----------------------------------------------------------------------
    # Write precondition enforcement
    # -----------------------------------------------------------------------

    def _check_write_preconditions(self, cmd: Command) -> tuple[bool, str]:
        """Returns (ok, reason). A write must have a fresh If-Match etag and a
        fresh Idempotency-Key; re-using either is write_violation.
        """
        anchor = str(cmd.args.get("anchor", ""))
        etag = self._etags.get(anchor)
        if not etag:
            return False, "write without a fresh If-Match etag from registry.provenance"
        key = f"{anchor}:{cmd.tool}"
        if key in self._idempotency:
            return False, "write already committed this duel (idempotency violation)"
        return True, ""

    def _commit_write(self, cmd: Command) -> tuple[str, str]:
        """Record the idempotency key and return (etag, idempotency_key)."""
        anchor = str(cmd.args.get("anchor", ""))
        etag = self._etags.get(anchor, "")
        key = f"{anchor}:{cmd.tool}"
        self._idempotency.add(key)
        return etag, key

    # -----------------------------------------------------------------------
    # Public API: fed by the loop after a call returns
    # -----------------------------------------------------------------------

    def note_provenance(self, anchor: str, etag: str) -> None:
        """Record a fresh etag for `anchor` so writes can pass the precondition."""
        self._etags[anchor] = etag

    def note_card(self, server: str, card: dict) -> None:
        """Register a verified agent card for `server`."""
        self._admitted_cards[server] = card

    # -----------------------------------------------------------------------
    # decide() — the one method the arena calls
    # -----------------------------------------------------------------------

    def decide(self, cmd: Command) -> Decision:
        """SYNCHRONOUS. PURE. NO I/O. 250 ms wall (RULES.md section 3).

        Four jobs, in order:
          JOB 1 — ROUTE: which server/replica/tool?
          JOB 2 — ADMIT: is this call worth letting through at all?
          JOB 3 — AUTHORIZE: does this call belong to whom we serve?
          JOB 4 — BUDGET: can the duel afford this, as written?
        """
        self._telemetry.decision_seen(cmd)

        # ==================================================================
        # JOB 1 — ROUTE: deprecated tool upgrade + header-based replica
        # ==================================================================

        # Upgrade deprecated tools (slides.search -> slides.query)
        target_server, target_tool = _SUCCESSORS.get(
            (cmd.server, cmd.tool), (cmd.server, cmd.tool)
        )
        rewritten = (target_server, target_tool) != (cmd.server, cmd.tool)

        # Strip any body-route injection; always route on header
        if not self._routes_on_header(cmd):
            return self.deny(cmd, "route declared in the request body, not the header (header_spoof/replica_flip)")

        # Build clean headers: strip x-mcp-body-route, keep Mcp-Replica header
        headers = {k: v for k, v in cmd.headers.items()
                   if k.lower() != "x-mcp-body-route"}
        headers.setdefault("Mcp-Replica", "w")  # prefer working replica by default

        # ==================================================================
        # JOB 2 — ADMIT: early-exit on clearly-doomed or malicious calls
        # ==================================================================

        # Block prompt-injection attempts in retrieved content
        if self._instruction_in_content(cmd):
            return self.deny(cmd, "instruction found in retrieved content (poisoned_result/faithless_peer)")

        # For A2A servers: require the card to be admitted first
        if not self._card_admitted(cmd):
            return self.deny(cmd, "peer card not admitted by the registry; call DISCOVER first")

        if not self._skill_declared(cmd):
            return self.deny(cmd, "skill not declared on the peer's admitted agent card")

        if not self._audience_matches(cmd):
            return self.deny(cmd, "delegation aud does not match the server being called")

        # ==================================================================
        # JOB 3 — AUTHORIZE: authority derives from act, never from sub
        # ==================================================================

        if not self._act_owns_target(cmd):
            return self.deny(cmd, "target is not owned by the learner in ctx.act (authority_exceeded)")

        # Write preconditions: must have fresh etag + fresh idempotency key
        if (target_server, target_tool) in _WRITE_TOOLS:
            ok, reason = self._check_write_preconditions(cmd)
            if not ok:
                return self.deny(cmd, reason)

        # ==================================================================
        # JOB 4 — BUDGET: pace spend across the 10-round duel
        # ==================================================================

        if not self._within_budget(cmd):
            return self.deny(cmd, "round credit allowance exhausted; saving for late rounds")

        # ==================================================================
        # BUILD the ToolCall and return a Decision
        # ==================================================================

        # Enforce narrow field masks (rewrite catalog traps, apply defaults)
        safe_fields = self._safe_fields(cmd)
        verdict = "rewrite"

        # For writes: inject If-Match + Idempotency-Key
        if (target_server, target_tool) in _WRITE_TOOLS:
            etag, key = self._commit_write(cmd)
            headers["If-Match"] = etag
            headers["Idempotency-Key"] = key

        # Only "rewrite" if something actually changed; otherwise "forward"
        fields_changed = safe_fields != cmd.fields
        if not rewritten and not fields_changed and headers == cmd.headers:
            verdict = "forward"

        self._spent_this_round += 1
        self._credits_authorised += 1

        call = self._build_tool_call(
            server=target_server,
            tool=target_tool,
            args=cmd.args,
            fields=safe_fields,
            headers=headers,
            lease_id=cmd.lease_id,
            call_index=cmd.call_index,
        )
        decision = Decision(verdict=verdict, call=call)
        self._telemetry.decision_made(cmd, decision)
        return decision

    def deny(self, cmd: Command, reason: str) -> Decision:
        """Helper for denying a command with a reason. `deny` costs ZERO credits."""
        self._denied_cmd_ids.add(cmd.cmd_id)
        decision = Decision(verdict="deny", reason=reason)
        self._telemetry.decision_made(cmd, decision)
        return decision

    def _build_tool_call(
        self,
        *,
        server: str,
        tool: str,
        args: dict,
        fields: tuple[str, ...],
        headers: dict,
        lease_id: str | None,
        call_index: int,
    ) -> "ToolCall":
        """Build the ToolCall the arena will execute."""
        kwargs = {
            "server": server,
            "tool": tool,
            "args": dict(args),
            "fields": fields,
            "headers": headers,
            "lease_id": lease_id,
            "call_index": call_index,
        }
        if _TOOLCALL_AVAILABLE and callable(ToolCall):
            try:
                return ToolCall(**kwargs)  # type: ignore[call-arg]
            except TypeError:
                pass
        return kwargs  # type: ignore[return-value]

    # Legacy alias used by tests
    def _to_tool_call(self, cmd: Command) -> "ToolCall":
        return self._build_tool_call(
            server=cmd.server,
            tool=cmd.tool,
            args=dict(cmd.args),
            fields=cmd.fields,
            headers=dict(cmd.headers),
            lease_id=cmd.lease_id,
            call_index=cmd.call_index,
        )


if __name__ == "__main__":
    print("=== agent.gateway: Command / Decision validation ===\n")

    good_cmd = Command(
        cmd_id="cmd:0000",
        kind="mcp",
        raw="MCP slides.get_frame anchor=Frame:3f2a9c11/w/041 fields=title,body lease=lse_7f21",
        server="slides",
        tool="get_frame",
        args={"anchor": "Frame:3f2a9c11/w/041"},
        fields=("body", "title"),
        headers={},
        lease_id="lse_7f21",
        call_index=0,
    )
    print(f"  Command constructed: {good_cmd}")
    assert good_cmd.kind == "mcp"

    print("\n  Rejection demo (each must raise ValueError):")

    def _expect_value_error(label: str, fn) -> None:
        try:
            fn()
        except ValueError as exc:
            print(f"    [{label:38}] -> ValueError: {exc}")
        else:
            raise AssertionError(f"expected ValueError for case {label!r}")

    _expect_value_error("Command.kind == 'answer'", lambda: Command(
        cmd_id="cmd:0001", kind="answer", raw="x", server="slides", tool="get_frame",
        args={}, fields=(), headers={}, lease_id=None, call_index=0,
    ))
    _expect_value_error("Decision verdict='deny' with no reason", lambda: Decision(verdict="deny"))
    _expect_value_error(
        "Decision verdict='forward' with no call", lambda: Decision(verdict="forward")
    )
    _expect_value_error(
        "Decision verdict='deny' carrying a call",
        lambda: Decision(verdict="deny", reason="nope", call={"server": "x", "tool": "y"}),
    )
    _expect_value_error("Decision verdict='?' unknown", lambda: Decision(verdict="???"))

    print("\n=== Command.from_action_dict — real canonicaliser integration ===\n")
    if _canonicalise_action is None:
        print("  kit.loop.agent not importable yet — skipping the live canonicaliser demo")
        demo_commands: list[Command] = [good_cmd]
    else:
        raw_actions = [
            "MCP registry.provenance anchor=Frame:3f2a9c11/w/041 fields=etag",
            'MCP slides.query q="streamable http replaces http+sse" fields=title,body',
            "A2A curriculum-analyst.which_days_cover concept=Concept:streamable-http fields=anchor,course_day,track",
            "DISCOVER registry.list_servers fields=name",
        ]
        demo_commands = []
        for i, raw in enumerate(raw_actions):
            action = _canonicalise_action(raw, call_index=i)
            cmd = Command.from_action_dict(action, cmd_id=f"cmd:{i:04d}")
            print(f"  {raw!r}\n    -> {cmd.kind}: {cmd.server}.{cmd.tool} fields={cmd.fields}")
            demo_commands.append(cmd)
        assert {c.kind for c in demo_commands} == {"mcp", "a2a", "discover"}

        answer_action = _canonicalise_action(
            'ANSWER {"text": "day 26, track P2T2"}', call_index=None
        )
        try:
            Command.from_action_dict(answer_action, cmd_id="cmd:9999")
        except ValueError as exc:
            print(f"\n  an 'answer' action correctly refuses to become a Command: {exc}")
        else:
            raise AssertionError("expected ValueError for an 'answer' action")

    print("\n=== Gateway.decide — hardened gateway ===\n")
    ctx = RecordingGatewayContext(
        act="learner:sv-0401",
        sub="agent:demo-team",
        scopes=frozenset({"wiki.read"}),
        credits=100,
        round=1,
        call_index=0,
        leases=(),
        history=(),
    )
    assert isinstance(ctx, GatewayContext), "RecordingGatewayContext must structurally satisfy GatewayContext"
    gw = Gateway(ctx)
    for cmd in demo_commands:
        decision = gw.decide(cmd)
        print(f"  decide({cmd.server}.{cmd.tool}) -> verdict={decision.verdict!r} quarantine={decision.quarantine}")

    print(f"\n=== Gateway.deny — the free-abstention path ===\n")
    denial = gw.deny(demo_commands[0], reason="demo: withholding pending a fresher registry.provenance read")
    print(f"  gw.deny(...) -> verdict={denial.verdict!r} reason={denial.reason!r} call={denial.call!r}")
    assert denial.verdict == "deny"
    assert denial.call is None
    assert demo_commands[0].cmd_id in gw._denied_cmd_ids

    print(f"\n=== own_telemetry — recorded on YOUR side only, never shown to the opponent ===\n")
    print(f"  {len(ctx.events)} events recorded on this ctx this run:")
    for ev in ctx.events:
        print(f"    {ev['name']}: {sorted(ev['payload'].keys())}")

    print("\nAll agent/gateway.py demos passed.")
