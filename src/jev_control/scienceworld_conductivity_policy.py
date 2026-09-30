"""Pure visible-input primitives for a non-model conductivity measurement policy.

This module never receives an environment handle, task index, label, score, or
arbitrary API metadata. The circuit is authored from generic electrical terminal
semantics, not a task's gold action sequence. It is an executable proposal until
the separate development runner verifies it in the pinned simulator.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import re
from collections.abc import Sequence


@dataclass(frozen=True)
class ControllerView:
    task_description: str
    observation: str
    legal_actions: tuple[str, ...]

    def sha256(self) -> str:
        raw = json.dumps(asdict(self), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode()).hexdigest()


def make_controller_view(*, task_description: str, observation: str,
                         legal_actions: Sequence[str]) -> ControllerView:
    """Explicit visible fields only; do not pass the API's mixed ``info`` dict."""
    if not isinstance(task_description, str) or not isinstance(observation, str):
        raise TypeError("task description and observation must be strings")
    if isinstance(legal_actions, (str, bytes)) or not isinstance(legal_actions, Sequence):
        raise TypeError("legal actions must be a sequence of strings")
    if any(not isinstance(action, str) for action in legal_actions):
        raise TypeError("every legal action must be a string")
    return ControllerView(task_description, observation, tuple(sorted(set(legal_actions))))


@dataclass(frozen=True)
class ConductivityTask:
    target: str
    location: str
    conductive_box: str
    nonconductive_box: str


_TASK_PATTERN = re.compile(
    r"Your task is to determine if (?P<target>unknown substance [A-Z]) is electrically conductive\. "
    r"The (?P=target) is located around the (?P<location>[a-z ]+)\. "
    r"First, focus on the (?P=target)\. "
    r"If it is electrically conductive, place it in the (?P<cbox>[a-z]+ box)\. "
    r"If it is electrically nonconductive, place it in the (?P<nbox>[a-z]+ box)\.\s*"
)


def parse_task(description: str) -> ConductivityTask:
    """Read the instruction's class-to-container mapping, never its answer."""
    if not isinstance(description, str):
        raise TypeError("description must be a string")
    description = re.sub(r"\ATask Description:\s*", "", description).strip()
    match = _TASK_PATTERN.fullmatch(description)
    if match is None or match["cbox"] == match["nbox"]:
        raise ValueError("description does not match the frozen conductivity task contract")
    return ConductivityTask(match["target"], match["location"], match["cbox"], match["nbox"])


@dataclass(frozen=True)
class Endpoint:
    object_ref: str
    terminal: str | None


@dataclass(frozen=True)
class CircuitKit:
    battery: str
    indicator: str
    wires: tuple[str, str, str]


def _connection_referents(view: ControllerView) -> list[str]:
    refs = []
    for action in view.legal_actions:
        match = re.fullmatch(r"connect (.+) to (.+)", action)
        if match:
            refs.extend((match[1], match[2]))
    return refs


def _terminal_object(ref: str) -> tuple[str, str] | None:
    # Accept normal visible referents used by the generic action parser. Retain
    # any container qualifier to avoid merging equal names in different places.
    match = re.fullmatch(r"(anode|cathode|terminal [12]) (?:on|in) (.+)", ref)
    if match:
        return match[1], match[2]
    match = re.fullmatch(r"(.+) (anode|cathode|terminal [12])", ref)
    if match:
        return match[2], match[1]
    return None


def choose_kit(view: ControllerView) -> CircuitKit:
    """Pick a lexicographically fixed visible battery, indicator and three wires."""
    terminals: dict[str, set[str]] = {}
    for ref in _connection_referents(view):
        parsed = _terminal_object(ref)
        if parsed:
            terminal, obj = parsed
            terminals.setdefault(obj, set()).add(terminal)
    polarized = sorted(obj for obj, ts in terminals.items() if {"anode", "cathode"} <= ts)
    batteries = [obj for obj in polarized if re.match(r"battery(?:$| in | on )", obj)]
    indicators = [obj for obj in polarized if re.match(
        r"(?:[a-z]+ light bulb|electric motor|electric buzzer)(?:$| in | on )", obj)]
    wires = sorted(obj for obj, ts in terminals.items()
                   if {"terminal 1", "terminal 2"} <= ts
                   and re.match(r"[a-z]+ wire(?:$| in | on )", obj))
    if not batteries or not indicators or len(wires) < 3:
        raise ValueError("visible legal actions lack a complete unused circuit kit")
    return CircuitKit(batteries[0], indicators[0], tuple(wires[:3]))


def measurement_connections(task: ConductivityTask, kit: CircuitKit) -> tuple[tuple[Endpoint, Endpoint], ...]:
    """Closed series circuit, with the unknown sample on the ground branch.

    Each ordinary unknown-object reference consumes its next free hidden terminal
    through the public connection action; the policy never reads these terminals.
    """
    w1, w2, w3 = kit.wires
    return (
        (Endpoint(kit.battery, "anode"), Endpoint(w1, "terminal 1")),
        (Endpoint(w1, "terminal 2"), Endpoint(kit.indicator, "cathode")),
        (Endpoint(kit.indicator, "anode"), Endpoint(w2, "terminal 1")),
        (Endpoint(w2, "terminal 2"), Endpoint(task.target, None)),
        (Endpoint(task.target, None), Endpoint(w3, "terminal 1")),
        (Endpoint(w3, "terminal 2"), Endpoint(kit.battery, "cathode")),
    )


def _endpoint_matches(ref: str, endpoint: Endpoint) -> bool:
    if endpoint.terminal is None:
        return (ref.casefold() == endpoint.object_ref.casefold()
                or ref.casefold().startswith(endpoint.object_ref.casefold() + " in ")
                or ref.casefold().startswith(endpoint.object_ref.casefold() + " on "))
    parsed = _terminal_object(ref)
    return parsed == (endpoint.terminal, endpoint.object_ref)


def resolve_target_ref(view: ControllerView, target: str) -> str:
    """Resolve a unique visible focus referent, retaining the source target name.

    The pinned parser lowercases and chooses one unique alias per visible object.
    Its generic ``unknown substance`` alias can precede the letter-specific name.
    Accept that alias only when exactly one is visible and no focus referent
    identifies a different unknown substance. The evaluator still uses ``target``.
    """
    refs = [a[len("focus on "):] for a in view.legal_actions if a.startswith("focus on ")]
    exact = [ref for ref in refs if _endpoint_matches(ref, Endpoint(target, None))]
    if len(exact) == 1:
        return exact[0]
    if len(exact) > 1:
        raise ValueError("specific target focus referent is ambiguous")
    generic = [ref for ref in refs if re.fullmatch(r"unknown substance(?: (?:in|on) .+)?", ref.casefold())]
    named = [ref for ref in refs if re.match(r"unknown substance [a-z](?:$| (?:in|on) )", ref.casefold())]
    if len(generic) == 1 and not named:
        return generic[0]
    raise ValueError("task target has no unambiguous visible focus referent")


def select_connection(view: ControllerView, pair: tuple[Endpoint, Endpoint]) -> str:
    resolved = tuple(Endpoint(resolve_target_ref(view, ep.object_ref), None)
                 if ep.terminal is None and re.fullmatch(r"unknown substance [A-Za-z]", ep.object_ref)
                 else ep for ep in pair)
    for action in view.legal_actions:
        match = re.fullmatch(r"connect (.+) to (.+)", action)
        if match and any(
            _endpoint_matches(match[1], a) and _endpoint_matches(match[2], b)
            for a, b in (pair, tuple(reversed(pair)), resolved, tuple(reversed(resolved)))
        ):
            return action
    raise ValueError("required connection is unavailable as a legal visible action")


def indicator_on(observation: str, indicator: str) -> bool | None:
    """Parse a named visible device state; ambiguous or absent readings abstain."""
    # Device descriptions state a base name, whereas legal referents may include
    # container qualifiers. If the base appears twice, abstain even if states agree.
    base = re.split(r" (?:in|on) ", indicator, maxsplit=1)[0]
    states = re.findall(r"\b" + re.escape(base) + r", which is (on|off)\b", observation)
    return states[0] == "on" if len(states) == 1 else None


def select_focus(view: ControllerView, task: ConductivityTask) -> str:
    target_ref = resolve_target_ref(view, task.target)
    return next(a for a in view.legal_actions if a == "focus on " + target_ref)


def select_answer(view: ControllerView, task: ConductivityTask, conductive: bool) -> str:
    if type(conductive) is not bool:
        raise TypeError("classification must be a Boolean")
    target_ref = resolve_target_ref(view, task.target)
    box = task.conductive_box if conductive else task.nonconductive_box
    candidates = []
    for action in view.legal_actions:
        match = re.fullmatch(r"move (.+) to (.+)", action)
        if match and any(_endpoint_matches(match[1], Endpoint(ref, None))
                         for ref in (target_ref, task.target)):
            if match[2] == box or match[2].startswith(box + " in "):
                candidates.append(action)
    if not candidates:
        raise ValueError("chosen answer has no visible legal placement action")
    return candidates[0]
