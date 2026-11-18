"""Parameterized pulse programs compiled to the public PULSAR event objects."""

import ast
import math
import operator
import re
from dataclasses import dataclass
from functools import lru_cache
from numbers import Real
from pathlib import Path


class SequenceError(ValueError):
    """Invalid pulse program, with source name and one-based line number."""

    def __init__(self, source, line, message):
        self.source, self.line = source, line
        super().__init__(f"{source}:{line}: {message}")


@lru_cache(maxsize=1)
def _units():
    from spectrochempy.core.units import ur

    return {
        name: ur.Quantity(1, name)
        for name in ("s", "ms", "us", "ns", "Hz", "kHz", "MHz", "deg", "rad")
    }


_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}
_OPTIONS = {
    "rf": "Hz",
    "phase": "deg",
    "offset": "Hz",
    "partner_rf": "Hz",
    "partner_phase": "deg",
    "partner_offset": "Hz",
}
_OPTION = re.compile(r"\b([A-Za-z_]\w*)\s*=")
_NAME = re.compile(r"[A-Za-z_]\w*\Z")
_NUMBER_UNIT = re.compile(r"(?<=\d)\s+(?=(?:s|ms|us|ns|Hz|kHz|MHz|deg|rad)\b)")


def _expression(text):
    tree = ast.parse(_NUMBER_UNIT.sub("*", text.strip()), mode="eval")
    if sum(1 for _ in ast.walk(tree)) > 128:
        raise ValueError("expression is too large")
    allowed = (
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Name,
        ast.Load,
        ast.Constant,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.UAdd,
        ast.USub,
    )
    if any(not isinstance(node, allowed) for node in ast.walk(tree)):
        raise ValueError("only names, numbers, +, -, *, / and parentheses are allowed")
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and (
            isinstance(node.value, bool) or not isinstance(node.value, (int, float))
        ):
            raise TypeError("only real numeric literals are allowed")
    return tree.body


def _evaluate(node, values):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        if node.id in _units():
            return _units()[node.id]
        if node.id not in values:
            raise ValueError(f"missing parameter '{node.id}'")
        value = values[node.id]
        if isinstance(value, str):
            value = _evaluate(_expression(value), {})
            values[node.id] = value
        if isinstance(value, bool) or not (
            isinstance(value, Real) or hasattr(value, "to")
        ):
            raise ValueError(
                f"parameter '{node.id}' must be a scalar number or quantity"
            )
        return value
    if isinstance(node, ast.UnaryOp):
        value = _evaluate(node.operand, values)
        return -value if isinstance(node.op, ast.USub) else value
    return _OPERATORS[type(node.op)](
        _evaluate(node.left, values), _evaluate(node.right, values)
    )


def _scalar(value, unit=None):
    if unit:
        if not hasattr(value, "to"):
            # Bare phases conventionally mean degrees; timed/RF fields require units.
            if unit != "deg":
                raise ValueError(f"explicit units compatible with {unit} are required")
        else:
            value = value.to(unit).magnitude
    elif hasattr(value, "to"):
        value = value.to("").magnitude
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(value)
    ):
        raise ValueError("expected a finite real scalar")
    return float(value)


@dataclass(frozen=True)
class _Instruction:
    line: int
    kind: str
    argument: object
    options: tuple = ()
    body: tuple = ()


def _parse(text, source):
    instructions, stack = [], []
    for number, raw in enumerate(text.splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        try:
            parts = line.split(maxsplit=1)
            kind = parts[0]
            rest = parts[1] if len(parts) == 2 else ""
            if kind == "end" and not rest:
                if not stack:
                    raise ValueError("unmatched end")
                parent, start, count = stack.pop()
                parent.append(
                    _Instruction(start, "repeat", count, body=tuple(instructions))
                )
                instructions = parent
            elif kind == "repeat":
                if len(stack) >= 16:
                    raise ValueError("at most 16 nested repeats are supported")
                count = _expression(rest)
                stack.append((instructions, number, count))
                instructions = []
            elif kind == "set":
                name, separator, expression = rest.partition("=")
                name = name.strip()
                if not separator or not _NAME.fullmatch(name) or name in _units():
                    raise ValueError(
                        "set requires a writable name followed by = expression"
                    )
                instructions.append(
                    _Instruction(number, kind, (name, _expression(expression)))
                )
            elif kind in {"pulse", "delay", "coherence"}:
                matches = list(_OPTION.finditer(rest))
                argument = _expression(rest[: matches[0].start()] if matches else rest)
                options = {}
                for i, match in enumerate(matches):
                    name = match[1]
                    if kind != "pulse" or name not in _OPTIONS or name in options:
                        raise ValueError(
                            f"unsupported or repeated {kind} option '{name}'"
                        )
                    stop = matches[i + 1].start() if i + 1 < len(matches) else len(rest)
                    options[name] = _expression(rest[match.end() : stop])
                instructions.append(
                    _Instruction(number, kind, argument, tuple(options.items()))
                )
            else:
                raise ValueError(f"unknown instruction '{kind}'")
        except (ValueError, TypeError, SyntaxError, RecursionError) as exc:
            raise SequenceError(source, number, str(exc)) from exc
    if stack:
        raise SequenceError(source, stack[-1][1], "repeat has no matching end")
    return tuple(instructions)


def _event(instruction, values):
    from spectrochempy_nmr.pulsar import Coherence, Delay, Pulse

    value = _evaluate(instruction.argument, values)
    if instruction.kind == "coherence":
        order = _scalar(value)
        if not order.is_integer():
            raise ValueError("coherence order must be an integer")
        return Coherence(int(order))
    duration = _scalar(value, "s")
    if duration < 0 or (instruction.kind == "pulse" and duration == 0):
        raise ValueError(
            "pulse duration must be positive; delay duration must be nonnegative"
        )
    if instruction.kind == "delay":
        return Delay(duration)
    overrides = dict(instruction.options)
    if not any(name in overrides or name in values for name in ("rf", "partner_rf")):
        raise ValueError(
            "pulse requires rf or partner_rf, as an option or a set parameter"
        )
    fields = {}
    for name, unit in _OPTIONS.items():
        if name in overrides:
            value = _evaluate(overrides[name], values)
        elif name in values:
            value = _evaluate(ast.Name(id=name), values)
        else:
            value = 0 if unit == "deg" else 0 * _units()[unit]
        fields[name] = _scalar(value, unit)
        if name in {"rf", "partner_rf"} and fields[name] < 0:
            raise ValueError(f"{name} must be nonnegative")
    return Pulse(
        duration,
        fields["rf"],
        fields["phase"],
        fields["offset"],
        fields["partner_rf"],
        fields["partner_phase"],
        fields["partner_offset"],
    )


@dataclass(frozen=True)
class PulseProgram:
    """Parsed sequence, reusable with different parameter mappings.

    Create with :func:`parse_sequence` or :func:`read_sequence`. Each build
    starts with fresh state. Assignments persist across loop iterations;
    per-pulse overrides do not change that state. A pulse describes one
    common interval on the observed and partner RF channels.
    """

    source: str
    _instructions: tuple

    def build(self, parameters=None, *, max_steps=100000):
        """Return a tuple of PULSAR events after binding and validation.

        Supply durations/frequencies as quantities or strings such as
        ``"8 us"`` and ``"10 kHz"``. Numbers are accepted for phases
        (degrees), coherence orders and loop counts. Expressions support
        real arithmetic and units. At most ``max_steps`` instructions and
        loop iterations may be expanded; the default is 100,000.
        """
        if (
            isinstance(max_steps, bool)
            or not isinstance(max_steps, int)
            or max_steps <= 0
        ):
            raise ValueError("max_steps must be a positive integer")
        values = dict(parameters or {})
        if any(
            not isinstance(name, str) or not _NAME.fullmatch(name) or name in _units()
            for name in values
        ):
            raise SequenceError(self.source, 1, "invalid or reserved parameter name")
        events, remaining = [], max_steps

        def execute(instructions):
            nonlocal remaining
            for instruction in instructions:
                try:
                    remaining -= 1
                    if remaining < 0:
                        raise ValueError("expanded program exceeds max_steps")
                    if instruction.kind == "set":
                        name, expression = instruction.argument
                        value = _evaluate(expression, values)
                        _scalar(value if not hasattr(value, "to") else value.magnitude)
                        if name in _OPTIONS:
                            scalar = _scalar(value, _OPTIONS[name])
                            if name in {"rf", "partner_rf"} and scalar < 0:
                                raise ValueError(f"{name} must be nonnegative")
                        values[name] = value
                    elif instruction.kind == "repeat":
                        count = _scalar(_evaluate(instruction.argument, values))
                        if count < 0 or not count.is_integer():
                            raise ValueError(
                                "repeat count must be a nonnegative integer"
                            )
                        if count > remaining:
                            raise ValueError("repeat expansion exceeds max_steps")
                        for _ in range(int(count)):
                            remaining -= 1
                            execute(instruction.body)
                    else:
                        events.append(_event(instruction, values))
                except SequenceError:
                    raise
                except (
                    ValueError,
                    TypeError,
                    ArithmeticError,
                    RecursionError,
                    SyntaxError,
                ) as exc:
                    raise SequenceError(
                        self.source, instruction.line, str(exc)
                    ) from exc

        execute(self._instructions)
        return tuple(events)


def parse_sequence(text, *, source="<sequence>"):
    """Parse sequence text once; parameters are bound by ``program.build``.

    Instructions: ``set name=expression``, ``repeat expression``/``end``,
    ``pulse duration`` with optional channel settings, ``delay duration``, and
    ``coherence order``. Comments start with ``#``. Whitespace indentation
    is cosmetic; repeat blocks terminate with ``end``.
    """
    if not isinstance(text, str):
        raise TypeError("sequence text must be a string")
    if len(text) > 1024**2:
        raise SequenceError(source, 1, "sequence source exceeds 1 MiB")
    return PulseProgram(str(source), _parse(text, str(source)))


def read_sequence(path):
    """Read a UTF-8 pulse program; file suffix is unrestricted."""
    path = Path(path)
    with path.open(encoding="utf-8") as stream:
        text = stream.read(1024**2 + 1)
    return parse_sequence(text, source=str(path))
