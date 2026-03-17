"""
Math-to-Speech Preprocessor for TTS.

Converts mathematical expressions into natural spoken language before sending
to Edge TTS. Prevents TTS from speeding up or mispronouncing math notation.

ROOT CAUSE (Phase 1 Investigation):
- Edge TTS uses SSML internally: text is XML-escaped and wrapped in
  <speak><voice><prosody rate="...">...</prosody></voice></speak>.
- Symbol-dense or short math expressions (e.g. "r=d/2", "1/2", "C/(2π)")
  can trigger inconsistent synthesis: the neural model may compress or
  accelerate these segments.
- Converting ALL math to full natural language (e.g. "the radius equals
  the diameter divided by two") ensures consistent speech rate and
  clear pronunciation across the entire response.
"""

import re
from typing import Dict

# Number words for natural fraction reading
NUM_WORDS: Dict[int, str] = {
    0: "zero", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
    6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten",
    11: "eleven", 12: "twelve", 13: "thirteen", 14: "fourteen", 15: "fifteen",
    16: "sixteen", 17: "seventeen", 18: "eighteen", 19: "nineteen", 20: "twenty",
}

# Common fractions: spoken form (natural, not "one over two")
COMMON_FRACTIONS: Dict[str, str] = {
    "1/2": "one half", "2/2": "two halves", "1/3": "one third", "2/3": "two thirds",
    "3/3": "three thirds", "1/4": "one quarter", "3/4": "three quarters",
    "2/4": "two quarters", "4/4": "four quarters", "1/5": "one fifth",
    "2/5": "two fifths", "3/5": "three fifths", "4/5": "four fifths",
    "1/6": "one sixth", "5/6": "five sixths", "1/8": "one eighth",
    "3/8": "three eighths", "5/8": "five eighths", "7/8": "seven eighths",
    "1/10": "one tenth", "1/100": "one hundredth",
}

# Ordinal suffixes for fractions (e.g. 5/8 -> five eighths)
ORDINAL_SUFFIX: Dict[int, str] = {
    2: "half", 3: "third", 4: "quarter", 5: "fifth", 6: "sixth",
    7: "seventh", 8: "eighth", 9: "ninth", 10: "tenth",
}

# Common variable names in formulas -> spoken form (circle geometry, etc.)
VARIABLE_NAMES: Dict[str, str] = {
    "C": "the circumference", "c": "the circumference",
    "r": "the radius", "R": "the radius",
    "d": "the diameter", "D": "the diameter",
    "A": "the area", "a": "the area",
    "π": "pi", "pi": "pi",
}


def _number_to_word(n: int) -> str:
    """Convert small integers to spoken words."""
    return NUM_WORDS.get(n, str(n))


def _convert_var_to_speech(name: str) -> str:
    """Convert variable name to spoken form when in formula context."""
    return VARIABLE_NAMES.get(name, name)


def latex_to_speech(latex: str) -> str:
    """Convert LaTeX math to natural speech. Handles \\frac, \\pi, ^, etc."""
    if not latex or not isinstance(latex, str):
        return ""
    s = latex.strip()
    # Nested \frac: apply repeatedly
    for _ in range(5):
        prev = s
        s = re.sub(r"\\frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}", r"\1 divided by \2", s)
        if s == prev:
            break
    s = re.sub(r"\\times|\\cdot", " times ", s)
    s = re.sub(r"\\div", " divided by ", s)
    s = re.sub(r"\\pm", " plus or minus ", s)
    s = re.sub(r"\\mp", " minus or plus ", s)
    s = re.sub(r"\\pi\b", " pi ", s)
    s = re.sub(r"\\alpha\b", " alpha ", s)
    s = re.sub(r"\\beta\b", " beta ", s)
    s = re.sub(r"\\theta\b", " theta ", s)
    s = re.sub(r"\\infty\b", " infinity ", s)
    s = re.sub(r"\s*=\s*", " equals ", s)
    s = re.sub(r"\^2\b", " squared ", s)
    s = re.sub(r"\^3\b", " cubed ", s)
    s = re.sub(r"\^(\d+)", r" to the power of \1 ", s)
    s = re.sub(r"\\[a-zA-Z]+\s*", " ", s)
    s = re.sub(r"\\[{}()\[\]]", " ", s)
    s = re.sub(r"\\", " ", s)
    return " ".join(s.split())


def _apply_variable_mappings(text: str) -> str:
    """Replace variable names in formula context with spoken form."""
    # "is r =", "is C =" etc. -> "equals" (avoid "is the radius equals")
    text = re.sub(r"\s+is\s+r\s*=\s*", " equals ", text, flags=re.I)
    text = re.sub(r"\s+is\s+C\s*=\s*", " equals ", text)
    text = re.sub(r"\s+is\s+d\s*=\s*", " equals ", text)
    # Equation LHS: "r =", "C =", "d =" at start or after punctuation
    text = re.sub(r"(^|[\s\.])\s*r\s*=\s*", r"\1the radius equals ", text, flags=re.I)
    text = re.sub(r"(^|[\s\.])\s*C\s*=\s*", r"\1the circumference equals ", text)
    text = re.sub(r"(^|[\s\.])\s*d\s*=\s*", r"\1the diameter equals ", text)
    text = re.sub(r"(^|[\s\.])\s*A\s*=\s*", r"\1the area equals ", text)
    # Standalone π
    text = re.sub(r"\bπ\b", " pi ", text)
    return text


def _expand_expr_part(part: str) -> str:
    """Expand expression part: C->circumference, 2π->two pi, 2->two."""
    part = re.sub(r"\\pi\b", " pi ", part.strip())
    part = re.sub(r"π", " pi ", part)
    part = part.strip()
    # Variable mapping
    if part in VARIABLE_NAMES:
        return _convert_var_to_speech(part)
    # "2π" or "2 pi" -> "two pi"
    m = re.match(r"^(\d+)\s*pi\s*$", part, re.I)
    if m:
        return _number_to_word(int(m.group(1))) + " pi"
    # Plain number
    if part.isdigit():
        return _number_to_word(int(part))
    return part


def _expr_to_speech(num: str, den: str) -> str:
    """Convert numerator/denominator to spoken form."""
    num_spoken = _expand_expr_part(num)
    den_spoken = _expand_expr_part(den)
    return f" {num_spoken} divided by {den_spoken} "


def fraction_to_speech(text: str) -> str:
    """Convert plain-text fractions and division to natural speech."""
    if not text:
        return text
    s = text

    # 1. Common fractions first
    for frac, spoken in COMMON_FRACTIONS.items():
        s = re.sub(re.escape(frac), " " + spoken + " ", s)

    # 2. Remaining numeric fractions: 5/6 -> five sixths, 7/8 -> seven eighths
    def numeric_frac(m):
        num, den = int(m.group(1)), int(m.group(2))
        if den == 0:
            return " "
        num_w = _number_to_word(num)
        if den in ORDINAL_SUFFIX:
            suffix = ORDINAL_SUFFIX[den]
            plural = "s" if num != 1 else ""
            return f" {num_w} {suffix}{plural} "
        den_w = _number_to_word(den) if den <= 20 else str(den)
        return f" {num_w} divided by {den_w} "

    s = re.sub(r"\b(\d+)\s*/\s*(\d+)\b", numeric_frac, s)

    # 3. Variable/expression fractions: C/(2π), d/2, a/b -> "divided by"
    s = re.sub(r"([a-zA-Zπ\d]+)\s*/\s*\(([^)]+)\)", lambda m: _expr_to_speech(m.group(1), m.group(2)), s)
    s = re.sub(r"([a-zA-Zπ\d]+)\s*/\s*([a-zA-Zπ\d]+)", lambda m: _expr_to_speech(m.group(1), m.group(2)), s)

    return s


def _expand_multiplication(text: str) -> str:
    """Convert 2πr, 2πr, 2 pi r -> two pi times the radius."""
    # 2πr or 2 π r -> two pi times the radius
    def repl(m):
        num = _number_to_word(int(m.group(1))) if m.group(1).isdigit() else m.group(1)
        var = m.group(2).strip()
        var_spoken = _convert_var_to_speech(var) if var in VARIABLE_NAMES else var
        return f" {num} pi times {var_spoken} "
    s = re.sub(r"(\d+)\s*π\s*([a-zA-Z])\b", repl, text)
    s = re.sub(r"(\d+)\s*pi\s+([a-zA-Z])\b", repl, s)
    return s


def _expand_divided_by_numbers(text: str) -> str:
    """Expand 'divided by 2' -> 'divided by two' for natural speech."""
    def repl(m):
        n = int(m.group(1))
        return " divided by " + _number_to_word(n) + " "
    return re.sub(r"\s+divided by\s+(\d+)(?=\s|$)", repl, text)


def sanitize_math_symbols(text: str) -> str:
    """Final pass: convert any remaining math symbols to spoken form."""
    if not text:
        return text
    s = text
    # Expand "divided by 2" etc. to "divided by two"
    s = _expand_divided_by_numbers(s)
    # Variable in "X divided by": "d divided by" -> "the diameter divided by"
    for var, spoken in [("d", "the diameter"), ("D", "the diameter"), ("r", "the radius"),
                        ("C", "the circumference"), ("c", "the circumference"), ("A", "the area")]:
        s = re.sub(rf"\b{re.escape(var)}\s+divided by\s+", f" {spoken} divided by ", s)
    # = -> equals (in case not caught earlier)
    s = re.sub(r"\s*=\s*", " equals ", s)
    # × · -> times
    s = re.sub(r"[×·]", " times ", s)
    # ^2, ^3, ^n
    s = re.sub(r"\^2\b", " squared ", s)
    s = re.sub(r"\^3\b", " cubed ", s)
    s = re.sub(r"\^(\d+)", r" to the power of \1 ", s)
    # Escape angle brackets (can cause SSML parsing issues)
    s = s.replace("<", " less than ")
    s = s.replace(">", " greater than ")
    s = s.replace("≤", " less than or equal to ")
    s = s.replace("≥", " greater than or equal to ")
    s = s.replace("≠", " not equal to ")
    return " ".join(s.split())


def math_to_speech(text: str) -> str:
    """
    Full math-to-speech conversion pipeline.
    Call this on text before sending to TTS.
    """
    if not text:
        return text
    s = fraction_to_speech(text)
    s = _expand_multiplication(s)
    s = _apply_variable_mappings(s)
    s = sanitize_math_symbols(s)
    return " ".join(s.split())
