"""Right-padding check.

Under right padding, mask.sum() - 1 and the last 1 in the mask are the same
index. Gemma's tokenizer left-pads, and the old exam used sum() - 1 on that
padding, which landed on an early token. Phase 6 builds the mask itself and
refuses a batch where those two indexes differ.
"""

from __future__ import annotations


def assert_right_pad(mask: list[int]) -> int:
    """Return the last content index, or raise if this row is not right-padded."""
    if not mask or not any(mask):
        raise RuntimeError("mask has no content token")
    seen_zero = False
    for bit in mask:
        if bit not in (0, 1):
            raise RuntimeError(f"mask bit {bit} is not 0 or 1")
        if bit == 0:
            seen_zero = True
        elif seen_zero:
            raise RuntimeError("content token follows padding; row is not right-padded")
    last = len(mask) - 1 - mask[::-1].index(1)
    shortcut = sum(mask) - 1
    if shortcut != last:
        raise RuntimeError(f"right-pad assert failed: sum-1={shortcut} last-one={last}")
    return last
