"""Script-owned labels for the two generated tasks.

The writer model fills in prose. It never chooses `target`.
"""

from __future__ import annotations

import random

from phase1.rows import TEAM_OPTIONS, YES_NO, contains_word, make_row, one_hot

REFUND_SCENARIOS = [
    ("billing", "charged twice", "The customer was charged twice for one order and wants the extra payment returned."),
    ("billing", "refund the shipping", "The customer wants the shipping fee returned because the box arrived crushed."),
    ("billing", "wrong amount", "The customer was billed the wrong amount and wants the charge corrected."),
    ("billing", "cancel the renewal", "A subscription renewed on its own and the customer wants that charge returned."),
    ("tech", "app crashes", "The mobile app crashes every time the customer opens the inbox."),
    ("tech", "error code", "Checkout shows an error code and the customer cannot pay."),
    ("tech", "will not load", "The settings page will not load on the customer's laptop."),
    ("tech", "reset link", "The password reset link never arrives."),
    ("sales", "pricing for annual", "The customer wants the price of an annual plan before buying."),
    ("sales", "upgrade my plan", "The customer wants to move from the free plan to a paid plan."),
    ("sales", "a written quote", "The customer wants a written quote for 40 seats."),
    ("sales", "student discount", "The customer is asking if a student discount exists."),
]

PRODUCTS = ["canvas shoes", "desk lamp", "wireless headset", "coffee grinder", "yoga mat", "keyboard"]

FACTS = [
    "Order 1842 shipped on Monday.",
    "The locker code is 4419.",
    "The plan name is Harbor.",
    "The spare key is in drawer B.",
    "The courier was Northwind.",
    "The meeting room is Cedar.",
    "The invoice number is 7730.",
    "The backup finished at 2 am.",
    "The warranty lasts 18 months.",
    "The pickup desk is on floor 3.",
    "The guest network is named Pebble.",
    "The replacement part is a gasket.",
    "The store closes at 7 pm.",
    "The driver name is Ruiz.",
    "The coupon code is MAPLE.",
    "The lab door uses badge 12.",
    "The shipment weighs 4 pounds.",
    "The account manager is Chen.",
    "The palette color is moss.",
    "The seminar starts at 9 am.",
    "The crate count is 6.",
    "The parking stall is D4.",
    "The translator is Idris.",
    "The sensor serial is K-18.",
]


def refund_specs(count: int) -> list[dict]:
    specs = []
    for index in range(count):
        team, phrase, brief = REFUND_SCENARIOS[index % len(REFUND_SCENARIOS)]
        specs.append(
            {
                "id": f"refund-{index:05d}",
                "kind": "refund",
                "team": team,
                "phrase": phrase,
                "brief": brief,
                "product": PRODUCTS[index % len(PRODUCTS)],
                "order": 1000 + index,
            }
        )
    return specs


def passage_specs(count: int, rng: random.Random) -> list[dict]:
    specs = []
    for index in range(count):
        included = rng.sample(FACTS, 3)
        answer_yes = index % 2 == 0
        if answer_yes:
            asked = included[index % 3]
        else:
            asked = rng.choice([fact for fact in FACTS if fact not in included])
        specs.append(
            {
                "id": f"passage-{index:05d}",
                "kind": "passage",
                "included": included,
                "asked": asked,
                "answer": "yes" if answer_yes else "no",
            }
        )
    return specs


def refund_prompt(spec: dict) -> str:
    return (
        "Write a short customer email, 2 to 4 sentences.\n"
        f"Situation: {spec['brief']}\n"
        f"Product: {spec['product']}\n"
        f"Order number: {spec['order']}\n"
        f'Include this exact phrase: "{spec["phrase"]}"\n'
        "Do not name a department. Do not use the words billing, tech, or sales.\n"
        "Output only the email."
    )


def passage_prompt(spec: dict) -> str:
    lines = "\n".join(f"- {fact}" for fact in spec["included"])
    return (
        "Write a two-sentence support note that contains each of these sentences exactly:\n"
        f"{lines}\n"
        "Do not add any other fact. Output only the note."
    )


def accept_refund(text: str, spec: dict) -> bool:
    email = text.strip()
    if not (40 <= len(email) <= 900):
        return False
    if spec["phrase"].lower() not in email.lower():
        return False
    if any(contains_word(email, team) for team in TEAM_OPTIONS):
        return False
    return True


def accept_passage(text: str, spec: dict) -> bool:
    note = text.strip()
    if len(note) < 40:
        return False
    if any(fact not in note for fact in spec["included"]):
        return False
    if spec["answer"] == "no" and spec["asked"] in note:
        return False
    return True


def row_from_refund(spec: dict, text: str) -> dict:
    return make_row(
        spec["id"],
        "refund_rule",
        "choice",
        text,
        "Which team?",
        TEAM_OPTIONS,
        one_hot(TEAM_OPTIONS.index(spec["team"]), len(TEAM_OPTIONS)),
    )


def row_from_passage(spec: dict, text: str) -> dict:
    return make_row(
        spec["id"],
        "passage_answer",
        "noul",
        text,
        f"Does the passage say this: {spec['asked']}",
        YES_NO,
        one_hot(1 if spec["answer"] == "yes" else 0, 2),
    )
