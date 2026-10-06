from phase2.loss import nll, text_only_names
from phase2.prompts import render_user_prompt


def test_prompt_binds_letters_in_row_order():
    text = render_user_prompt(
        {
            "id": "r",
            "state": "I was charged twice.",
            "question": "Which team?",
            "options": ["sales", "billing", "tech"],
        }
    )
    assert "A. sales" in text
    assert "B. billing" in text
    assert "C. tech" in text


def test_loss_is_small_when_the_correct_letter_wins():
    assert nll([5.0, 0.0], [1.0, 0.0]) < 0.1


def test_loss_is_large_when_the_correct_letter_loses():
    assert nll([0.0, 5.0], [1.0, 0.0]) > 4.0


def test_text_only_rejects_a_vision_lora():
    try:
        text_only_names(["language_model.layers.0.mlp.lora_A", "vision_tower.lora_A"])
    except ValueError as error:
        assert "vision_tower" in str(error)
    else:
        raise AssertionError("vision LoRA was accepted")
