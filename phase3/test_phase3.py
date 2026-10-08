from phase3.fit import assert_calibration_row, fit_temperature, nll_at_temperature, winning_index


def test_temperature_does_not_change_the_winner():
    logits = [2.0, 0.5, -1.0]
    assert winning_index(logits) == winning_index([value / 2.0 for value in logits])
    assert winning_index(logits) == winning_index([value / 0.5 for value in logits])


def test_a_larger_temperature_helps_when_the_model_is_sure_and_wrong():
    sure_and_wrong = nll_at_temperature([0.0, 5.0], 0, 0.5)
    flatter = nll_at_temperature([0.0, 5.0], 0, 2.0)
    assert flatter < sure_and_wrong


def test_a_smaller_temperature_helps_when_the_model_is_sure_and_right():
    sharp = nll_at_temperature([5.0, 0.0], 0, 0.5)
    flat = nll_at_temperature([5.0, 0.0], 0, 2.0)
    assert sharp < flat


def test_fit_picks_the_grid_point_with_the_lower_loss():
    records = [{"logits": [0.0, 4.0], "label_index": 0}]
    fitted = fit_temperature(records, grid=[0.5, 1.0, 2.0])
    assert fitted["T"] == 2.0
    assert fitted["accuracy"] == 0.0
    assert fitted["nll"] < fitted["nll_at_1"]


def test_calibration_rejects_a_shuffled_or_held_out_row():
    try:
        assert_calibration_row(
            {
                "id": "x",
                "source": "snli",
                "type": "choice",
                "options": ["a", "b"],
                "target": [1.0, 0.0],
                "shuffled": False,
            }
        )
    except ValueError as error:
        assert "snli" in str(error)
    else:
        raise AssertionError("held-out source was accepted")
