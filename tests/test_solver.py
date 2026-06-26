from solver import parse_herd_output


def test_parse_herd_never_as_forbidden() -> None:
    parsed = parse_herd_output("Test MP Allowed\nObservation MP Never 0 10\n")
    assert parsed["verdict"] == "forbidden"
    assert parsed["allowed"] is False


def test_parse_herd_sometimes_as_allowed() -> None:
    parsed = parse_herd_output("Test MP Allowed\nObservation MP Sometimes 1 9\n")
    assert parsed["verdict"] == "allowed"
    assert parsed["allowed"] is True


def test_parse_herd_unparsed_is_unknown() -> None:
    parsed = parse_herd_output("no recognizable verdict")
    assert parsed["verdict"] == "unknown"
    assert parsed["allowed"] is None
