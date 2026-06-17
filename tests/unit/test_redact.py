from agent_triage.observability import redact


def test_redact_email() -> None:
    assert "user@example.com" not in redact("Contact: user@example.com please")
    assert "[REDACTED_EMAIL]" in redact("Contact: user@example.com please")


def test_redact_phone() -> None:
    assert "[REDACTED_PHONE]" in redact("Call 555-123-4567 today")
    assert "[REDACTED_PHONE]" in redact("Call 555.123.4567 today")
    assert "[REDACTED_PHONE]" in redact("Call 555 123 4567 today")


def test_redact_phone_bare_ten_digits() -> None:
    assert redact("Call 5551234567 today") == "Call [REDACTED_PHONE] today"


def test_redact_phone_parenthesized_area_code() -> None:
    assert redact("Call (555) 123-4567 today") == "Call [REDACTED_PHONE] today"
    assert redact("Call (555)123-4567 today") == "Call [REDACTED_PHONE] today"


def test_redact_phone_country_code() -> None:
    assert redact("Call +1 555-123-4567 today") == "Call [REDACTED_PHONE] today"
    assert redact("Call +1-555-123-4567 today") == "Call [REDACTED_PHONE] today"
    assert redact("Call +1 (555) 123-4567 today") == "Call [REDACTED_PHONE] today"
    assert redact("Call 1-555-123-4567 today") == "Call [REDACTED_PHONE] today"
    assert redact("Call 1 555 123 4567 today") == "Call [REDACTED_PHONE] today"


def test_redact_phone_nine_digit_number_not_matched() -> None:
    assert redact("Order id 123456789 confirmed") == "Order id 123456789 confirmed"


def test_redact_ssn() -> None:
    assert "123-45-6789" not in redact("SSN: 123-45-6789")
    assert "[REDACTED_SSN]" in redact("SSN: 123-45-6789")


def test_redact_account() -> None:
    assert "[REDACTED_ACCOUNT]" in redact("Your account #1234567890 is overdue")


def test_redact_passthrough_when_clean() -> None:
    assert redact("Nothing sensitive here") == "Nothing sensitive here"


def test_redact_multiple_in_one_string() -> None:
    result = redact("Email me at foo@bar.com or call 555-123-4567")
    assert "[REDACTED_EMAIL]" in result
    assert "[REDACTED_PHONE]" in result
    assert "foo@bar.com" not in result
    assert "555-123-4567" not in result
