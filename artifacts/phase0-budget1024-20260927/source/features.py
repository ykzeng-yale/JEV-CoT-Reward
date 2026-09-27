"""Frozen semantic rubric v1; never provide reference answers to judges."""
SCHEMA_VERSION = "semantic-v1"
QUESTIONS = {
    "hypothesis": {"type": "noul", "instructions": "Does `latest_segment` explicitly propose a tentative hypothesis or assumption to be tested, rather than assert it as established?"},
    "contradiction": {"type": "noul", "instructions": "Does `latest_segment` assert a claim that explicitly conflicts with an established claim in `history` under the same assumptions?", "criteria": {"true": "An asserted conflict remains unresolved.", "false": "No explicit conflict, or it is a clearly marked hypothetical, correction, or retraction."}},
    "unsupported": {"type": "noul", "instructions": "Does `latest_segment` treat an unavailable observation, source, or tool result as if it had already been obtained?", "criteria": {"true": "Claims observed evidence absent from the provided record.", "false": "Evidence present, or only proposing a test or conditional assumption."}},
    "repeated_failure": {"type": "noul", "instructions": "Does `latest_segment` repeat an approach already reported to fail in `history`, without addressing the stated failure reason?"},
    "localized_error": {"type": "noul", "instructions": "Does `latest_segment` explicitly identify a specific error in the ongoing solution that can be corrected?"},
    "testable": {"type": "noul", "instructions": "Does `latest_segment` identify a concrete calculation or test that would distinguish currently active alternatives?"},
    "locally_valid": {"type": "noul", "instructions": "Are the asserted conclusions in `latest_segment` supported by the stated premises and `history`? Treat an explicitly tentative hypothesis as tentative, not as a proven claim."},
}


def judge_state(task, history, latest_segment):
    # Allowlist excludes gold answer, replay outcomes, future actions and labels.
    return {"task": task, "history": history, "latest_segment": latest_segment}
