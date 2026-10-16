import pytest

from retailassist.rag import PolicyIndex, load_chunks

QUESTIONS = {
    # Phrased without the policy's own wording, so keyword search alone would struggle.
    "how long until my money comes back": "refund-policy#refunds",
    "can I send back the clearance gloves": "refund-policy#items-that-can-t-be-returned",
    "is delivery free": "shipping-faq#what-are-the-shipping-options-and-costs",
    "the tracking says it arrived but there's nothing on my porch": "shipping-faq#my-tracking-says-delivered-but-i-don-t-have-my-package",
    "do you ship to Canada": "shipping-faq#where-do-you-ship",
}


@pytest.fixture(scope="module")
def index(tmp_path_factory):
    return PolicyIndex(tmp_path_factory.mktemp("vectordb"))


def test_chunks_have_sources_and_passages():
    chunks = load_chunks()
    assert len(chunks) == 19
    refunds = next(c for c in chunks if c.id == "refund-policy#refunds")
    assert refunds.source == "data/policies/refund-policy.md#refunds"
    assert refunds.title == "Returns & Refund Policy > Refunds"
    # Each bullet is its own passage so citations can point at it.
    assert all(p.startswith("- ") for p in refunds.passages)


@pytest.mark.parametrize("question,expected", QUESTIONS.items())
def test_semantic_search_finds_the_right_section(index, question, expected):
    assert index.search(question)[0].chunk.id == expected


def test_unrelated_questions_get_no_results(index):
    assert index.search("what's the capital of France") == []


def test_search_result_block_format(index):
    block = index.search("how long do refunds take")[0].to_search_result()
    assert block["type"] == "search_result"
    assert block["source"] == "data/policies/refund-policy.md#refunds"
    assert block["citations"] == {"enabled": True}
    assert all(item["type"] == "text" for item in block["content"])


def test_index_rebuilds_when_policies_change(tmp_path):
    policy_dir = tmp_path / "policies"
    policy_dir.mkdir()
    (policy_dir / "faq.md").write_text("# FAQ\n\n## Gift cards\n\nGift cards never expire.\n")
    index = PolicyIndex(tmp_path / "db", policy_dir=policy_dir)
    assert len(index) == 1

    (policy_dir / "faq.md").write_text("# FAQ\n\n## Gift cards\n\nGift cards never expire.\n\n"
                                       "## Price matching\n\nWe match competitors' prices.\n")
    reopened = PolicyIndex(tmp_path / "db", policy_dir=policy_dir)
    assert len(reopened) == 2
    assert reopened.search("will you match a lower price elsewhere")[0].chunk.title == "FAQ > Price matching"
