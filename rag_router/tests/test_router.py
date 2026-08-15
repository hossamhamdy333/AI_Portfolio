from router import DOMAIN_DESCRIPTIONS, route_and_query


def test_domain_descriptions_cover_all_four_domains():
    assert set(DOMAIN_DESCRIPTIONS.keys()) == {"sports", "tech", "history", "english_literature"}


def test_domain_descriptions_are_nonempty_strings():
    for domain, desc in DOMAIN_DESCRIPTIONS.items():
        assert isinstance(desc, str) and len(desc) > 20


class FakeSelection:
    def __init__(self, index):
        self.index = index


class FakeSelectorResult:
    def __init__(self, index):
        self.selections = [FakeSelection(index)]


class FakeNode:
    def __init__(self, title):
        self.node = type("N", (), {"metadata": {"title": title}})()


class FakeResponse:
    def __init__(self, text, selector_index, source_nodes):
        self._text = text
        self.metadata = {"selector_result": FakeSelectorResult(selector_index)}
        self.source_nodes = source_nodes

    def __str__(self):
        return self._text


class FakeRouter:
    def __init__(self, response):
        self._response = response

    def query(self, question):
        return self._response


def test_route_and_query_maps_selector_index_to_domain_name():
    response = FakeResponse("some answer", selector_index=2, source_nodes=[FakeNode("Some Article")])
    router = FakeRouter(response)
    domains = ["sports", "tech", "history", "english_literature"]

    answer_text, routed_domain, source_nodes = route_and_query(router, domains, "a question")

    assert answer_text == "some answer"
    assert routed_domain == "history"  # index 2
    assert source_nodes[0].node.metadata["title"] == "Some Article"


def test_route_and_query_handles_missing_selector_result():
    response = FakeResponse("some answer", selector_index=0, source_nodes=[])
    response.metadata = {}  # no selector_result at all
    router = FakeRouter(response)

    answer_text, routed_domain, source_nodes = route_and_query(router, ["sports"], "q")

    assert routed_domain is None
