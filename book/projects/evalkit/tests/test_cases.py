# path: book/projects/evalkit/tests/test_cases.py
import json

import pytest

from evalkit import Dataset, DatasetError, EvalCase, check_leakage


def _cases(n: int, group_mod: int | None = None) -> list[EvalCase]:
    return [
        EvalCase(
            id=f"c{i:03d}",
            input={"q": f"question {i}"},
            expected=str(i),
            tags=["even" if i % 2 == 0 else "odd"],
            metadata={"customer": f"cust{i % group_mod}"} if group_mod else {},
        )
        for i in range(n)
    ]


def test_roundtrip_preserves_cases_header_and_hash(tmp_path):
    ds = Dataset(_cases(5), name="demo", version="3", description="d")
    path = ds.save_jsonl(tmp_path / "demo.jsonl")
    header = json.loads(path.read_text().splitlines()[0])["_dataset"]
    assert header["version"] == "3" and header["content_hash"] == ds.content_hash
    loaded = Dataset.load_jsonl(path)
    assert loaded.name == "demo" and loaded.version == "3"
    assert loaded.ids == ds.ids and loaded.content_hash == ds.content_hash


def test_load_without_header_uses_file_stem(tmp_path):
    p = tmp_path / "plain.jsonl"
    p.write_text('{"id": "a", "input": "x"}\n\n{"id": "b", "input": "y"}\n')
    ds = Dataset.load_jsonl(p)
    assert ds.name == "plain" and len(ds) == 2


def test_hash_is_order_independent_and_content_sensitive():
    cases = _cases(4)
    a = Dataset(cases, name="x")
    b = Dataset(list(reversed(cases)), name="x")
    assert a.content_hash == b.content_hash
    edited = [c.model_copy(update={"expected": "changed"}) if c.id == "c001" else c for c in cases]
    assert Dataset(edited, name="x").content_hash != a.content_hash


def test_verify_hash_detects_edits():
    ds = Dataset(_cases(3), name="x")
    ds.verify_hash(ds.content_hash[:12])
    with pytest.raises(DatasetError):
        ds.verify_hash("0" * 12)


def test_duplicate_ids_rejected():
    c = _cases(2)
    with pytest.raises(DatasetError):
        Dataset([c[0], c[1], c[0]], name="dup")


def test_bad_line_reports_location(tmp_path):
    p = tmp_path / "bad.jsonl"
    p.write_text('{"id": "a", "input": 1}\n{not json}\n')
    with pytest.raises(DatasetError, match=":2:"):
        Dataset.load_jsonl(p)


def test_split_is_deterministic_keeps_groups_together_and_stable_under_additions():
    ds = Dataset(_cases(200, group_mod=40), name="x")
    dev1, hold1 = ds.split(0.3, group_by="customer", seed=7)
    dev2, hold2 = ds.split(0.3, group_by="customer", seed=7)
    assert hold1.ids == hold2.ids
    assert 0.15 < len(hold1) / len(ds) < 0.45
    assert check_leakage(dev1, hold1, group_by="customer").clean
    # adding cases never moves existing ones
    bigger = Dataset([*ds.cases, *[c.model_copy(update={"id": f"new{c.id}"}) for c in _cases(50, 40)]], name="x")
    _, hold3 = bigger.split(0.3, group_by="customer", seed=7)
    assert set(hold1.ids) <= set(hold3.ids)


def test_check_leakage_finds_shared_groups_and_normalized_duplicates():
    a = Dataset([EvalCase(id="a1", input="How do I reset my VPN?", metadata={"doc": "d1"})], name="a")
    b = Dataset(
        [
            EvalCase(id="b1", input="how do i reset my vpn", metadata={"doc": "d9"}),
            EvalCase(id="b2", input="other", metadata={"doc": "d1"}),
        ],
        name="b",
    )
    rep = check_leakage(a, b, group_by="doc")
    assert rep.duplicate_inputs == [("a1", "b1")]
    assert rep.shared_groups == ["d1"]
    assert not rep.clean


def test_filter_slices_and_tag_counts():
    ds = Dataset(_cases(6), name="x")
    assert len(ds.filter(tags=["even"])) == 3
    assert ds.tag_counts() == {"even": 3, "odd": 3}
    assert ds.slices()["odd"] == ["c001", "c003", "c005"]
    assert len(ds.filter(lambda c: c.id.endswith("1"))) == 1
