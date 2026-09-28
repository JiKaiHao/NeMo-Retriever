# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import lancedb
import pyarrow as pa
import pytest

import nemo_retriever.ingest.index_mode as index_mode_module
from nemo_retriever.common.vdb.lancedb import LanceDB
from nemo_retriever.ingest.index_mode import resolve_ingest_index_mode, resolve_lancedb_upload_kwargs


def _create_table(uri: str, table_name: str, *, vector: bool, fts: bool) -> None:
    fields = [pa.field("text", pa.string()), pa.field("id", pa.string())]
    row = {"text": "alpha safety manual", "id": "alpha"}
    if vector:
        fields.insert(0, pa.field("vector", pa.list_(pa.float32(), 2)))
        row["vector"] = [0.1, 0.2]
    table = lancedb.connect(uri).create_table(table_name, data=[row], schema=pa.schema(fields), mode="overwrite")
    if fts:
        table.create_fts_index("text", replace=True)


@pytest.mark.parametrize(
    ("requested", "existing", "expected"),
    [
        ("auto", None, "hybrid"),
        ("dense", None, "dense"),
        ("hybrid", None, "hybrid"),
        ("sparse", None, "sparse"),
        ("auto", "dense", "dense"),
        ("dense", "dense", "dense"),
        ("hybrid", "dense", "hybrid"),
        ("auto", "hybrid", "hybrid"),
        ("hybrid", "hybrid", "hybrid"),
        ("auto", "sparse", "sparse"),
        ("sparse", "sparse", "sparse"),
    ],
)
def test_resolve_ingest_index_mode_compatible_transitions(requested, existing, expected) -> None:
    assert resolve_ingest_index_mode(requested, overwrite=False, existing_mode=existing) == expected


@pytest.mark.parametrize("requested", ["auto", "dense", "hybrid", "sparse"])
def test_resolve_ingest_index_mode_overwrite_ignores_existing_mode(requested) -> None:
    expected = "hybrid" if requested == "auto" else requested
    assert resolve_ingest_index_mode(requested, overwrite=True, existing_mode="sparse") == expected


@pytest.mark.parametrize(
    ("requested", "existing"),
    [
        ("dense", "hybrid"),
        ("sparse", "hybrid"),
        ("dense", "sparse"),
        ("hybrid", "sparse"),
        ("sparse", "dense"),
    ],
)
def test_resolve_ingest_index_mode_rejects_incompatible_append(requested, existing) -> None:
    with pytest.raises(ValueError, match="Cannot append"):
        resolve_ingest_index_mode(requested, overwrite=False, existing_mode=existing)


def test_lancedb_upload_defaults_mirror_lancedb_constructor() -> None:
    vdb = LanceDB()

    assert vdb.uri == index_mode_module._LANCEDB_DEFAULT_URI
    assert vdb.table_name == index_mode_module._LANCEDB_DEFAULT_TABLE_NAME
    assert vdb.overwrite is index_mode_module._LANCEDB_DEFAULT_OVERWRITE


@pytest.mark.parametrize("vdb_kwargs", [{}, {"overwrite": True, "table_name": "docs"}])
def test_resolve_lancedb_upload_kwargs_defaults_overwrite_to_hybrid(monkeypatch, vdb_kwargs) -> None:
    monkeypatch.setattr(
        index_mode_module,
        "inspect_existing_lancedb_mode",
        lambda *_: pytest.fail("overwrite must not inspect the existing table"),
    )

    assert resolve_lancedb_upload_kwargs(vdb_kwargs) == {**vdb_kwargs, "hybrid": True}


@pytest.mark.parametrize("explicit", [{"hybrid": False}, {"hybrid": True}, {"sparse": True}])
def test_resolve_lancedb_upload_kwargs_keeps_explicit_mode(monkeypatch, explicit) -> None:
    monkeypatch.setattr(
        index_mode_module,
        "inspect_existing_lancedb_mode",
        lambda *_: pytest.fail("explicit modes must not inspect the existing table"),
    )
    vdb_kwargs = {"uri": "lancedb", "overwrite": False, **explicit}

    assert resolve_lancedb_upload_kwargs(vdb_kwargs) == vdb_kwargs


def test_resolve_lancedb_upload_kwargs_append_to_missing_table_is_hybrid(tmp_path) -> None:
    vdb_kwargs = {"uri": str(tmp_path), "table_name": "docs", "overwrite": False}

    assert resolve_lancedb_upload_kwargs(vdb_kwargs) == {**vdb_kwargs, "hybrid": True}


@pytest.mark.parametrize(
    ("vector", "fts", "expected"),
    [
        (True, False, {"hybrid": False}),
        (True, True, {"hybrid": True}),
        (False, True, {"sparse": True}),
    ],
)
def test_resolve_lancedb_upload_kwargs_append_preserves_existing_mode(tmp_path, vector, fts, expected) -> None:
    _create_table(str(tmp_path), "docs", vector=vector, fts=fts)
    vdb_kwargs = {"uri": str(tmp_path), "table_name": "docs", "overwrite": False}

    assert resolve_lancedb_upload_kwargs(vdb_kwargs) == {**vdb_kwargs, **expected}


def test_resolve_lancedb_upload_kwargs_append_uses_lancedb_default_target(monkeypatch) -> None:
    inspected: list[tuple[str, str]] = []

    def fake_inspect(uri: str, table_name: str) -> str:
        inspected.append((uri, table_name))
        return "dense"

    monkeypatch.setattr(index_mode_module, "inspect_existing_lancedb_mode", fake_inspect)

    assert resolve_lancedb_upload_kwargs({"overwrite": False}) == {"overwrite": False, "hybrid": False}
    assert inspected == [("lancedb", "nemo-retriever")]
