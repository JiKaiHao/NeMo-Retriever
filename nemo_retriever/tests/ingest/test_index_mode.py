# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import lancedb
import pytest

from nemo_retriever.ingest.index_mode import resolve_ingest_index_mode, resolve_lancedb_upload_kwargs


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


@pytest.mark.parametrize(
    ("vdb_kwargs", "expected"),
    [
        ({}, {"hybrid": True}),
        ({"overwrite": False}, {"hybrid": True}),
        ({"hybrid": False}, {}),
        ({"sparse": True, "overwrite": False}, {}),
    ],
)
def test_resolve_lancedb_upload_kwargs_fresh_tables_and_explicit_modes(tmp_path, vdb_kwargs, expected) -> None:
    vdb_kwargs = {"uri": str(tmp_path), **vdb_kwargs}
    assert resolve_lancedb_upload_kwargs(vdb_kwargs) == {**vdb_kwargs, **expected}


@pytest.mark.parametrize(
    ("row", "fts", "expected"),
    [
        ({"vector": [0.1, 0.2], "text": "alpha"}, False, {"hybrid": False}),
        ({"vector": [0.1, 0.2], "text": "alpha"}, True, {"hybrid": True}),
        ({"text": "alpha"}, True, {"sparse": True}),
    ],
)
def test_resolve_lancedb_upload_kwargs_append_preserves_existing_mode(tmp_path, row, fts, expected) -> None:
    table = lancedb.connect(str(tmp_path)).create_table("docs", data=[row])
    if fts:
        table.create_fts_index("text")
    vdb_kwargs = {"uri": str(tmp_path), "table_name": "docs", "overwrite": False}

    assert resolve_lancedb_upload_kwargs(vdb_kwargs) == {**vdb_kwargs, **expected}
