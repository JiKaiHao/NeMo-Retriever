# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Policy for resolving requested ingest modes against an existing LanceDB table."""

from __future__ import annotations

from typing import Any, Literal, Mapping, cast

from nemo_retriever.common.vdb.lancedb_capabilities import inspect_lancedb_table_object

RequestedIngestIndexMode = Literal["auto", "dense", "hybrid", "sparse"]
ResolvedIngestIndexMode = Literal["dense", "hybrid", "sparse"]

SUPPORTED_INGEST_INDEX_MODES: tuple[RequestedIngestIndexMode, ...] = (
    "auto",
    "dense",
    "hybrid",
    "sparse",
)


def validate_requested_index_mode(index_mode: str) -> RequestedIngestIndexMode:
    """Normalize and validate the public ingest index-mode vocabulary."""
    normalized = index_mode.strip().lower()
    if normalized not in SUPPORTED_INGEST_INDEX_MODES:
        raise ValueError(f"index_mode must be one of {', '.join(SUPPORTED_INGEST_INDEX_MODES)}, got {index_mode!r}.")
    return cast(RequestedIngestIndexMode, normalized)


def resolve_ingest_index_mode(
    requested_mode: RequestedIngestIndexMode,
    *,
    overwrite: bool,
    existing_mode: ResolvedIngestIndexMode | None,
) -> ResolvedIngestIndexMode:
    """Resolve one ingest request without mutating storage.

    ``auto`` makes fresh and overwritten tables hybrid, but preserves the
    physical mode of a table during append. The sole compatible mode-changing
    append is an explicit dense-to-hybrid upgrade.
    """
    if overwrite or existing_mode is None:
        return "hybrid" if requested_mode == "auto" else cast(ResolvedIngestIndexMode, requested_mode)

    if requested_mode == "auto" or requested_mode == existing_mode:
        return existing_mode

    if existing_mode == "dense" and requested_mode == "hybrid":
        return "hybrid"

    raise ValueError(
        f"Cannot append with index_mode={requested_mode!r} to an existing {existing_mode!r} table. "
        "Use index_mode='auto' to preserve the table mode, request 'hybrid' to upgrade a dense table, "
        "or overwrite the table to replace it."
    )


def inspect_existing_lancedb_mode(uri: str, table_name: str) -> ResolvedIngestIndexMode | None:
    """Return the physical mode of an existing table, or ``None`` when absent."""
    import lancedb  # type: ignore

    db = lancedb.connect(uri)
    if table_name not in db.list_tables().tables:
        return None

    capabilities = inspect_lancedb_table_object(db.open_table(table_name))
    if capabilities.retrieval_mode == "unknown":
        raise ValueError(
            f"Cannot determine physical retrieval capabilities for LanceDB table {table_name!r} at {uri!r}."
        )
    return cast(ResolvedIngestIndexMode, capabilities.retrieval_mode)


def resolve_lancedb_upload_kwargs(vdb_kwargs: Mapping[str, Any]) -> dict[str, Any]:
    """Apply ``auto`` to LanceDB upload kwargs that set neither ``hybrid`` nor ``sparse``."""
    kwargs = dict(vdb_kwargs)
    if "hybrid" in kwargs or "sparse" in kwargs:
        return kwargs
    # Fall back to the LanceDB constructor defaults for the target table.
    overwrite = bool(kwargs.get("overwrite", True))
    existing_mode = (
        None
        if overwrite
        else inspect_existing_lancedb_mode(
            str(kwargs.get("uri") or "lancedb"), str(kwargs.get("table_name") or "nemo-retriever")
        )
    )
    mode = resolve_ingest_index_mode("auto", overwrite=overwrite, existing_mode=existing_mode)
    if mode == "sparse":
        return {**kwargs, "sparse": True}
    return {**kwargs, "hybrid": mode == "hybrid"}
