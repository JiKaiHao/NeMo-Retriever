# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LanceDB ingest policy: resolve requested index modes against an existing table."""

from __future__ import annotations

from typing import Any, Literal, Mapping, cast

from nemo_retriever.common.vdb.lancedb_capabilities import (
    _metadata_retrieval_mode,
    _table_schema,
    inspect_lancedb_table_object,
)
from nemo_retriever.models import NEMOTRON_3_EMBED_MODEL, resolve_embed_model
from nemo_retriever.models.embed_model_spec import resolve_embed_model_revision

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
    """Return the mode of an existing table, or ``None`` when absent."""
    import lancedb  # type: ignore

    db = lancedb.connect(uri)
    if table_name not in db.list_tables().tables:
        return None

    table = db.open_table(table_name)
    capabilities = inspect_lancedb_table_object(table)
    if capabilities.retrieval_mode == "unknown":
        raise ValueError(
            f"Cannot determine physical retrieval capabilities for LanceDB table {table_name!r} at {uri!r}."
        )
    # A hybrid table written with build_index=False has no FTS index yet; keep its recorded mode.
    if capabilities.retrieval_mode == "dense" and _metadata_retrieval_mode(_table_schema(table)) == "hybrid":
        return "hybrid"
    return cast(ResolvedIngestIndexMode, capabilities.retrieval_mode)


def lancedb_index_mode_kwargs(mode: ResolvedIngestIndexMode) -> dict[str, bool]:
    """Return the ``LanceDB`` constructor kwargs that select one resolved mode."""
    return {"sparse": True} if mode == "sparse" else {"hybrid": mode == "hybrid"}


def resolve_lancedb_upload_kwargs(vdb_kwargs: Mapping[str, Any]) -> dict[str, Any]:
    """Apply the ``auto`` ingest index mode to LanceDB upload kwargs.

    Parameters
    ----------
    vdb_kwargs
        ``LanceDB`` constructor kwargs from ``VdbUploadParams``. An explicit
        ``hybrid`` or ``sparse=True`` is authoritative and returned unchanged.

    Returns
    -------
    dict[str, Any]
        A copy of ``vdb_kwargs`` with the resolved mode: ``hybrid=True`` for new
        or overwritten tables, and the existing table's mode on append.

    Raises
    ------
    ValueError
        If appending to an existing table whose retrieval mode cannot be determined.
    """
    kwargs = dict(vdb_kwargs)
    if "hybrid" in kwargs or kwargs.get("sparse"):
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
    return {**kwargs, **lancedb_index_mode_kwargs(mode)}


def _embedding_model_kwargs(embed_params: Any) -> dict[str, Any]:
    """Describe the embed stage's vectors on the table, as ``retriever ingest`` does."""
    kwargs: dict[str, Any] = {}
    if embed_params.embed_model_name or embed_params.model_name or embed_params.dimensions:
        # Infer the width instead of assuming the default model's 2048-d vectors.
        kwargs["vector_dim"] = None
    remote = str(embed_params.embed_invoke_url or embed_params.embedding_endpoint or "").strip()
    if not remote and embed_params.embed_model_name not in (None, embed_params.model_name):
        # A local actor would use embed_model_name, a CPU-only fallback model_name; don't guess.
        return kwargs
    # Remote and CPU-fallback actors embed with model_name, and a local actor agrees here.
    model_name = resolve_embed_model(embed_params.model_name)
    kwargs["embedding_model_name"] = model_name
    if model_name != NEMOTRON_3_EMBED_MODEL and not remote:
        revision = resolve_embed_model_revision(model_name, embed_params.embed_model_revision)
        if revision:
            kwargs["embedding_model_revision"] = revision
    return kwargs


def resolve_vdb_upload_kwargs(vdb_upload_params: Any, embed_params: Any = None) -> dict[str, Any]:
    """Return ``IngestVdbOperator`` kwargs with the LanceDB ingest policy applied.

    LanceDB uploads resolve the ``auto`` index mode and, after an ``.embed()`` stage,
    record its model as ``retriever ingest`` does. Other VDBs pass through unchanged.
    """
    vdb_kwargs = vdb_upload_params.to_ingest_operator_kwargs()
    if vdb_upload_params.vdb_op != "lancedb":
        return vdb_kwargs
    vdb_kwargs = resolve_lancedb_upload_kwargs(vdb_kwargs)
    if embed_params is None or vdb_kwargs.get("sparse") or "embedding_model_name" in vdb_kwargs:
        return vdb_kwargs
    return {**_embedding_model_kwargs(embed_params), **vdb_kwargs}
