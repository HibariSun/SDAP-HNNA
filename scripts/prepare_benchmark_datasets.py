"""Prepare benchmark datasets for SDAP-MHNN.

This script does not modify the original paper folders. It writes normalized
copies under benchmarks/<dataset>/ with the filenames expected by DataLoader:

  adj_index.csv
  sno_p2p_smith.csv
  disease_do_similarity.csv

When external embeddings can be matched completely by identifier, it also writes:

  sno_rnaernie_emb.npy
  disease_feat_bgeicl.csv

Otherwise it writes missing_sno_rnaernie_ids.csv and/or
missing_disease_descriptions.csv so the embeddings can be generated later from
authoritative descriptions.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def _read_indexed_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, index_col=0)


def _write_indexed_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)


def _as_binary(df: pd.DataFrame) -> pd.DataFrame:
    out = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return (out > 0).astype(np.float32)


def _normalize_square(df: pd.DataFrame, ids: list[str]) -> pd.DataFrame:
    values = df.apply(pd.to_numeric, errors="coerce").fillna(0.0).values.astype(np.float32)
    if values.shape != (len(ids), len(ids)):
        raise ValueError(f"Similarity matrix shape {values.shape} does not match {len(ids)} ids")
    values = (values + values.T) / 2.0
    values = np.clip(values, 0.0, 1.0)
    np.fill_diagonal(values, 1.0)
    return pd.DataFrame(values, index=ids, columns=ids)


def _cosine_similarity(features: pd.DataFrame, ids: list[str]) -> pd.DataFrame:
    x = features.apply(pd.to_numeric, errors="coerce").fillna(0.0).values.astype(np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    x = x / norms
    sim = x @ x.T
    sim = np.clip(sim, 0.0, 1.0)
    np.fill_diagonal(sim, 1.0)
    return pd.DataFrame(sim, index=ids, columns=ids)


def _write_metadata(out_dir: Path, metadata: dict) -> None:
    (out_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _copy_matching_embeddings(out_dir: Path, sno_ids: list[str], disease_ids: list[str]) -> None:
    """Copy current embeddings only when every target id is available.

    Partial embedding matrices are intentionally not written because DataLoader
    requires the embedding row count and order to match the adjacency matrix.
    """
    source_adj = ROOT / "data" / "adj_index.csv"
    source_sno_emb = ROOT / "data" / "sno_rnaernie_emb.npy"
    source_dis_emb = ROOT / "data" / "disease_feat_bgeicl.csv"

    if source_adj.exists() and source_sno_emb.exists():
        source_sno_ids = list(map(str, pd.read_csv(source_adj, index_col=0).index))
        source_pos = {sid: i for i, sid in enumerate(source_sno_ids)}
        missing = [sid for sid in sno_ids if sid not in source_pos]
        if not missing:
            emb = np.load(source_sno_emb).astype(np.float32)
            subset = emb[[source_pos[sid] for sid in sno_ids]]
            np.save(out_dir / "sno_rnaernie_emb.npy", subset)
        else:
            pd.DataFrame({"snoRNA": missing}).to_csv(
                out_dir / "missing_sno_rnaernie_ids.csv", index=False
            )

    if source_dis_emb.exists():
        dis_df = pd.read_csv(source_dis_emb)
        id_col = dis_df.columns[0]
        dis_df[id_col] = dis_df[id_col].astype(str)
        dis_df = dis_df.set_index(id_col)
        missing = [did for did in disease_ids if did not in dis_df.index]
        if not missing:
            dis_df.loc[disease_ids].reset_index().to_csv(
                out_dir / "disease_feat_bgeicl.csv", index=False
            )
        else:
            pd.DataFrame(
                {
                    "disease_id": missing,
                    "disease_name": missing,
                    "preferred_source": ["DOID/MeSH/MedGen"] * len(missing),
                    "description": [""] * len(missing),
                }
            ).to_csv(out_dir / "missing_disease_descriptions.csv", index=False)


def prepare_psnod() -> dict:
    src = ROOT / "PSnoD(2022)" / "PSnoD-master" / "input_data"
    out_dir = ROOT / "benchmarks" / "psnod"
    out_dir.mkdir(parents=True, exist_ok=True)

    rel = _as_binary(_read_indexed_csv(src / "relationship_matrix_filtered.csv"))
    # PSnoD stores diseases as rows and snoRNAs as columns; SDAP-MHNN expects
    # snoRNAs as rows and diseases as columns.
    adj = rel.T
    adj.index = adj.index.astype(str)
    adj.columns = adj.columns.astype(str)

    sno_sim_raw = _read_indexed_csv(src / "snoRNA_sim" / "snoRNA_4mer_similarity.csv")
    sno_sim_raw.index = sno_sim_raw.index.astype(str)
    sno_sim_raw.columns = sno_sim_raw.columns.astype(str)
    sno_sim = _normalize_square(sno_sim_raw.loc[adj.index, adj.index], list(adj.index))

    disease_sim_raw = _read_indexed_csv(src / "disease_sim_graph_filtered.csv")
    disease_sim_raw.index = disease_sim_raw.index.astype(str)
    disease_sim_raw.columns = disease_sim_raw.columns.astype(str)
    disease_sim = _normalize_square(
        disease_sim_raw.loc[adj.columns, adj.columns], list(adj.columns)
    )

    _write_indexed_csv(adj, out_dir / "adj_index.csv")
    _write_indexed_csv(sno_sim, out_dir / "sno_p2p_smith.csv")
    _write_indexed_csv(disease_sim, out_dir / "disease_do_similarity.csv")
    _copy_matching_embeddings(out_dir, list(adj.index), list(adj.columns))

    metadata = {
        "dataset": "psnod",
        "source": "PSnoD(2022)/PSnoD-master/input_data",
        "snoRNAs": int(adj.shape[0]),
        "diseases": int(adj.shape[1]),
        "associations": int(adj.values.sum()),
        "notes": "Disease ids are MeSH ids from the PSnoD source files.",
    }
    _write_metadata(out_dir, metadata)
    return metadata


def prepare_gbdtsvm() -> dict:
    src = ROOT / "GBDTSVM（2024）" / "gbdtsvm-main"
    out_dir = ROOT / "benchmarks" / "gbdtsvm"
    out_dir.mkdir(parents=True, exist_ok=True)

    adj_values = pd.read_csv(src / "Data" / "known_snoRNA_disease.csv", header=None)
    adj_values = _as_binary(adj_values)

    sno_feature_df = pd.read_csv(src / "Data" / "snoRNA_features.csv")
    sno_ids = list(map(str, sno_feature_df.iloc[:, 0]))
    if len(sno_ids) != adj_values.shape[0]:
        raise ValueError("GBDTSVM snoRNA feature rows do not match association matrix rows")

    disease_names = pd.read_csv(src / "Data" / "disease_name.csv", header=None)
    disease_ids = list(map(str, disease_names.iloc[:, 1]))
    if len(disease_ids) != adj_values.shape[1]:
        raise ValueError("GBDTSVM disease_name rows do not match association matrix columns")

    adj = pd.DataFrame(adj_values.values, index=sno_ids, columns=disease_ids)
    sno_sim = _cosine_similarity(sno_feature_df.iloc[:, 1:], sno_ids)

    disease_sim_raw = pd.read_csv(src / "Data" / "disease_similarity.csv", header=None)
    disease_sim = _normalize_square(
        pd.DataFrame(disease_sim_raw.values, index=disease_ids, columns=disease_ids),
        disease_ids,
    )

    _write_indexed_csv(adj, out_dir / "adj_index.csv")
    _write_indexed_csv(sno_sim, out_dir / "sno_p2p_smith.csv")
    _write_indexed_csv(disease_sim, out_dir / "disease_do_similarity.csv")
    _copy_matching_embeddings(out_dir, list(adj.index), list(adj.columns))

    pd.DataFrame({"disease_id": disease_ids, "disease_name": disease_ids}).to_csv(
        out_dir / "disease_names.csv", index=False
    )

    metadata = {
        "dataset": "gbdtsvm",
        "source": "GBDTSVM（2024）/gbdtsvm-main/Data",
        "snoRNAs": int(adj.shape[0]),
        "diseases": int(adj.shape[1]),
        "associations": int(adj.values.sum()),
        "notes": "Disease ids are names from disease_name.csv; snoRNA similarity is cosine similarity from snoRNA_features.csv.",
    }
    _write_metadata(out_dir, metadata)
    return metadata


def prepare_gcnmf_sda() -> dict:
    src = ROOT / "GCNMF-SDA(2025)" / "GCNMF-SDA-main" / "data"
    out_dir = ROOT / "benchmarks" / "gcnmf_sda"
    out_dir.mkdir(parents=True, exist_ok=True)

    adj = _as_binary(_read_indexed_csv(src / "adj_index.csv"))
    adj.index = adj.index.astype(str)
    adj.columns = adj.columns.astype(str)

    sno_sim_raw = _read_indexed_csv(src / "sno_p2p_smith.csv")
    sno_sim_raw.index = sno_sim_raw.index.astype(str)
    sno_sim_raw.columns = sno_sim_raw.columns.astype(str)
    sno_sim = _normalize_square(sno_sim_raw.loc[adj.index, adj.index], list(adj.index))

    disease_sim_raw = _read_indexed_csv(src / "disease_similarity.csv")
    disease_sim_raw.index = disease_sim_raw.index.astype(str)
    disease_sim_raw.columns = disease_sim_raw.columns.astype(str)
    disease_sim = _normalize_square(
        disease_sim_raw.loc[adj.columns, adj.columns], list(adj.columns)
    )

    _write_indexed_csv(adj, out_dir / "adj_index.csv")
    _write_indexed_csv(sno_sim, out_dir / "sno_p2p_smith.csv")
    _write_indexed_csv(disease_sim, out_dir / "disease_do_similarity.csv")
    _copy_matching_embeddings(out_dir, list(adj.index), list(adj.columns))

    metadata = {
        "dataset": "gcnmf_sda",
        "source": "GCNMF-SDA(2025)/GCNMF-SDA-main/data",
        "snoRNAs": int(adj.shape[0]),
        "diseases": int(adj.shape[1]),
        "associations": int(adj.values.sum()),
        "notes": "Disease ids are disease names from GCNMF-SDA adj_index.csv.",
    }
    _write_metadata(out_dir, metadata)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare benchmark datasets for SDAP-MHNN.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["psnod", "gbdtsvm", "gcnmf_sda"],
        choices=["psnod", "gbdtsvm", "gcnmf_sda"],
    )
    parser.add_argument("--clean", action="store_true", help="Remove benchmarks/<dataset> before writing.")
    args = parser.parse_args()

    preparers = {
        "psnod": prepare_psnod,
        "gbdtsvm": prepare_gbdtsvm,
        "gcnmf_sda": prepare_gcnmf_sda,
    }

    summaries = []
    for dataset in args.datasets:
        out_dir = ROOT / "benchmarks" / dataset
        if args.clean and out_dir.exists():
            shutil.rmtree(out_dir)
        print(f"[prepare] {dataset}")
        summary = preparers[dataset]()
        summaries.append(summary)
        print(
            f"  -> {summary['snoRNAs']} snoRNAs, {summary['diseases']} diseases, "
            f"{summary['associations']} associations"
        )

    summary_path = ROOT / "benchmarks" / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[done] Wrote {summary_path}")


if __name__ == "__main__":
    main()
