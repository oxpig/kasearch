"""Command-line interface to KA-Search.

Searches pre-aligned OAS for the sequences most identical to each query, by
region.  Queries come from a FASTA file, or a CSV file with a `sequence`
column and an optional `id` column.

`--database` is `oasdb-tiny` (the default) or `oasdb-small`, which are
downloaded into the installed package on first use, or the path to any
KA-Search database.

By default the output identifies each hit only by its OAS data unit and
line, which needs no network.  `--metadata` adds the hit's full OAS record,
which KA-Search reads from the OAS website unless `--local-oas` points at
a local copy of OAS.
"""

import argparse
import sys

import pandas as pd

REGIONS = ["whole", "cdrs", "cdr3"]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="kasearch",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input", help="FASTA, or CSV with a `sequence` column")
    parser.add_argument("-o", "--output", required=True, help="output CSV file")
    parser.add_argument(
        "--database",
        default="oasdb-tiny",
        help="`oasdb-tiny`, `oasdb-small` or a path (default oasdb-tiny)",
    )
    parser.add_argument(
        "-n", "--top", type=int, default=10, help="hits per query and region"
    )
    parser.add_argument(
        "--region",
        dest="regions",
        action="append",
        choices=REGIONS,
        help="region to compare; repeat for several (default all three)",
    )
    parser.add_argument(
        "--no-length-match",
        action="store_true",
        help="compare CDRs of different lengths (by default `cdrs` and "
        "`cdr3` compare only hits of matching length)",
    )
    parser.add_argument("--chain", default="Any", choices=["Any", "Heavy", "Light"])
    parser.add_argument(
        "--species", default="Any", help="e.g. Human, Mouse (default Any)"
    )
    parser.add_argument(
        "--metadata", action="store_true", help="add each hit's OAS record"
    )
    parser.add_argument("--local-oas", help="local copy of OAS, for --metadata")
    parser.add_argument("--ncpu", type=int, default=1)
    return parser.parse_args(argv)


def read_queries(path):
    with open(path) as handle:
        is_fasta = handle.read(1) == ">"
    if not is_fasta:
        table = pd.read_csv(path, dtype=str)
        if "sequence" not in table.columns:
            sys.exit("kasearch: CSV input needs a `sequence` column")
        ids = table["id"] if "id" in table.columns else table.index.astype(str)
        return list(ids), list(table["sequence"])

    ids, sequences = [], []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if line.startswith(">"):
                ids.append(line[1:].split()[0])
                sequences.append("")
            elif line:
                sequences[-1] += line
    return ids, sequences


def main(argv=None):
    args = parse_args(argv)
    regions = args.regions or REGIONS
    length_matched = [r != "whole" and not args.no_length_match for r in regions]
    ids, sequences = read_queries(args.input)

    # Imported late: loading JAX is slow, and `--help` shouldn't wait.
    from kasearch import AlignSequences, SearchDB

    queries = AlignSequences(n_jobs=args.ncpu)(sequences)
    db = SearchDB(
        database_path=args.database,
        allowed_chain=args.chain,
        allowed_species=args.species,
        regions=regions,
        length_matched=length_matched,
        local_oas_path=args.local_oas,
    )
    db.search(queries, keep_best_n=args.top)

    frames = []
    for q, query_id in enumerate(ids):
        for r, region in enumerate(regions):
            if args.metadata:
                hits = db.get_meta(n_query=q, n_region=r, n_jobs=args.ncpu)
            else:
                found = db.current_best_ids[q, :, r]
                hits = pd.DataFrame(
                    {
                        "data_unit": [db.id_to_study.get(s) for s in found[:, 0]],
                        "line": found[:, 1],
                        "Identity": db.current_best_identities[q, :, r],
                    }
                )
                # Fewer than --top hits leaves unfilled slots.
                hits = hits[found[:, 0] >= 0]
            hits.insert(0, "rank", range(1, len(hits) + 1))
            hits.insert(0, "region", region)
            hits.insert(0, "query", query_id)
            frames.append(hits)

    pd.concat(frames, ignore_index=True).to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
