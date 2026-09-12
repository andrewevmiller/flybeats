"""Phase 0 gate: resolve the pathway concepts named in PLAN.md to *real* MaleCNS
v1.0 cell-type strings, and refuse to invent any that are not there.

Writes ``data/verified_types.json``. Every later phase reads type names from that
file and nothing else -- no phase is allowed to hardcode a type string.

Why this file and not the neuPrint API: neuprint.janelia.org is not reachable
from every environment (and needs a personal token), while the flat-connectome
``body-annotations`` feather is the same v1.0 release, is public CC-BY, and is
what the weights table is keyed against. If you do have a token and want to
cross-check, ``--neuprint`` re-runs the same resolution against the server.

Resolution works on three columns, in order of authority:
  1. ``type``            -- the dataset's own primary type string
  2. ``flywireType`` / ``hemibrainType`` / ``mancType``  -- cross-dataset names
  3. ``synonyms``        -- free-text literature aliases, e.g. "Vaughan 2014: aPN1"

Concepts that only resolve via (3) are still confirmed, but the JSON records
which column matched so a reader can see the evidence.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "verified_types.json"

ANNOTATIONS = "body-annotations-male-cns-v1.0-minconf-0.5.feather"
NEUROTRANSMITTERS = "body-neurotransmitters-male-cns-v1.0.feather"

ALIAS_COLUMNS = ["flywireType", "hemibrainType", "mancType"]
SYNONYM_COLUMNS = ["synonyms"]


# --------------------------------------------------------------------------
# What we are looking for.
#
# ``type_regex``    matched against the primary ``type`` column
# ``alias_regex``   matched against cross-dataset type columns
# ``synonym_regex`` matched against the free-text literature synonyms column
# ``exclude_types`` literal type strings to drop even if a pattern matched them
#                   (guards against homonyms -- see vPN1 below)
# ``subclass``      restrict to an annotation subclass, when the dataset gives
#                   us one that is cleaner than any name pattern
# --------------------------------------------------------------------------
REQUESTS: list[dict] = [
    # ---- auditory afferents -------------------------------------------------
    dict(
        concept="JO_A",
        role="auditory_afferent",
        note="Johnston's Organ zone A. Plan said 'JO-A'; the dataset subdivides.",
        type_regex=r"^JO-A(\d+|-unclear)$",
    ),
    dict(
        concept="JO_B",
        role="auditory_afferent",
        note="JO zone B -- the courtship-song-tuned afferents (~100-500 Hz).",
        type_regex=r"^JO-B(\d+(_[a-z])?|-unclear)$",
    ),
    dict(
        concept="JO_E",
        role="auditory_afferent",
        note="JO zone E (gravity/wind). Splits into ED* and EV* subtypes.",
        type_regex=r"^JO-E[DV]?\d*(_[a-z])?$",
    ),
    dict(
        concept="JO_other",
        role="auditory_afferent",
        note="Remaining JO zones (C/D/F/mz) plus unassigned fragments.",
        type_regex=r"^JO-(C[ALM]\d*|D[AP]|F[DV]\d*|mz|unclear)$",
    ),
    # ---- second-order auditory ---------------------------------------------
    dict(
        concept="AMMC_interneuron",
        role="auditory_interneuron",
        note="AMMC local/projection interneurons, numbered AMMC001-038 in v1.0.",
        type_regex=r"^AMMC(\d+|-A\d+)$",
    ),
    dict(
        concept="WED_interneuron",
        role="auditory_interneuron",
        note="Wedge interneurons and wedge projection neurons (WEDPN*).",
        type_regex=r"^WED(PN)?\w*$",
    ),
    dict(
        concept="B1_interneuron",
        role="auditory_interneuron",
        note=(
            "The classic AMMC 'B1' interneuron of the song-response literature. "
            "No v1.0 type, alias or synonym carries this name -- the AMMC/WED "
            "populations above are numbered instead. Left UNCONFIRMED on "
            "purpose: do not silently map it onto AMMC001 or JO-B1_a."
        ),
        type_regex=r"^B1$",
        synonym_regex=r"(?:^|[^\w])B1(?:$|[^\w\d])",
        optional=True,
    ),
    dict(
        concept="aPN1",
        role="auditory_interneuron",
        note="Resolves only via synonyms (Vaughan 2014). Real types are SAD051_*/CB*.",
        synonym_regex=r"aPN1",
    ),
    dict(
        concept="vPN1",
        role="auditory_interneuron",
        note=(
            "Resolves only via synonyms (Zhou 2015) to AVLP76[123]m. "
            "VP1m+VP2_lvPN1 is an olfactory projection neuron and a pure name "
            "collision -- excluded explicitly."
        ),
        synonym_regex=r"vPN1",
        exclude_types=["VP1m+VP2_lvPN1"],
    ),
    # ---- courtship drive / internal state -----------------------------------
    dict(
        concept="pC1",
        role="state_drive",
        note="pC1 courtship-drive cluster; v1.0 splits it into pC1_* and pC1x_*.",
        type_regex=r"^pC1(x)?_\w+$",
    ),
    dict(
        concept="pC2",
        role="state_drive",
        note=(
            "pC2 is NOT a v1.0 type string. It survives only in synonyms, where "
            "it maps onto AVLP*/LAL*/SIP*/VES*/PVLP* 'm' types (pC2l and pC2m)."
        ),
        synonym_regex=r"(?:^|[^\w])pC2[lm]?(?:$|[^\w])",
    ),
    # ---- song command -------------------------------------------------------
    dict(
        concept="pIP10",
        role="descending_command",
        note="The song command descending neuron. Exactly one bilateral pair.",
        type_regex=r"^pIP10$",
    ),
    # ---- motor --------------------------------------------------------------
    dict(
        concept="wing_steering_MN",
        role="motor",
        note=(
            "Steering MNs named in PLAN.md. iii2 and iii4 do NOT exist in v1.0 "
            "(only iii1 and iii3); ps2 does exist and the plan omitted it."
        ),
        type_regex=r"^(b[123]|hg[1-4]|i[12]|iii[1-4]|ps[12]|tp[12])\s+MN$",
    ),
    dict(
        concept="wing_power_MN",
        role="motor",
        note="Indirect power muscles. Note the literal commas/spaces in the names.",
        type_regex=r"^(DLMn|DVMn|hDVM)\b.*$",
    ),
    dict(
        concept="wing_motor_all",
        role="motor",
        note=(
            "Authoritative wing-motor set: annotation subclass 'wm'. Preferred "
            "over any name pattern -- this is the dataset's own grouping."
        ),
        subclass="wm",
    ),
    dict(
        concept="leg_MN_T2",
        role="motor",
        note="Mesothoracic leg MNs, for the v2 'leg mode' large-kit tier.",
        subclass="ml",
    ),
    # ---- neuromodulation ----------------------------------------------------
    dict(
        concept="octopaminergic",
        role="neuromodulator",
        note=(
            "Genre-embedding injection site. OA-VUMa*/OA-VPM3/4 are real types; "
            "'OA-VPM1'/'OA-VPM2' are NOT -- they are synonyms of DNg34/DNg104."
        ),
        type_regex=r"^OA-(VUM|VPM)\w+$",
        synonym_regex=r"OA-V(?:UM|PM)|VUM[a-z]?\d",
    ),
    dict(
        concept="dopaminergic",
        role="neuromodulator",
        note="v2 reward channel hook (PPL1/PAM).",
        type_regex=r"^(PPL1|PAM)\d*\w*$",
        optional=True,
    ),
]


def load_annotations(raw: Path) -> pd.DataFrame:
    path = raw / ANNOTATIONS
    if not path.exists():
        raise SystemExit(
            f"missing {path}\nRun: python scripts/fetch_data.py"
        )
    return pd.read_feather(path)


def load_nt(raw: Path) -> pd.DataFrame | None:
    path = raw / NEUROTRANSMITTERS
    if not path.exists():
        return None
    nt = pd.read_feather(path)
    return nt[["body", "consensus_nt", "celltype_predicted_nt", "ground_truth"]]


def _col(df: pd.DataFrame, name: str) -> pd.Series:
    return df[name].fillna("").astype(str) if name in df.columns else pd.Series("", index=df.index)


def resolve(ann: pd.DataFrame, req: dict) -> tuple[pd.DataFrame, dict[str, str]]:
    """Return matching bodies plus, per resolved type, which column proved it."""
    mask = pd.Series(False, index=ann.index)
    evidence: dict[str, str] = {}

    def mark(sub_mask: pd.Series, label: str) -> None:
        nonlocal mask
        for ty in ann.loc[sub_mask, "type"].dropna().unique():
            evidence.setdefault(ty, label)
        mask = mask | sub_mask

    if req.get("subclass"):
        mark(_col(ann, "subclass") == req["subclass"], "subclass")
    if req.get("type_regex"):
        mark(_col(ann, "type").str.fullmatch(req["type_regex"], case=False).fillna(False), "type")
    if req.get("alias_regex"):
        for c in ALIAS_COLUMNS:
            mark(_col(ann, c).str.contains(req["alias_regex"], case=False, regex=True), f"alias:{c}")
    if req.get("synonym_regex"):
        for c in SYNONYM_COLUMNS:
            mark(_col(ann, c).str.contains(req["synonym_regex"], case=False, regex=True), f"synonym:{c}")

    sub = ann[mask]
    for bad in req.get("exclude_types", []):
        sub = sub[sub["type"] != bad]
        evidence.pop(bad, None)
    # bodies with no type string at all are unusable downstream
    sub = sub[sub["type"].notna()]
    return sub, evidence


def summarise(sub: pd.DataFrame, evidence: dict[str, str], nt: pd.DataFrame | None) -> list[dict]:
    if sub.empty:
        return []
    if nt is not None:
        sub = sub.merge(nt, left_on="bodyId", right_on="body", how="left")

    out = []
    for ty, g in sub.groupby("type", sort=True):
        rec = {
            "type": ty,
            "n_bodies": int(len(g)),
            "body_ids": sorted(int(b) for b in g["bodyId"]),
            "sides": sorted({s for s in g.get("somaSide", pd.Series(dtype=str)).dropna()}),
            "superclass": sorted({s for s in g["superclass"].dropna()}),
            "class": sorted({s for s in g["class"].dropna()}),
            "matched_on": evidence.get(ty, "type"),
            "n_traced": int((g.get("status", pd.Series(dtype=str)) == "Traced").sum()),
        }
        if nt is not None:
            votes = pd.concat([g["ground_truth"], g["consensus_nt"], g["celltype_predicted_nt"]]).dropna()
            votes = votes[votes != "unclear"]
            rec["neurotransmitter"] = votes.mode().iloc[0] if len(votes) else "unclear"
        out.append(rec)
    return sorted(out, key=lambda r: -r["n_bodies"])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", type=Path, default=RAW)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--strict", action="store_true",
                    help="exit non-zero if any non-optional concept is unconfirmed")
    args = ap.parse_args(argv)

    ann = load_annotations(args.raw)
    nt = load_nt(args.raw)
    print(f"annotations: {len(ann):,} bodies, {ann['type'].nunique():,} distinct types")
    print(f"neurotransmitters: {'loaded' if nt is not None else 'MISSING (signs will be unavailable)'}\n")

    concepts: dict[str, dict] = {}
    unconfirmed: list[str] = []

    for req in REQUESTS:
        sub, evidence = resolve(ann, req)
        types = summarise(sub, evidence, nt)
        confirmed = len(types) > 0
        concepts[req["concept"]] = {
            "role": req["role"],
            "note": req["note"],
            "confirmed": confirmed,
            "optional": bool(req.get("optional")),
            "n_types": len(types),
            "n_bodies": int(sum(t["n_bodies"] for t in types)),
            "types": types,
        }
        flag = "OK  " if confirmed else ("skip" if req.get("optional") else "MISS")
        if not confirmed and not req.get("optional"):
            unconfirmed.append(req["concept"])
        n_bodies = sum(t["n_bodies"] for t in types)
        print(f"[{flag}] {req['concept']:<22} {len(types):>4} types  {n_bodies:>6} bodies"
              + (f"   e.g. {', '.join(t['type'] for t in types[:4])}" if types else ""))

    payload = {
        "dataset": "MaleCNS v1.0 (flat-connectome, minconf-0.5)",
        "source": "gs://flyem-male-cns/v1.0/connectome-data/flat-connectome/",
        "license": "CC-BY",
        "generated": date.today().isoformat(),
        "generator": "scripts/verify_types.py",
        "resolution_policy": "type > flywire/hemibrain/mancType alias > literature synonym",
        "totals": {
            "bodies_in_annotation_table": int(len(ann)),
            "distinct_types": int(ann["type"].nunique()),
            "traced": int((ann["status"] == "Traced").sum()),
        },
        "unconfirmed": unconfirmed,
        "concepts": concepts,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {args.out}")

    if unconfirmed:
        print(f"UNCONFIRMED (non-optional): {', '.join(unconfirmed)}")
        if args.strict:
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
