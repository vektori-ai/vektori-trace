"""Write declared skip identities in the shape the rescore gate actually reads.

Usage: python scripts/pilotd_fix_declared_shape.py <manifest.json> <out.json>

`rescore` resolves declared skips through exactly one path:

    declared_exclusions[*]["observed_update_0"]["identities"]

The class declaration added earlier records the same facts under "observed",
which is readable but invisible to that code -- so update 1 failed its own
preregistration *after* the teacher was paid. This adds the key the gate
parses, listing every affected action across both updates.

The key name "observed_update_0" is a misnomer in the source: it is the only
path consulted for any update. Identities from update 1 therefore have to live
under it too. The semantic "observed" block is left intact.
"""
import json
import sys

IDENTITIES = [
    "u000-task76-seed0@8#0",
    "u000-task76-seed0@9#0",
    "u001-task4-seed1@4#0",
    "u001-task76-seed1@4#0",
    "u001-task76-seed1@7#0",
    "u001-task76-seed1@10#0",
]


def main():
    m = json.load(open(sys.argv[1]))
    plan_before = m.get("plan_hash")
    excl = m.get("declared_exclusions") or {}
    key = "hermes_markup_inside_reasoning_span"
    if key not in excl:
        raise SystemExit("class declaration missing; run pilotd_declare_interleave.py first")

    excl[key]["observed_update_0"] = {
        "identities": IDENTITIES,
        "key_name_note": (
            "Named observed_update_0 because that is the only path "
            "rescore reads; it holds identities from every update, not just 0."
        ),
    }

    m.setdefault("amendments", []).append({
        "date": "2026-08-29",
        "kind": "declaration_shape_fix",
        "what_changed": [
            "declared_exclusions.%s gains observed_update_0.identities" % key,
        ],
        "why": (
            "The class declaration recorded its instances under 'observed', "
            "which rescue's gate does not parse -- it reads only "
            "observed_update_0.identities. Update 1 was therefore refused "
            "after the teacher had already been paid. This adds the key the "
            "gate reads. The exclusion policy itself is unchanged; only its "
            "encoding is corrected."
        ),
        "unchanged": [
            "plan_hash and the frozen 80 (task, seed) pairs",
            "every stop rule and threshold",
            "the projection, the scores already bought, all fingerprints",
        ],
    })

    assert m.get("plan_hash") == plan_before
    with open(sys.argv[2], "w") as fh:
        json.dump(m, fh, indent=1, sort_keys=True)
        fh.write("\n")
    print("identities declared: %d" % len(IDENTITIES))
    for i in IDENTITIES:
        print("   %s" % i)
    print("plan_hash unchanged: %s" % m["plan_hash"])


main()
