"""
Compare area-based states against the binary baseline on annotated scenarios.

Scenario file (JSON):
{
  "spots": {"P-01": [[x, y], ...], ...},
  "scenarios": [
    {"id": 1, "vehicles": [[[x, y], ...], ...], "truth": {"P-01": "free", ...}},
    ...
  ]
}
"""
import argparse
import json

from .occupancy import binary_state, occupancy_ratios, state_from_ratio


def run(path: str):
    data = json.load(open(path))
    ids = list(data["spots"])
    spots = [data["spots"][i] for i in ids]

    rows = []
    for sc in data["scenarios"]:
        ratios = occupancy_ratios(spots, sc["vehicles"])
        truth = sc["truth"]
        area_ok = sum(state_from_ratio(r).value == truth[i] for i, r in zip(ids, ratios))
        # binary is right when it says occupied for anything that is not free
        bin_ok = sum((binary_state(r).value == "occupied") == (truth[i] != "free") for i, r in zip(ids, ratios))
        rows.append((sc["id"], len(sc["vehicles"]), area_ok, bin_ok, len(ids)))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("scenarios")
    args = parser.parse_args()

    rows = run(args.scenarios)
    total = sum(r[4] for r in rows)
    print(f"{'id':>4} {'cars':>5} {'area':>6} {'binary':>7}")
    for sid, cars, a, b, n in rows:
        print(f"{sid:>4} {cars:>5} {a:>3}/{n} {b:>4}/{n}")
    print(f"area {100 * sum(r[2] for r in rows) / total:.2f} %  "
          f"binary {100 * sum(r[3] for r in rows) / total:.2f} %  ({total} evaluations)")


if __name__ == "__main__":
    main()
