from __future__ import annotations

from analysis import metrics, parser


def main() -> int:
    df = parser.load_phase_cd()
    hs = metrics.honest_summary(df)
    print("Phase C+D honest summary")
    for k, v in hs.items():
        print(f"  {k}: {v}")
    best = metrics.best_per_benchmark(df)
    print("\nBest per benchmark:")
    for _, r in best.iterrows():
        print(
            f"  {r['benchmark']:18s} {r['coupling']:22s} k={r['k']:g}  "
            f"dMS={r['delta_mutation_score']:+.4f}  dent={r['delta_entropy_norm']:+.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
