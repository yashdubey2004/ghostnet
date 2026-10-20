"""Train everything from the command line and write results to ./artifacts.

    python run_pipeline.py                         # synthetic demo data
    python run_pipeline.py --csv data/Wednesday-workingHours.pcap_ISCX.csv   # CIC-IDS2017 file(s)
"""
import argparse
from pathlib import Path

from src.data import generate_synthetic, load_flows
from src.evaluate import make_report
from src.pipeline import save_bundle, train_all

ap = argparse.ArgumentParser()
ap.add_argument("--csv", nargs="*", help="one or more CIC-IDS2017 CSV files")
ap.add_argument("--n", type=int, default=12000, help="synthetic flows to generate")
ap.add_argument("--rbm-epochs", type=int, default=40)
ap.add_argument("--vae-epochs", type=int, default=100)
args = ap.parse_args()

if args.csv:
    df, source = load_flows(args.csv), "CIC-IDS2017 subset"
else:
    df, source = generate_synthetic(args.n), "synthetic demo traffic"
print(f"{len(df)} flows loaded:", df["label"].value_counts().to_dict())

bundle = train_all(df, source, rbm_epochs=args.rbm_epochs, vae_epochs=args.vae_epochs)
out = Path("artifacts"); out.mkdir(exist_ok=True)
save_bundle(bundle, out / "bundle.pkl")
(out / "report.md").write_text(make_report(bundle))
print(bundle.metrics.round(3).to_string())
print("\nSaved artifacts/bundle.pkl and artifacts/report.md")
