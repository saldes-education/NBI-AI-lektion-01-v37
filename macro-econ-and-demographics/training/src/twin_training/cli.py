import argparse
import logging
import os
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

from twin_training.data import TRAINING_WINDOW, TWIN_COUNTRIES, build_inputs, country_frame
from twin_training.evaluate import DEFAULT_SPLIT_YEAR, run_backtest
from twin_training.export import export_artifacts
from twin_training.fit import fit_twin
from twin_training.tracking import DEFAULT_EXPERIMENT, run_params, tracking_session

log = logging.getLogger("twin_training")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="twin-train", description="Fit the country twins and export a serving artifact."
    )
    parser.add_argument(
        "--data-dir", type=Path, default=Path(os.environ.get("TWIN_DATA_DIR", "data/raw"))
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(os.environ.get("TWIN_ARTIFACT_ROOT", "artifacts")),
    )
    parser.add_argument(
        "--version",
        default=datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"),
        help="Artifact directory name under --output-root (default: UTC timestamp).",
    )
    parser.add_argument("--countries", nargs="+", default=list(TWIN_COUNTRIES))
    parser.add_argument("--maxiter", type=int, default=250)
    parser.add_argument(
        "--backtest-split",
        type=int,
        default=DEFAULT_SPLIT_YEAR,
        help="Last training year of the out-of-sample backtest (default: %(default)s).",
    )
    parser.add_argument(
        "--no-backtest", action="store_true", help="Skip the backtest for faster runs."
    )
    parser.add_argument(
        "--tracking-uri",
        default=None,
        help="MLflow tracking URI (default: $MLFLOW_TRACKING_URI, else sqlite:///mlflow.db).",
    )
    parser.add_argument(
        "--experiment",
        default=DEFAULT_EXPERIMENT,
        help="MLflow experiment name (default: %(default)s).",
    )
    parser.add_argument(
        "--no-tracking", action="store_true", help="Don't record the run in MLflow."
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    log.info("Building inputs from %s", args.data_dir)
    inputs = build_inputs(args.data_dir)

    params = run_params(
        version=args.version,
        countries=args.countries,
        maxiter=args.maxiter,
        backtest_split=args.backtest_split,
        backtest_enabled=not args.no_backtest,
        training_window=TRAINING_WINDOW,
        data_dir=args.data_dir,
    )
    with tracking_session(
        enabled=not args.no_tracking,
        tracking_uri=args.tracking_uri,
        experiment=args.experiment,
        run_name=args.version,
        params=params,
    ) as recorder:
        fits = {}
        for country in args.countries:
            with recorder.country(country):
                frame = country_frame(inputs, country)
                log.info(
                    "Fitting %s on %d years (%d-%d)",
                    country,
                    len(frame),
                    frame.index.min(),
                    frame.index.max(),
                )
                fitted = fit_twin(frame, maxiter=args.maxiter)
                log.info("  %s", fitted.diagnostics)

                if not args.no_backtest:
                    # The backtest is a second, evaluation-only fit; the served model stays the
                    # full fit.
                    backtest = run_backtest(
                        frame, split_year=args.backtest_split, maxiter=args.maxiter
                    )
                    log.info(
                        "  backtest %d-%d skill vs persistence: %s",
                        backtest.train_window_end + 1,
                        backtest.train_window_end + backtest.n_eval_years,
                        ", ".join(f"{name}={value:+.2f}" for name, value in backtest.skill.items()),
                    )
                    fitted = replace(
                        fitted, diagnostics={**fitted.diagnostics, **backtest.diagnostics()}
                    )
                fits[country] = fitted
                recorder.log_country(fitted.diagnostics)

        out_dir = export_artifacts(
            fits, args.output_root / args.version, artifact_version=args.version
        )
        log.info("Wrote artifact %s", out_dir)
        recorder.log_artifacts(out_dir)


if __name__ == "__main__":
    main()
