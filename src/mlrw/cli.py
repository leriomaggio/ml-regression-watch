"""Command-line interface for ml-regression-watch.

Four commands cover the workflow: ``run`` benchmarks the models and writes an
artifact, ``validate`` checks numerical correctness, ``compare`` detects performance
regressions against a baseline, and ``update-baseline`` promotes a run to the stored
baseline. The commands compose through the JSON artifact, so each can be invoked
independently in a CI pipeline.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import typer

from . import __version__
from .artifacts import RunArtifact, RunMetadata, load_artifact, save_artifact
from .config import default_configs, resolve_device

app = typer.Typer(
    add_completion=False,
    help="Benchmarking, numerical validation, and regression detection for ML models.",
)


def _utc_timestamp() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


@app.command()
def run(
    out: Path = typer.Option(
        Path("artifacts/run.json"), "--out", "-o", help="Artifact output path."
    ),
    device: str = typer.Option("auto", "--device", "-d", help="Device: auto, cpu, or cuda."),
    models: list[str] | None = typer.Option(
        None, "--model", "-m", help="Model key to run; repeatable. Defaults to all models."
    ),
    warmup: int = typer.Option(5, help="Warmup iterations discarded before timing."),
    iters: int = typer.Option(20, help="Timed iterations per repeat block."),
    repeats: int = typer.Option(5, help="Number of repeat blocks; each yields one latency sample."),
    seed: int = typer.Option(1234, help="Random seed for determinism."),
) -> None:
    """Benchmark the reference models across the configuration matrix."""
    from .models import default_model_specs, get_model_spec, set_determinism
    from .runner import benchmark
    from .validate import attach_validation

    resolved = resolve_device(device)
    set_determinism(seed)
    configs = default_configs(resolved)

    specs = [get_model_spec(key) for key in models] if models else default_model_specs()

    all_records = []
    for spec in specs:
        typer.echo(f"Running {spec.key} on {resolved.value} ...")
        model = spec.loader()
        inputs = spec.input_factory()
        model_records = []
        model_logits = {}
        for config in configs:
            result = benchmark(model, inputs, config, warmup=warmup, iters=iters, repeats=repeats)
            if result.compile_failed:
                typer.echo(f"  warning: torch.compile failed for {config.name}; ran eager.")
            record = _record_from_result(spec.key, result)
            model_records.append(record)
            model_logits[config.name] = result.logits
            typer.echo(
                f"  {config.name}: median {result.metrics.median_latency_ms:.2f} ms, "
                f"p90 {result.metrics.p90_latency_ms:.2f} ms, "
                f"{result.metrics.throughput_items_per_s:.1f} items/s"
            )
        attach_validation(model_records, model_logits)
        all_records.extend(model_records)

    metadata = _build_metadata(seed)
    artifact = RunArtifact(metadata=metadata, results=all_records)
    written = save_artifact(artifact, out)
    typer.echo(f"Wrote artifact to {written}")


@app.command()
def validate(
    artifact: Path = typer.Argument(..., help="Run artifact to validate."),
    report: Path | None = typer.Option(None, "--report", "-r", help="Markdown report output."),
) -> None:
    """Validate numerical correctness against the fp32 eager baseline."""
    from .validate import render_markdown, validate_run

    run_artifact = load_artifact(artifact)
    report_obj = validate_run(run_artifact)
    markdown = render_markdown(report_obj)
    typer.echo(markdown)
    if report is not None:
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(markdown, encoding="utf-8")
        typer.echo(f"Wrote validation report to {report}")
    if not report_obj.passed:
        raise typer.Exit(code=1)


@app.command()
def compare(
    baseline: Path = typer.Option(..., "--baseline", "-b", help="Baseline artifact path."),
    current: Path = typer.Option(
        Path("artifacts/run.json"), "--current", "-c", help="Current artifact."
    ),
    margin: float = typer.Option(
        0.05, "--margin", help="Relative slowdown margin before flagging."
    ),
    report: Path | None = typer.Option(None, "--report", "-r", help="Markdown report output."),
    fail_on_regression: bool = typer.Option(
        True,
        "--fail-on-regression/--no-fail-on-regression",
        help=(
            "Exit non-zero when a regression is detected. Disable on shared CI runners "
            "where latency variance is high; keep enabled on dedicated hardware."
        ),
    ),
) -> None:
    """Detect performance regressions against a stored baseline."""
    from .compare import CompareReport, detect_regressions, render_markdown

    if not baseline.exists():
        report_obj = CompareReport(comparisons=[], baseline_present=False)
        markdown = render_markdown(report_obj, margin=margin)
        typer.echo(markdown)
        typer.echo(f"No baseline at {baseline}; skipping regression check.")
        return

    current_artifact = load_artifact(current)
    baseline_artifact = load_artifact(baseline)
    report_obj = detect_regressions(current_artifact, baseline_artifact, margin=margin)
    markdown = render_markdown(report_obj, margin=margin)
    typer.echo(markdown)
    if report is not None:
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(markdown, encoding="utf-8")
        typer.echo(f"Wrote regression report to {report}")
    if report_obj.has_regression:
        if fail_on_regression:
            raise typer.Exit(code=1)
        typer.echo("Regression detected; not failing because --no-fail-on-regression is set.")


@app.command(name="update-baseline")
def update_baseline(
    from_artifact: Path = typer.Option(..., "--from", help="Source run artifact."),
    out: Path = typer.Option(
        Path("baselines/cpu_ci.json"), "--out", "-o", help="Baseline output path."
    ),
) -> None:
    """Promote a run artifact to the stored baseline."""
    artifact = load_artifact(from_artifact)
    written = save_artifact(artifact, out)
    typer.echo(f"Updated baseline at {written}")


@app.command()
def plot(
    artifact: Path = typer.Argument(..., help="Run artifact to plot."),
    out: Path = typer.Option(
        Path("docs/latency_comparison.png"), "--out", "-o", help="PNG output."
    ),
) -> None:
    """Render the latency comparison plot from a run artifact."""
    from .plotting import plot_latency_comparison

    run_artifact = load_artifact(artifact)
    written = plot_latency_comparison(run_artifact, out)
    typer.echo(f"Wrote plot to {written}")


@app.command()
def version() -> None:
    """Print the installed version."""
    typer.echo(__version__)


def _record_from_result(model_key: str, result):
    from .artifacts import ResultRecord

    config = result.config
    return ResultRecord(
        model=model_key,
        config=config.name,
        device=config.device.value,
        precision=config.precision.value,
        mode=config.mode.value,
        is_baseline=config.is_baseline,
        metrics=result.metrics,
        latency_samples_ms=list(result.latency_samples_ms),
    )


def _build_metadata(seed: int) -> RunMetadata:
    from .hardware import git_sha, hardware_info, library_versions

    return RunMetadata(
        timestamp=_utc_timestamp(),
        git_sha=git_sha(),
        seed=seed,
        hardware=hardware_info(),
        library_versions=library_versions(),
    )


if __name__ == "__main__":
    app()
