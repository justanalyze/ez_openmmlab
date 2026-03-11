from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from .model_factory import ModelFactory

app = typer.Typer(
    help="ez-mmlab: A EZ CLI for the OpenMMLab ecosystem.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()


@app.command()
def train(
    model_name: str = typer.Argument(
        ..., help="Name of the model (e.g., 'rtmdet_tiny') or path to 'config.toml'"
    ),
    dataset_config: Path = typer.Argument(..., help="Path to your 'dataset.toml' file"),
    # General training settings
    epochs: int = typer.Option(100, help="Total training epochs"),
    batch_size: int = typer.Option(8, help="Total batch size"),
    lr: float = typer.Option(0.001, help="Initial learning rate"),
    device: str = typer.Option("cuda", help="Hardware device (cuda, cpu, mps)"),
    work_dir: str = typer.Option("./runs/train", help="Output directory"),
    num_workers: int = typer.Option(4, help="Data loader workers"),
    amp: bool = typer.Option(True, "--amp/--no-amp", help="Use Mixed Precision"),
    # Visualization
    tensorboard: bool = typer.Option(False, help="Enable TensorBoard logging"),
):
    """Starts a training run on a custom dataset."""
    try:
        model = ModelFactory.get_model(model_name)
        model.train(
            dataset_config_path=dataset_config,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=lr,
            device=device,
            work_dir=work_dir,
            num_workers=num_workers,
            amp=amp,
            enable_tensorboard=tensorboard,
        )
    except Exception as e:
        console.print(f"[bold red]Training Failed:[/bold red] {e}")
        raise typer.Exit(code=1)


@app.command()
def predict(
    model_name: str = typer.Argument(..., help="Model name (e.g., 'rtmdet_tiny')"),
    image: Path = typer.Argument(..., help="Path to image, folder, or list of images"),
    # Weights and Logic
    weights: Optional[Path] = typer.Option(None, help="Path to custom .pth weights"),
    device: str = typer.Option("cuda", help="Inference device (cuda, cpu)"),
    confidence: float = typer.Option(
        0.3, "--conf", help="General detection confidence threshold"
    ),
    # Pose-Specific Parameters
    bbox_thr: float = typer.Option(0.3, help="BBox threshold for top-down pose"),
    kpt_thr: float = typer.Option(0.3, help="Keypoint threshold for pose"),
    # Output and Visualization
    output: Optional[str] = typer.Option(
        "runs/preds", "--out", "-o", help="Directory for results"
    ),
    show: bool = typer.Option(True, help="Display the results window"),
    log_level: str = typer.Option("INFO", help="Framework log level"),
):
    """Performs inference (Detection, Segmentation, or Pose) on images."""
    try:
        # Instantiate model via factory
        model = ModelFactory.get_model(
            model_name, checkpoint_path=weights, log_level=log_level
        )

        # Run prediction
        # (predict() automatically handles detection vs pose via kwargs)
        results = model.predict(
            image_path=image,
            device=device,
            out_dir=output,
            show=show,
            # Kwargs forwarded to child engines
            confidence=confidence,
            bbox_thr=bbox_thr,
            kpt_thr=kpt_thr,
        )

        console.print(f"\n[bold green]Success![/bold green] Results saved to: {output}")
        console.print(f"Processed {len(results)} image(s).")

    except Exception as e:
        console.print(f"[bold red]Prediction Failed:[/bold red] {e}")
        raise typer.Exit(code=1)


@app.command()
def export(
    model_name: str = typer.Argument(..., help="Model name (e.g., 'rtmdet_tiny')"),
    image: Path = typer.Argument(..., help="Sample image path for model tracing"),
    # Export Options
    format: str = typer.Option("onnx", help="Target format (onnx, tensorrt)"),
    weights: Optional[Path] = typer.Option(None, help="Path to custom .pth weights"),
    num_classes: Optional[int] = typer.Option(None, help="Number of classes for custom weights"),
    output: str = typer.Option("runs/deploy", "--out", "-o", help="Output directory"),
    device: str = typer.Option("cpu", help="Device for export (cpu, cuda)"),
):
    """Exports a model to a production format (ONNX, TensorRT) via Docker."""
    try:
        # Instantiate model via factory
        model = ModelFactory.get_model(
            model_name, checkpoint_path=weights, num_classes=num_classes
        )

        # Run export
        model.export(
            format=format,
            image=image,
            output_dir=output,
            device=device,
        )

        console.print(f"\n[bold green]Export Successful![/bold green]")
        console.print(f"Artifacts saved to: [cyan]{output}[/cyan]")

    except Exception as e:
        console.print(f"[bold red]Export Failed:[/bold red] {e}")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
