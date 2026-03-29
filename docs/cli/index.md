# CLI Reference

PedSense provides a command-line interface built with [Typer](https://typer.tiangolo.com/) and [Rich](https://github.com/Textualize/rich).

## Usage

```bash
uv run pedsense [COMMAND] [OPTIONS]
```

## Commands

| Command | Description |
|---------|-------------|
| [`setup`](setup.md) | Create project directory structure |
| [`attributes`](attributes.md) | List available annotation attributes and class values |
| [`preprocess`](preprocess.md) | Extract frames and prepare datasets |
| [`train`](train.md) | Train YOLO26, ResNet+LSTM, or Hybrid models |
| [`resume`](resume.md) | Resume training a YOLO model for additional epochs |
| [`demo`](demo.md) | Launch Gradio web interface |
| [`convert-sequences`](convert-sequences.md) | Convert CSV keypoint sequences to npy for training |

## Global Options

| Option | Description |
|--------|-------------|
| `--help` | Show help message and exit |
| `--install-completion` | Install shell completion |
| `--show-completion` | Show shell completion script |
