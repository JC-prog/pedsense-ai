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
| [`preprocess`](preprocess.md) | Extract frames and prepare datasets |
| [`train`](train.md) | Train YOLO26, ResNet+LSTM, or Hybrid models |
| [`demo`](demo.md) | Launch Gradio web interface |

## Global Options

| Option | Description |
|--------|-------------|
| `--help` | Show help message and exit |
| `--install-completion` | Install shell completion |
| `--show-completion` | Show shell completion script |
