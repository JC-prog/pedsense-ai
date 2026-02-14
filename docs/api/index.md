# API Reference

Module-level documentation for the PedSense-AI Python package.

## Package Structure

```
pedsense/
    config.py           # Path constants and defaults
    cli.py              # Typer CLI entry point
    processing/
        annotations.py  # XML parsing → dataclasses
        frames.py       # Frame extraction from MP4
        yolo_format.py  # YOLO dataset conversion
        resnet_format.py # ResNet sequence conversion
    train/
        yolo_trainer.py    # YOLO26 training
        resnet_lstm.py     # Model architectures (ResNetLSTM, ResNetClassifier)
        resnet_trainer.py  # ResNet+LSTM training loop
        hybrid_trainer.py  # Hybrid pipeline training
    demo.py             # Gradio web interface
```

## Modules

- [Config](config.md) — Path constants, training defaults
- [Processing](processing.md) — Annotation parsing, frame extraction, format converters
- [Training](training.md) — Model definitions and training functions
- [Demo](demo-module.md) — Gradio interface and inference
