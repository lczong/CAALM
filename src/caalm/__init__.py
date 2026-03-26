__version__ = "1.0.0"


def __getattr__(name):
    if name == "PredictionPipeline":
        from .pipeline import PredictionPipeline

        return PredictionPipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["PredictionPipeline"]
