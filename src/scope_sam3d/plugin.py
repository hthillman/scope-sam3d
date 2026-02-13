import scope.core


@scope.core.hookimpl
def register_pipelines(register):
    from .pipeline import SAM3DPipeline

    register(SAM3DPipeline)
