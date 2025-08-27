from configs.transformers.nlp.training_args import TrainingArgs

class ModelArgsAssertions:
    """Assertions for model arguments."""
    @staticmethod
    def validate(model_args) -> None:
        """Validate model arguments through assertions/ValueErrors.
        
        Args:
            model_args: Model arguments to be used for assertions/validation.
        """
        if model_args.d_model % model_args.num_heads != 0:
            raise ValueError(
                f"Expected d_model to be divisble by num_heads, "
                f"got {model_args.d_model} % {model_args.num_heads} != 0"
            )
        
        if model_args.num_heads % model_args.query_groups != 0:
            raise ValueError(
                f"Expected d_model to be divisble by num_heads, "
                f"got {model_args.num_heads} % {model_args.query_groups} != 0"
            )
        
        if model_args.d_model * 4 != model_args.d_ffn:
            raise ValueError(
                f"Expected d_model * 4 = d_ffn, "
                f"got {model_args.d_model} * 4 != {model_args.d_ffn}"
            )

class PostInitMixin:
    """Post initialization interface."""
    def __post_init__(self):
        ModelArgsAssertions.validate(self)
