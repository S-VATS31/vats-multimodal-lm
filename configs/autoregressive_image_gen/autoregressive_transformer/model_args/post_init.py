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
        
        if not 0 < model_args.dropout < 1:
            raise ValueError(
                f"dropout must be between 0 and 1, got {model_args.dropout}"
            )
        
        if model_args.use_ntk_rope and model_args.ntk_scale_factor is None:
            raise ValueError(
                "must be given ntk_scale_factor for NTK RoPE."
            )

class PostInitMixin:
    """Post initialization interface."""
    def __post_init__(self):
        ModelArgsAssertions.validate(self)