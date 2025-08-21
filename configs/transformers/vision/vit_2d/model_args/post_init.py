import warnings

class ModelArgsAssertions:
    """Assertions for model arguments."""
    @staticmethod
    def validate(args) -> None:
        """Validate model arguments through assertions/ValueErrors.
        
        Args:
            args: Model arguments to be used for assertions.
        """
        if args.d_model % args.num_heads != 0:
            raise ValueError(
                f"Expected d_model to be divisible by num_heads, "
                f"got {args.d_model} % {args.num_heads} != 0."
            )

        if args.num_heads % args.query_groups != 0:
            raise ValueError(
                f"Expected num_heads to be divisible by query_groups, "
                f"got {args.num_heads} % {args.query_groups} != 0."
            )

        if args.d_model * 4 != args.d_ffn:
            raise ValueError(
                f"Expected d_ffn = d_model * 4, "
                f"got {args.d_model} * 4 != {args.d_ffn}"
            )
        
        if args.target_size % args.patch_size != 0:
            raise ValueError(
                f"target_size must be divisble by patch size, "
                f"got {args.target_size} % {args.patch_size} != 0."
            )
        
        if args.left_window == -1 and args.right_window == -1 and not args.use_windowed_attn:
            warnings.warn(
                "Sliding window attention not being used. Using global attention."
            )

        if not args.use_checkpointing:
            warnings.warn(
                f"Gradient checkpointing is currently False. It is highly recommended "
                f"to enable it when training large models."
            )

class PostInitMixin:
    """Post initialization interface."""
    def __post_init__(self):
        ModelArgsAssertions.validate(self)
