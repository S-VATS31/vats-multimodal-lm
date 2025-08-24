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

class PostInitMixin:
    """Post initialization interface."""
    def __post_init__(self):
        ModelArgsAssertions.validate(self)
