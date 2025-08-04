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

        if len(args.window_size) != 2:
            raise ValueError(
                f"Expected len(window_size) to be equal to 2, "
                f"got {len(args.window_size)}"
            )

        if args.window_size[0] != args.window_size[1]:
            raise ValueError(
                f"Expected left and right windows to be equal, "
                f"got {args.window_size[0]} != {args.window_size[1]}"
            )

        if len(args.patch_size) != 3:
            raise ValueError(
                f"Expected len(patch_size) == 3 for T, H, W dimensions, "
                f"got {len(args.patch_size)} != 3."
            )

        if len(args.target_size) != 2:
            raise ValueError(
                f"len(target_size) must be 2 for H/W dimensions, "
                f"got {len(args.target_size)}"
            )

        if args.max_frames % args.patch_size[0] != 0:
            raise ValueError(
                f"max_frames must be divisible by patch frames, "
                f"got {args.max_frames} % {args.patch_size[0]} != 0"
            )

        if (
            args.target_size[0] % args.patch_size[1] != 0 or
            args.target_size[1] % args.patch_size[2] != 0
        ):
            raise ValueError(
                f"target H/W sizes must be divisible by patch sizes, "
                f"got {args.target_size[0]} % {args.patch_size[1]} != 0 "
                f"or {args.target_size[1]} % {args.patch_size[2]} != 0"
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
