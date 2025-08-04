import warnings

from configs.transformers.nlp.training_args import TrainingArgs
training_args = TrainingArgs()

class ModelArgsAssertions:
    """Assertions for NLP model arguments."""
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
        
        if model_args.max_batch_size < training_args.batch_size:
            raise ValueError(
                f"Expected max_batch_size >= batch_size, "
                f"got {training_args.batch_size} < {model_args.max_batch_size}"
            )
        
        if model_args.num_experts < model_args.top_k:
            raise ValueError(
                f"Expected num_experts >= top_k, "
                f"got {model_args.top_k} > {model_args.num_experts}"
            )
        
        if not model_args.use_causal:
            raise ValueError(
                "use_causal must be True for causal language modeling."
            )
        
        if model_args.right_window != 0:
            raise ValueError(
                f"right_window must be 0 for causal language modeling, "
                f"got {model_args.right_window}"
            )
        
        if model_args.left_window <= 0:
            raise ValueError(
                f"left_window must be greater than 0, "
                f"got, {model_args.left_window}"
            )
        
        if not model_args.gradient_checkpointing:
            warnings.warn(
                "gradient_checkpointing currently false, set to True to save VRAM."
            )

        if model_args.use_mqa and model_args.query_groups != 1:
            warnings.warn(
                "Got use_mqa=True, but query_groups != 1, setting attention to GQA. "
                "To use MQA, set use_mqa=True AND query_groups = 1."
            )

        if not model_args.use_mqa and model_args.query_groups == 1:
            warnings.warn(
                "Got query_groups=1, but use_mqa=False, setting attention to GQA "
                "To use MQA set query_groups = 1 AND use_mqa=True"
            )

        if model_args.top_k == 1 and model_args.num_experts == 1:
            warnings.warn(
                "Got top_k=1 and num_experts=1, MoE not being used."
            )

class PostInitMixin:
    """Post initialization interface."""
    def __post_init__(self):
        ModelArgsAssertions.validate(self)