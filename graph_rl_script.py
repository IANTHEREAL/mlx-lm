import types

from mlx_lm.lora import run

args = {
    "model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    "train": True,
    "fine_tune_type": "lora",
    "training_mode": "grpo",
    "optimizer": "adam",
    "optimizer_config": {
        "adam": {},
        "adamw": {},
    },
    "data": "material/graph_optimization_dataset",
    "seed": 0,
    "num_layers": 16,
    "batch_size": 1,
    "iters": 100,
    "val_batches": 1,
    "learning_rate": 1e-5,
    "steps_per_report": 1,
    "steps_per_eval": 10,
    "resume_adapter_file": None,
    "adapter_path": "material/test-grpo-full",
    "save_every": 10,
    "test": False,
    "test_batches": 500,
    "max_seq_length": 12048,
    "config": None,
    "grad_checkpoint": True,
    "lr_schedule": None,
    "lora_parameters": {"rank": 8, "alpha": 16, "dropout": 0.0, "scale": 10.0},
    "mask_prompt": False,
    # GRPO args
    "reference_model_path": None,
    "group_size": 2,
    "beta": 0.1,
    "epsilon": 1e-4,
    "max_completion_length": 2048,
    "use_chat_template": True,
    "use_prompt": False,
    "temperature": 0.6,
    "reward_weights": None,
}

run(types.SimpleNamespace(**args))