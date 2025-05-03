import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType
from trl import DPOTrainer, DPOConfig

def load_dpo_dataset(path: str, split_ratio=0.2):
    print(f"📂 Loading dataset from: {path}")
    dataset = load_dataset("json", data_files=path, split="train")
    print(f"✅ Loaded {len(dataset)} total examples.")

    # Split into train and validation
    dataset = dataset.train_test_split(test_size=split_ratio, seed=42)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]
    print(f"📊 Train: {len(train_dataset)} | Validation: {len(eval_dataset)}")
    return train_dataset, eval_dataset

def get_tokenizer(model_name: str):
    return AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

def get_lora_model_4bit(model_name: str):
    print(f"🔧 Loading 4-bit quantized model: {model_name}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        trust_remote_code=True
    )

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False
    )

    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()
    return model

def configure_dpo_training(output_dir: str = "./dpo_planner_model"):
    return DPOConfig(
        output_dir=output_dir,
        per_device_train_batch_size=2,  # 🔽 reduce batch size
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=2,
        learning_rate=5e-5,
        num_train_epochs=3,
        evaluation_strategy="epoch",
        logging_steps=10,
        save_strategy="epoch",
        beta=0.1,
        max_length=512,             # 🔽 reduce max input length
        max_prompt_length=256,      # 🔽 reduce prompt length
        report_to="none"
    )

def train_dpo_model(model, tokenizer, train_dataset, eval_dataset, training_args):
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer  # deprecated but still usable unless removed in your version
    )

    print("🚀 Starting training...")
    trainer.train()

    model.save_pretrained(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)
    print(f"💾 Model checkpoint saved to {training_args.output_dir}")

    print("📈 Evaluating on train set...")
    train_metrics = trainer.evaluate(eval_dataset=train_dataset, metric_key_prefix="train")
    print(train_metrics)

    print("📈 Evaluating on validation set...")
    eval_metrics = trainer.evaluate(eval_dataset=eval_dataset, metric_key_prefix="eval")
    print(eval_metrics)

def main():
    model_name = "deepseek-ai/deepseek-llm-7b-chat"
    dataset_path = "/work/users/c/h/chidaksh/hackathons/hack2/combined_dpo_data.json"
    output_dir = "/work/users/c/h/chidaksh/hackathons/hack2/dpo_planner_model"

    tokenizer = get_tokenizer(model_name)
    model = get_lora_model_4bit(model_name)
    train_dataset, eval_dataset = load_dpo_dataset(dataset_path)
    training_args = configure_dpo_training(output_dir)

    train_dpo_model(model, tokenizer, train_dataset, eval_dataset, training_args)

if __name__ == "__main__":
    main()
