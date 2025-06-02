import os
import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    PreTrainedModel,
    pipeline
)
from datasets import Dataset, load_dataset
from typing import Optional, Type, Union
import torch.nn.functional as F
import numpy as np
from trl import DPOTrainer, DPOConfig, ModelConfig,get_quantization_config,get_kbit_device_map

QWEN_SFT_MODEL = "august66/qwen2-sft-final"
DPO_TRAIN_DATA = 'august66/reward_data_for_dpo_train'


#This script only train dpo model, no calculation of scores, no preparation of dataset, no training of qwen
def dpo_train(
    lm_model: Optional[str] = None,
    train_data: Optional[Union[Dataset, str]] = None,
    beta: Optional[Union[int, float]] = None,
    device: Optional[str] = None
) -> tuple[PreTrainedModel, PreTrainedModel]:

    if beta is None:
        beta = 0.1
    elif not isinstance(beta, (int, float)):
        raise ValueError("beta must be an integer or a float.")

    if isinstance(train_data, str):
        train_data = load_dataset(train_data)
    elif isinstance(train_data, Dataset):
        train_data = train_data
    else:
        raise ValueError("train_data must be a string (dataset name) or a Dataset instance.")

    if device is None or device == 'cuda' or device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    elif device == 'cpu':
        device = 'cpu'
    else:
        raise ValueError(f"Unsupported device: {device}. Use 'cuda', 'cpu', or 'auto'.")

    if lm_model is None or not isinstance(lm_model, str):
        raise ValueError("lm_model must be a string representing the model path.")
    

    model_args = ModelConfig(lm_model)
    model_torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ['auto', None] else torch.float16
    )

    model_kwargs = dict(
        revision = model_args.model_revision,
        torch_dtype = model_torch_dtype, 
        trust_remote_code = model_args.trust_remote_code,
    )
    
    lm_model_instance = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path, 
        **model_kwargs
    ).to(device)

    ref_model_instance = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path, 
        **model_kwargs
    ).to(device)

    lm_model_tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, 
        padding_side = 'left', 
        use_fast = True,
        trust_remote_code = model_args.trust_remote_code
    )

    if not lm_model_tokenizer.pad_token:
        lm_model_tokenizer.pad_token = lm_model_tokenizer.eos_token

    training_args = DPOConfig(

            gradient_checkpointing=False,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=2,
            learning_rate=5.0e-7,
            logging_steps= 500,
            num_train_epochs=1,
            push_to_hub=True,  
            output_dir = "/root/autodl-tmp/.autodl/DPO_tldr",
            report_to = 'none',
            beta = beta,
            hub_model_id = f'august66/qwen2-sft-dpo-imdb-beta-{beta}',
            save_strategy="no", 
    )

    trainer = DPOTrainer(
        model=lm_model_instance,
        ref_model=ref_model_instance,
        args=training_args,
        train_dataset=train_data,
        processing_class =  lm_model_tokenizer
    )

    trainer.train()


if __name__ == '__main__':

    beta_list = [0, 0.25, 0.5, 0.75, 1.0]

    for beta in beta_list:
        print(f"Training DPO model with beta = {beta}")
        dpo_train(
            lm_model=QWEN_SFT_MODEL,
            train_data=DPO_TRAIN_DATA,
            beta=beta,
            device='cuda'
        )
        print(f"DPO model with beta = {beta} trained successfully.")


