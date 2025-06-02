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
SENTIMENT_MODEL = "siebert/sentiment-roberta-large-english"



def sft_sample_generation(
        lm_model: Optional[str] = None,
        sentiment_model: Optional[str] = None,
        train_data: Optional[Union[Dataset, str]] = None,
        temperature: Optional[Union[int, float]] = None,
        n_prefix: Optional[int] = None,
        device: Optional[str] = None
) -> Dataset:
    
    if temperature is None:
        temperature = 1.0
    elif not isinstance(beta, (int, float)):
        raise ValueError("beta must be an integer or a float.")

    if n_prefix is None:
        raise ValueError("n_prefix must be specified as an integer.")
    
    if device is None or device == 'cuda' or device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    elif device == 'cpu':
        device = 'cpu'
    else:
        raise ValueError(f"Unsupported device: {device}. Use 'cuda', 'cpu', or 'auto'.")

    if lm_model is None or not isinstance(lm_model, str):
        raise ValueError("lm_model must be a string representing the model path.")

    if sentiment_model is None or not isinstance(sentiment_model, str):
        raise ValueError("sentiment_model must be a string representing the model path.")

    if isinstance(train_data, str):
        train_data = load_dataset(train_data)['test']
    elif isinstance(train_data, Dataset):
        train_data = train_data
    else:
        raise ValueError("train_data must be a string (dataset name) or a Dataset instance.")

    if 'prompt' not in train_data.column_names:
        raise ValueError("train_data must contain a 'prompt' column.")
    
    model_args = ModelConfig(lm_model)
    model_torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ['auto', None] else torch.float16
    )
    lm_model_instance = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path, 
        revision=model_args.model_revision,
        torch_dtype=model_torch_dtype, 
        trust_remote_code=model_args.trust_remote_code
    )

    lm_model_tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, 
        padding_side='left', 
        use_fast=True,
        trust_remote_code=model_args.trust_remote_code
    )
    if not lm_model_tokenizer.pad_token:
        lm_model_tokenizer.pad_token = lm_model_tokenizer.eos_token

    pipe_lm_sft = pipeline(
        'text-generation',
        model = lm_model_instance,
        tokenizer = lm_model_tokenizer,
        device = device,
    )
    pipe_sentiment = pipeline(
        'sentiment-analysis',
        model = sentiment_model,
        device = device
    )

    prompts = train_data['prompt']
    generated_completions = pipe_lm_sft(
        prompts,
        max_new_tokens = 128,
        do_sample = True,
        top_p = 0.95,
        top_k = 50,
        temperature = temperature,
        num_return_sequences = 2,
        batch_size = 128
    )
    generated_completions_dataset = Dataset.from_list([gen['generated_text'] for gen in generated_completions])
    sentiment_res = pipe_sentiment(
        generated_completions_dataset['generated_text'],
        batch_size = 128
    )

    prompt_completion_list_train = []
    for i in range(len(sentiment_res)):
        prompt = prompts[i]
        completion_1 = generated_completions_dataset[2*i]['generated_text']
        reward_1 = sentiment_res[2*i]['score'] if sentiment_res[2*i]['label'] == 'POSITIVE' else 1-sentiment_res[2*i]['score']
        completion_2 = generated_completions_dataset[2*i + 1]['generated_text']
        reward_2 = sentiment_res[2*i + 1]['score'] if sentiment_res[2*i + 1]['label'] == 'POSITIVE' else 1-sentiment_res[2*i + 1]['score']
        preference_prob = F.sigmoid(torch.tensor(reward_1-reward_2))
        bernoulli_indicator = torch.bernoulli(preference_prob).item()
        if bernoulli_indicator == 1:
            chosen, rejected = completion_1, completion_2
            reward_chosen, reward_rejected = reward_1, reward_2
        else:
            chosen, rejected = completion_2, completion_1
            reward_chosen, reward_rejected = reward_2, reward_1
        prompt_completion_list_train.append({
            'prompt': prompt,
            'chosen': " ".join(chosen.split()[n_prefix:]),
            'rejected': " ".join(rejected.split()[n_prefix:]),
            'reward_chosen': reward_chosen,
            'reward_rejected': reward_rejected
    })
        
    prompt_completion_dataset_train = Dataset.from_list(prompt_completion_list_train)
    prompt_completion_dataset_train.push_to_hub(
        repo_id = f'august66/reward_data_for_dpo_train_{temperature}'
    )

    return prompt_completion_dataset_train







    


    
    


#This script only train dpo model, no calculation of scores, no preparation of dataset, no training of qwen
def dpo_train(
    lm_model: Optional[str] = None,
    train_data: Optional[Union[Dataset, str]] = None,
    beta: Optional[Union[int, float]] = None,
    temperature: Optional[Union[int, float]] = None,
    device: Optional[str] = None
) -> None:

    if beta is None:
        beta = 0.1
    elif not isinstance(beta, (int, float)):
        raise ValueError("beta must be an integer or a float.")

    if isinstance(train_data, str):
        train_data = load_dataset(train_data)['train']
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


