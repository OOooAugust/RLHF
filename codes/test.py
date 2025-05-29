import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset

#tokenize all batch at the same time, use map and in cpu
# for model generation, better in gpu using dataloader to manage batch 
# use fp16 instead of fp32, but what's the difference?



def make_completions(batch):
    prompts = batch['prompt']
    batch_size = len(prompts)
    
    inputs = sft_tokenizer(
        text = prompts,
        padding = True,
        truncation = True,
        max_length = 128,
        padding_side = 'left',
        add_special_tokens = True,
        return_tensors='pt',
    ).to(device)

    with torch.inference_mode():
        outputs = sft_model.generate(
            **inputs,
            num_return_sequences=2,
            do_sample = True,
            top_k=50,
            top_p=0.95,
            temperature=1.0,
            use_cache = True,
            pad_token_id = sft_model.config.eos_token_id,
            output_scores=False,
            return_dict_in_generate=False,
            max_new_tokens=100
        ).view((batch_size, 2, -1)).cpu()
    
    comp1, comp2 = [], []
    for output in outputs:
        completion_texts = sft_tokenizer.batch_decode(
            output,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        comp1.append(completion_texts[0])
        comp2.append(completion_texts[1])
    
    return {'completion_1':comp1, 'completion_2':comp2}



if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sft_model = AutoModelForCausalLM.from_pretrained(
        "august66/qwen2-sft-final",
        torch_dtype=torch.float16,
        ).eval()
    sft_tokenizer = AutoTokenizer.from_pretrained("august66/qwen2-sft-final") 

    dataset_test = load_dataset("stanfordnlp/imdb", split="test")

    def prompt_completion_preprocess(example):
        words = example['text'].split()
        prompt = ' '.join(words[:5])
        completion = ' '.join(words[5:])
        return {'prompt': prompt, 'completion': completion}

    dataset_test = dataset_test.map(prompt_completion_preprocess, remove_columns=['text', 'label'])

    sentiment_model = "siebert/sentiment-roberta-large-english"
    sentiment_tokenizer = AutoTokenizer.from_pretrained(sentiment_model)
    sentiment_model = AutoModelForSequenceClassification.from_pretrained(sentiment_model)

    print(f"Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"Reserved : {torch.cuda.memory_reserved()  / 1024**3:.2f} GB")


    sft_model = sft_model.to(device)
    sft_tokenizer.pad_token = sft_tokenizer.eos_token
    sft_model.config.pad_token_id = sft_model.config.eos_token_id

    new_data = dataset_test.map(
        make_completions,
        batched = True,
        batch_size = 64,
        remove_columns = ['prompt'])


    

