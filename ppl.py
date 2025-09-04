import math

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def compute_ppl(model, inputs):
    outputs = model(
        input_ids=inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        labels=inputs['input_ids'])
    return outputs.loss.exp().item() / math.log(2)


def main():
    model_name = 'distilbert/distilgpt2'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).cuda()
    model.eval()

    text = open('text.txt').read().strip()
    inputs = tokenizer(text, return_tensors='pt').to('cuda')
    inputs = {k: v.cuda() for k, v in inputs.items()}

    max_len = min(tokenizer.model_max_length, model.config.n_positions)
    assert inputs['input_ids'].shape[1] < max_len, "Input text is too long"

    ppl = compute_ppl(model, inputs)
    print(f"Before shuffle: ppl={ppl:.2f}")

    perm = torch.randperm(inputs['input_ids'].shape[1])
    inputs['input_ids'] = inputs['input_ids'][:, perm]

    ppl = compute_ppl(model, inputs)
    print(f"After shuffle: ppl={ppl:.2f}")


if __name__ == "__main__":
    main()
