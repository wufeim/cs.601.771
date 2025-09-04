from transformers import AutoTokenizer, AutoModelForCausalLM


def generate(inputs, tokenizer, model, temp):
    outputs = model.generate(
        **inputs,
        max_length=500,
        do_sample=temp > 0.0,
        top_k=50,
        top_p=0.95,
        temperature=temp)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def main():
    model_name = 'distilbert/distilgpt2'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).cuda()
    model.eval()

    prompt = "Once upon a time"
    inputs = tokenizer(prompt, return_tensors='pt').to('cuda')
    inputs = {k: v.cuda() for k, v in inputs.items()}

    with open("text_gen_output.txt", "w") as fp:
        for temp in [0.0, 0.3, 0.6, 0.9, 1.2, 1.5]:
            fp.write(f"=== Temperature: {temp} ===\n")
            fp.write(generate(inputs, tokenizer, model, temp))
            fp.write("\n\n")


if __name__ == "__main__":
    main()
