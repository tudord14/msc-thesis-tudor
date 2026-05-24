import torch
import time
from transformers import LlamaForCausalLM, PreTrainedTokenizerFast, MistralForCausalLM, FalconForCausalLM

# -> some paths
model_path = "models/models-good/llama_gqa/final"
#model_path = "models/models-good/llama_mha/final"
#model_path = "models/models-good/falcon_mqa/final"
#model_path = "models/models-good/mistral_swa/final"

# model_path = "models/models-bad/falcon_mqa/final"
tokenizer_file = "ro_tokenizer_40k.json"

tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_file)
tokenizer.pad_token = tokenizer.eos_token

model = LlamaForCausalLM.from_pretrained(model_path)
model.eval()
model.to("cuda" if torch.cuda.is_available() else "cpu")

prompt = "Cand eram copil obisnuiam sa"

inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)

if "token_type_ids" in inputs:
    del inputs["token_type_ids"]

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.8,
        repetition_penalty=1.1,
        eos_token_id=tokenizer.eos_token_id,
    )

output_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
print(f"{prompt}")
print("\nModel Output:\n", output_text)

