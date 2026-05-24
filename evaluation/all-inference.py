import torch
import gc
from transformers import (
    LlamaForCausalLM, 
    MistralForCausalLM, 
    FalconForCausalLM, 
    PreTrainedTokenizerFast
)

tokenizer_file = "ro_tokenizer_40k.json"
tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_file)
tokenizer.pad_token = tokenizer.eos_token
device = "cuda" if torch.cuda.is_available() else "cpu"

test_prompts = [
    "Când eram copil obișnuiam să",
    "Conform prevederilor legale în vigoare, acest document reprezintă",
    "În urma evenimentelor recente din capitală, autoritățile au declarat că",
    "Pentru a configura corect acest sistem, utilizatorul trebuie să urmeze următorii pași:"
]

models_to_test = [
    {"name": "Llama-GQA", "path": "models/models-good/llama_gqa/final", "class": LlamaForCausalLM},
    {"name": "Llama-MHA", "path": "models/models-good/llama_mha_baseline/final", "class": LlamaForCausalLM},
    {"name": "Falcon-MQA", "path": "models/models-good/falcon_mqa/final", "class": FalconForCausalLM},
    {"name": "Mistral-SWA", "path": "models/models-good/mistral_sliding/final", "class": MistralForCausalLM},
]

gen_scenarios = [
    {"label": "Conservative", "temp": 0.3, "top_k": 40, "top_p": 0.85},
    {"label": "Balanced", "temp": 0.8, "top_k": 50, "top_p": 0.95},
    {"label": "Creative", "temp": 1.2, "top_k": 80, "top_p": 0.98},
]

for prompt_text in test_prompts:
    print(f"\n\ntest prompt: '{prompt_text}'")
    inputs = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).to(device)
    if "token_type_ids" in inputs:
        del inputs["token_type_ids"]

    for m_info in models_to_test:
        print(f"\narch: {m_info['name']}")
        
        try:
            model = m_info['class'].from_pretrained(m_info['path'], local_files_only=True).to(device)
            model.eval()

            for scenario in gen_scenarios:
                print(f"\n [{scenario['label']} T={scenario['temp']}  K={scenario['top_k']}]")
                
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=100,
                        do_sample=True,
                        temperature=scenario['temp'],
                        top_k=scenario['top_k'],
                        top_p=scenario['top_p'],
                        repetition_penalty=1.1,
                        eos_token_id=tokenizer.eos_token_id,
                    )

                full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                print(f"{full_text}")

            del model
            gc.collect()
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"failed to load {m_info['name']}: {e}")

