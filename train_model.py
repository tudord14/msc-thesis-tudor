from transformers import (
    LlamaConfig, LlamaForCausalLM,
    MistralConfig, MistralForCausalLM,
    FalconConfig, FalconForCausalLM,
    Mamba2Config, Mamba2ForCausalLM,
    PreTrainedTokenizerFast, Trainer, TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
import copy

# -> we will train some model architectures: Llama, Mistral, Falcon, Mamba and a Llama-MHA baseline
# -> the idea is to have models of similar size and context window but different intelligence methods
# -> this way we can find out, for small language models, which architecture is best at understanding language
# -> corpus: 5.6B tokens Romanian, so ~300M params is chinchilla-optimal (20 tokens/param)
# -> architecture choices and what each one tests:
# -> Llama (GQA): Grouped Query Attention -> fewer KV heads, memory efficient
# -> Mistral (Sliding + GQA): Sliding Window Attention -> local context window within GQA
# -> Falcon (Parallel + MQA): Multi-Query Attention + Parallel attn/MLP -> max KV compression
# -> Mamba2 (SSM): State Space Model -> no attention at all, linear scaling
# -> Llama-MHA (baseline): Standard Multi-Head Attention -> full KV heads, classical transformer

CONTEXT_LENGTH = 2048
VOCAB_SIZE = 40000
TOKENIZER_NAME = 'tokenizer'
# -> all models target ~300-310M parameters
# -> shared dimensions: hidden=1024, heads=16, head_dim=64, intermediate=2816
# -> each architecture has different num_layers to compensate for param count differences
HIDDEN_SIZE = 1024
NUM_HEADS = 16          
INTERMEDIATE_SIZE = 2816

# 1. LLAMA — GQA (Grouped Query Attention)
# -> 4 KV heads for 16 Q heads -> 4:1 group ratio
# -> 23 layers -> ~300M params
llama_config = LlamaConfig(
    vocab_size=VOCAB_SIZE,
    hidden_size=HIDDEN_SIZE,
    num_hidden_layers=23,
    num_attention_heads=NUM_HEADS,
    num_key_value_heads=4,           
    intermediate_size=INTERMEDIATE_SIZE,
    max_position_embeddings=CONTEXT_LENGTH,
    rms_norm_eps=1e-6,
    hidden_act="silu",
    tie_word_embeddings=True
)

# 2. MISTRAL —> Sliding Window Attention + GQA
# -> same params as Llama GQA (sliding window doesn't add parameters)
# -> isolates the effect of local vs global attention
# -> 23 layers -> ~300M params
mistral_config = MistralConfig(
    vocab_size=VOCAB_SIZE,
    hidden_size=HIDDEN_SIZE,
    num_hidden_layers=23,
    num_attention_heads=NUM_HEADS,
    num_key_value_heads=4,               
    intermediate_size=INTERMEDIATE_SIZE,
    sliding_window=512,                
    max_position_embeddings=CONTEXT_LENGTH,
    rms_norm_eps=1e-6,
    hidden_act="silu",
    tie_word_embeddings=True
)

# 3. FALCON —> Parallel Attention + Multi-Query Attention (MQA)
# -> 1 KV head (maximum KV compression), attn and MLP computed in parallel
# -> Falcon uses standard GELU MLP (2 matrices, 4x hidden) not SwiGLU
# -> 25 layers -> ~307M params
falcon_config = FalconConfig(
    vocab_size=VOCAB_SIZE,
    hidden_size=HIDDEN_SIZE,
    num_hidden_layers=25,
    num_attention_heads=NUM_HEADS,
    num_kv_heads=1,                       
    parallel_attn=True,      
    new_decoder_architecture=True,
    max_position_embeddings=CONTEXT_LENGTH,
    bias=False,
    tie_word_embeddings=True
)

# 4. MAMBA2 —> State Space Model (SSM)
# -> no attention mechanism at all — uses selective state spaces
# -> SSM layers have fewer params than transformer layers -> need more layers
# -> 40 layers -> ~305M params
mamba_config = Mamba2Config(
    vocab_size=VOCAB_SIZE,
    hidden_size=HIDDEN_SIZE,
    num_hidden_layers=40,
    state_size=128,
    expand=2,
    n_groups=1,
    rms_norm=True,
    chunk_size=256,
    tie_word_embeddings=True
)

# 5. LLAMA-MHA —> Standard Multi-Head Attention (baseline)
# -> every head has its own KV —> maximum representational power, most KV memory
# -> more KV params per layer -> fewer layers to match budget
# -> 21 layers -> ~311M params
llama_mha_config = LlamaConfig(
    vocab_size=VOCAB_SIZE,
    hidden_size=HIDDEN_SIZE,
    num_hidden_layers=21,
    num_attention_heads=NUM_HEADS,
    num_key_value_heads=NUM_HEADS,        
    intermediate_size=INTERMEDIATE_SIZE,
    max_position_embeddings=CONTEXT_LENGTH,
    rms_norm_eps=1e-6,
    hidden_act="silu",
    tie_word_embeddings=True
)

arch_selectors = [
    (LlamaForCausalLM, llama_config, 'llama_gqa'),
    (MistralForCausalLM, mistral_config, 'mistral_sliding'),
    (FalconForCausalLM, falcon_config, 'falcon_mqa'),
    (Mamba2ForCausalLM, mamba_config, 'mamba2_ssm'),
    (LlamaForCausalLM, llama_mha_config, 'llama_mha_baseline'),
]

print("param count check")
param_counts = {}
for model_class, model_config, model_name in arch_selectors:
    model = model_class(model_config)
    total = sum(p.numel() for p in model.parameters())
    param_counts[model_name] = total
    print(f" {model_name:25s} -> {total / 1e6:8.1f}M params")
    del model

# -> check that all models are within 15% of each other
counts = list(param_counts.values())
min_c, max_c = min(counts), max(counts)
spread_pct = (max_c - min_c) / min_c * 100
print(f"smallest: {min_c/1e6:.1f}M | largest: {max_c/1e6:.1f}M | spread: {spread_pct:.1f}%")
if spread_pct > 15:
    print("warning: models differ by more than 15% -> Adjust num_layers.")
else:
    print("models are within 15% of each other —> fair comparison.")


# -> data loading
TRAINING_CORPUS = '/Volumes/KINGSTON/Packed_File.jsonl'
OUTPUT_DIR = 'output'
# -> tokenizer only needed for the data collator (padding/eos token config)
tokenizer = PreTrainedTokenizerFast(tokenizer_file=TOKENIZER_NAME)
tokenizer.pad_token = tokenizer.eos_token

def add_labels(examples):
    # -> for causal LM, labels = input_ids (model shifts them internally)
    examples['labels'] = examples['input_ids'].copy()
    return examples

print("loading pre-tokenized data...")
raw_dataset = load_dataset('json', data_files=TRAINING_CORPUS, split='train')
print(f"loaded {len(raw_dataset)} packed sequences")
dataset = raw_dataset.map(add_labels, batched=True)

# -> split into train/eval
dataset = dataset.train_test_split(test_size=0.05, seed=42)
print(f"train: {len(dataset['train'])} sequences | eval: {len(dataset['test'])} sequences")

# -> training arguments
print("initializing training arguments...")
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    lr_scheduler_type='cosine',
    warmup_steps=100,
    weight_decay=0.01,
    num_train_epochs=1,
    logging_steps=10,
    eval_strategy='steps',
    eval_steps=500,
    save_strategy='steps',
    save_steps=1000,
    save_total_limit=2,
    fp16=True,
    gradient_checkpointing=True,
    report_to='tensorboard',
)

# -> train llama first (GQA) to verify everything works
model_class, model_config, model_name = arch_selectors[0]
print(f"training {model_name}...")
model = model_class(model_config)
print(f"model size: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params")

model_args = copy.deepcopy(training_args)
model_args.output_dir = f"{OUTPUT_DIR}/{model_name}"

trainer = Trainer(
    model=model,
    args=model_args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['test'],
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
)

trainer.train()
trainer.save_model(f"{OUTPUT_DIR}/{model_name}/final")
del model, trainer 