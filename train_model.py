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
from torch.utils.data import default_collate
import torch

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
TOKENIZER_NAME = 'ro_tokenizer_40k.json'
# -> all models target ~300-310M paramters
# -> shared dimensions: hidden=1024, heads=16, head_dim=64, intermediate=2816
# -> each architecture has different num_layers to compensate for param count differences
HIDDEN_SIZE = 1024
NUM_HEADS = 16          
INTERMEDIATE_SIZE = 2816

# 1. LLAMA —> GQA (Grouped Query Attention)
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
#mamba_config = Mamba2Config(
#    vocab_size=VOCAB_SIZE,
#    hidden_size=HIDDEN_SIZE,
#    num_hidden_layers=40,
#    state_size=128,
#    expand=2,
#    n_groups=1,
#    rms_norm=True,
#    chunk_size=256,
#    tie_word_embeddings=True
#)

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
#    (LlamaForCausalLM, llama_config, 'llama_gqa'),
#    (MistralForCausalLM, mistral_config, 'mistral_sliding'),
#    (FalconForCausalLM, falcon_config, 'falcon_mqa'),
#    (Mamba2ForCausalLM, mamba_config, 'mamba2_ssm'),
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

# -> data loading — raw text jsonl, we tokenize and pack inline during dataset preparation
# -> this mirrors the old successful training approach (FULL_CORPUS_BIG.jsonl pipeline)
# -> packing inline at 4096 gives the model longer context windows for better coherence
TRAINING_CORPUS = 'preprocessing/WEB_BOOKS_LITERARY.jsonl'
OUTPUT_DIR = 'models'

tokenizer = PreTrainedTokenizerFast(tokenizer_file=TOKENIZER_NAME)
tokenizer.bos_token, tokenizer.eos_token = "<s>", "</s>"
tokenizer.unk_token, tokenizer.pad_token = "<unk>", "<pad>"
tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids("<pad>")
# -> silence max_length warning during tokenization — we handle length ourselves in pack_fn
tokenizer.model_max_length = int(1e12)

print("BOS id:", tokenizer.bos_token_id)
print("EOS id:", tokenizer.eos_token_id)
print("PAD id:", tokenizer.pad_token_id)
print("Vocab size:", tokenizer.vocab_size)

print("loading raw corpus...")
dataset = load_dataset('json', data_files=TRAINING_CORPUS, split='train')
print(f"loaded {len(dataset)} raw documents")

# -> tokenize each document individually, no special tokens (we add BOS/EOS manually in pack_fn)
def tok_fn(ex):
    return tokenizer(
        ex["text"],
        add_special_tokens=False,
        truncation=False,
        return_attention_mask=False,
        return_token_type_ids=False,
    )

# -> pack tokenized documents into fixed-size blocks of CONTEXT_LENGTH
# -> each document is wrapped with BOS/EOS so model learns document boundaries
# -> tokens stream continuously across documents — no padding, no wasted compute
def pack_fn(ex):
    ids = []
    bos_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id
    for seq in ex["input_ids"]:
        # -> wrap each document with <s> ... </s> boundary markers
        ids.append(bos_id)
        ids.extend(seq)
        ids.append(eos_id)
    # -> trim to exact multiple of CONTEXT_LENGTH (discard leftover partial block)
    total = (len(ids) // CONTEXT_LENGTH) * CONTEXT_LENGTH
    if total == 0:
        return {"input_ids": [], "attention_mask": [], "labels": []}
    ids = ids[:total]
    chunks = [ids[i:i + CONTEXT_LENGTH] for i in range(0, total, CONTEXT_LENGTH)]
    return {
        "input_ids": chunks,
        "attention_mask": [[1] * CONTEXT_LENGTH for _ in chunks],
        "labels": [c.copy() for c in chunks],  # -> causal LM: labels = input_ids
    }

print("tokenizing corpus...")
tok_ds = dataset.map(
    tok_fn,
    batched=True,
    remove_columns=dataset.column_names,
    desc="tokenizing",
)

# -> remove any extra columns before packing (attention_mask etc from tokenizer)
cols_to_keep = ["input_ids"]
extra_cols = [c for c in tok_ds.column_names if c not in cols_to_keep]
if extra_cols:
    tok_ds = tok_ds.remove_columns(extra_cols)

print(f"packing into {CONTEXT_LENGTH}-token blocks...")
train_ds = tok_ds.map(
    pack_fn,
    batched=True,
    remove_columns=tok_ds.column_names,
    desc=f"packing to {CONTEXT_LENGTH}",
)

print(f"packed dataset: {len(train_ds)} sequences of {CONTEXT_LENGTH} tokens")
print(f"total tokens: ~{len(train_ds) * CONTEXT_LENGTH / 1e9:.2f}B")

# -> split into train/eval
train_ds = train_ds.train_test_split(test_size=2000, seed=42)
print(f"train: {len(train_ds['train'])} sequences | eval: {len(train_ds['test'])} sequences")

# -> simple collator — data is already packed with labels, just stack into tensors
# -> DataCollatorForLanguageModeling is NOT used here because it may corrupt packed sequences
class SimpleCollator:
    def __call__(self, batch):
        input_ids = torch.tensor([item['input_ids'] for item in batch], dtype=torch.long)
        labels = torch.tensor([item['labels'] for item in batch], dtype=torch.long)
        attention_mask = torch.tensor([item['attention_mask'] for item in batch], dtype=torch.long)
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
        }

data_collator = SimpleCollator()

# -> training arguments
print("initializing training arguments...")
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=12,       # -> reduced from 12 because sequences are now 4096 tokens (2x longer)
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=2,       # -> effective batch = 24 sequences of 4k tokens
    learning_rate=2e-4,
    lr_scheduler_type='cosine',
    warmup_steps=500,
    weight_decay=0.01,
    num_train_epochs=1,
    logging_steps=1000,
    eval_strategy='steps',
    eval_steps=2000,
    save_strategy='steps',
    save_steps=1000,
    save_total_limit=2,
    bf16=True,                           # -> switched from fp16 to bf16 — more stable for LLM training
    fp16=False,
    gradient_checkpointing=True,
    report_to='tensorboard',
    eval_accumulation_steps=1,
)

# -> train each architecture sequentially
for model_class, model_config, model_name in arch_selectors:
    print(f"\ntraining {model_name}...")
    model = model_class(model_config)
    print(f"model size: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params")
    print(f"model vocab size: {model_config.vocab_size}")  # -> must match tokenizer vocab size

    model_args = copy.deepcopy(training_args)
    model_args.output_dir = f"{OUTPUT_DIR}/{model_name}"

    trainer = Trainer(
        model=model,
        args=model_args,
        train_dataset=train_ds['train'],
        eval_dataset=train_ds['test'],
        data_collator=data_collator
    )

    trainer.train()
    trainer.save_model(f"{OUTPUT_DIR}/{model_name}/final")
    del model, trainer
    torch.cuda.empty_cache()
    print(f"saved {model_name} to {OUTPUT_DIR}/{model_name}/final")

