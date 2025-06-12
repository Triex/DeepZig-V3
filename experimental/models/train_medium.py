# # Create a venv with HF tooling
# python -m venv .venv && source .venv/bin/activate
# pip install torch transformers datasets safetensors accelerate tokenizers

# python train_medium.py        # (script below)

from transformers import PretrainedConfig, PreTrainedModel, TrainingArguments, Trainer
from transformers.modeling_outputs import CausalLMOutput
import torch, torch.nn as nn, torch.nn.functional as F
from datasets import load_dataset
from tokenizers import Tokenizer, models, pre_tokenizers, trainers, processors
import json, os

# 1. Build a generic HF Config that will happily accept DeepZig-specific keys
cfg_dict = dict(
    model_type         = "deepzig_v3",
    architectures      = ["DeepZigMediumLM"],
    vocab_size         = 32_000,
    hidden_size        = 768,
    intermediate_size  = 2_048,
    num_hidden_layers  = 12,
    num_key_value_heads= 12,
    num_attention_heads= 12,
    max_position_embeddings = 2_048,
    qk_nope_head_dim   = 32,
    qk_rope_head_dim   = 32,
    v_head_dim         = 64,
    qk_rope_base       = 10_000.0,
    num_experts        = 8,
    num_experts_per_token = 2,
    moe_layer_freq     = 2,
    first_k_dense_replace = 1,
    moe_intermediate_size = 2_048,
    bos_token_id       = 1,
    eos_token_id       = 2,
    rms_norm_eps       = 1e-6,
    temperature        = 0.8,
    top_p              = 0.9,
    top_k              = 40,
    torch_dtype        = "float32"
)
config = PretrainedConfig.from_dict(cfg_dict)

# 2. Define a medium causal-LM that matches the config dimensions
class DeepZigMediumLM(PreTrainedModel):
    config_class = PretrainedConfig
    def __init__(self, cfg):
        super().__init__(cfg)
        self.embed = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        
        # Create a simple transformer with the correct number of layers
        # This is a placeholder that will generate weights with the correct dimensions
        # for the Zig model to load
        self.layers = nn.ModuleList([
            nn.Linear(cfg.hidden_size, cfg.hidden_size) 
            for _ in range(cfg.num_hidden_layers)
        ])
        
        self.norm = nn.LayerNorm(cfg.hidden_size)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        self.init_weights()

    def forward(self, input_ids, labels=None):
        x = self.embed(input_ids)
        
        # Pass through each layer
        for layer in self.layers:
            x = layer(x) + x  # Simple residual connection
            
        x = self.norm(x)
        logits = self.lm_head(x)
        
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1)
            )
        # Return a dictionary-like object (CausalLMOutput) as required by Trainer
        return CausalLMOutput(loss=loss, logits=logits)

model = DeepZigMediumLM(config)

# 3. Toy dataset: random tokens (1-2 minutes on CPU)
# ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:1%]")
ds_full = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
ds = ds_full.shuffle(seed=42).select(range(int(0.005 * len(ds_full))))
def rnd(ex):
    ex["input_ids"] = torch.randint(0, config.vocab_size, (128,))
    ex["labels"]    = ex["input_ids"].clone()
    return ex
ds = ds.map(rnd, remove_columns=ds.column_names)

args = TrainingArguments(
    "deepzig-medium-demo",  # Output directly to the current directory
    num_train_epochs          = 1,
    per_device_train_batch_size= 32,
    logging_steps             = 10,
    save_total_limit          = 1,
)
Trainer(model, args, train_dataset=ds).train()

# 4. Create and save a basic BPE tokenizer
def create_tokenizer():
    # Create a basic BPE tokenizer
    tokenizer = Tokenizer(models.BPE())
    
    # Add pre-tokenization (split on whitespace and punctuation)
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.WhitespaceSplit(),
        pre_tokenizers.Punctuation()
    ])
    
    # Train on a small sample of text
    trainer = trainers.BpeTrainer(
        vocab_size=cfg_dict["vocab_size"],
        special_tokens=["<unk>", "<s>", "</s>", "<pad>", "<mask>"]
    )
    
    # Use a small sample of the dataset for tokenizer training
    files = ["tokenizer_training_data.txt"]
    with open(files[0], "w") as f:
        for i in range(min(1000, len(ds_full))):
            f.write(ds_full[i]["text"] + "\n")
    
    tokenizer.train(files, trainer)
    
    # Add post-processing
    tokenizer.post_processor = processors.TemplateProcessing(
        single="<s> $A </s>",
        pair="<s> $A </s> $B </s>",
        special_tokens=[
            ("<s>", 1),
            ("</s>", 2),
        ],
    )
    
    return tokenizer

# 5. Save in HF-style layout (Zig side just needs the files)
out_dir = args.output_dir
model.save_pretrained(out_dir, safe_serialization=True)      # model.safetensors
json.dump(cfg_dict, open(f"{out_dir}/config.json","w"))

# Create and save tokenizer
tokenizer = create_tokenizer()
tokenizer.save(f"{out_dir}/tokenizer.json")

print("✅ Saved model, config, and tokenizer to", out_dir)