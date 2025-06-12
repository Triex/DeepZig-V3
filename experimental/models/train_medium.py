# # Create a venv with HF tooling
# python -m venv .venv && source .venv/bin/activate
# pip install torch transformers datasets safetensors accelerate tokenizers

# python train_medium.py        # (script below)
# zig build run -- --medium --model models/deepzig-medium-demo --no-server

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

# 2. Define a medium causal-LM that matches the config dimensions and Zig engine expectations
class DeepZigMediumLM(PreTrainedModel):
    config_class = PretrainedConfig
    def __init__(self, cfg):
        super().__init__(cfg)
        self.embed = nn.Embedding(cfg.vocab_size, cfg.hidden_size)

        # Create transformer layers with attention and MLP blocks
        self.layers = nn.ModuleList([])
        for i in range(cfg.num_hidden_layers):
            # Create a layer with attention and MLP
            layer = nn.ModuleDict({
                # Self-attention block
                'attention': nn.ModuleDict({
                    'q_proj': nn.Linear(cfg.hidden_size, cfg.num_attention_heads * cfg.qk_rope_head_dim),
                    'k_proj': nn.Linear(cfg.hidden_size, cfg.num_key_value_heads * cfg.qk_rope_head_dim),
                    'v_proj': nn.Linear(cfg.hidden_size, cfg.num_key_value_heads * cfg.v_head_dim),
                    'o_proj': nn.Linear(cfg.num_attention_heads * cfg.v_head_dim, cfg.hidden_size),
                    'norm': nn.LayerNorm(cfg.hidden_size, eps=cfg.rms_norm_eps),
                }),

                # MLP block (with MoE for some layers)
                'mlp': nn.ModuleDict({
                    'norm': nn.LayerNorm(cfg.hidden_size, eps=cfg.rms_norm_eps),
                })
            })

            # Add MLP components based on whether this is an MoE layer
            if i >= cfg.first_k_dense_replace and i % cfg.moe_layer_freq == 0:
                # MoE layer
                layer['mlp']['experts'] = nn.ModuleList([
                    nn.ModuleDict({
                        'gate_proj': nn.Linear(cfg.hidden_size, cfg.moe_intermediate_size),
                        'down_proj': nn.Linear(cfg.moe_intermediate_size, cfg.hidden_size)
                    }) for _ in range(cfg.num_experts)
                ])
                layer['mlp']['router'] = nn.Linear(cfg.hidden_size, cfg.num_experts)
            else:
                # Standard MLP
                layer['mlp']['gate_proj'] = nn.Linear(cfg.hidden_size, cfg.intermediate_size)
                layer['mlp']['down_proj'] = nn.Linear(cfg.intermediate_size, cfg.hidden_size)

            self.layers.append(layer)

        # Final layer norm
        self.norm = nn.LayerNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)

        # LM head
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

        # Initialize weights
        self.apply(self._init_weights)

        # Tie weights
        self.lm_head.weight = self.embed.weight

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, input_ids, labels=None, attention_mask=None, **kwargs):
        # Get embeddings
        hidden_states = self.embed(input_ids)
        batch_size, seq_length = input_ids.shape

        # Create causal mask for self-attention
        causal_mask = torch.triu(torch.ones((seq_length, seq_length), dtype=torch.bool, device=input_ids.device), diagonal=1)
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_length, seq_length]

        # Process through transformer layers
        for layer in self.layers:
            # Self-attention block with residual connection
            residual = hidden_states
            hidden_states = layer['attention']['norm'](hidden_states)

            # Compute QKV projections
            q = layer['attention']['q_proj'](hidden_states)
            k = layer['attention']['k_proj'](hidden_states)
            v = layer['attention']['v_proj'](hidden_states)

            # Reshape for attention computation
            head_dim_q = self.config.qk_rope_head_dim
            head_dim_v = self.config.v_head_dim
            num_heads = self.config.num_attention_heads
            num_kv_heads = self.config.num_key_value_heads

            q = q.view(batch_size, seq_length, num_heads, head_dim_q).transpose(1, 2)  # [batch, heads, seq, head_dim]
            k = k.view(batch_size, seq_length, num_kv_heads, head_dim_q).transpose(1, 2)  # [batch, heads, seq, head_dim]
            v = v.view(batch_size, seq_length, num_kv_heads, head_dim_v).transpose(1, 2)  # [batch, heads, seq, head_dim]

            # Compute attention scores and mask
            attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (head_dim_q ** 0.5)
            attn_weights = attn_weights.masked_fill(causal_mask, float('-inf'))

            # Apply attention mask if provided
            if attention_mask is not None:
                # Expand attention_mask to match attn_weights dimensions
                expanded_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, seq]
                expanded_mask = (1.0 - expanded_mask) * -10000.0  # Convert 0s to large negative values
                attn_weights = attn_weights + expanded_mask

            # Compute attention probabilities and weighted sum
            attn_weights = F.softmax(attn_weights, dim=-1)
            attn_output = torch.matmul(attn_weights, v)  # [batch, heads, seq, head_dim]

            # Reshape and project back
            attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_length, -1)
            hidden_states = residual + layer['attention']['o_proj'](attn_output)

            # MLP block with residual connection
            residual = hidden_states
            hidden_states = layer['mlp']['norm'](hidden_states)

            # Process through MLP (either standard or MoE)
            if 'experts' in layer['mlp']:
                # MoE layer
                router_logits = layer['mlp']['router'](hidden_states)  # [batch, seq, num_experts]
                routing_weights = F.softmax(router_logits, dim=-1)

                # Get top-2 experts
                top_k = min(2, len(layer['mlp']['experts']))  # Use top-2 experts
                _, indices = torch.topk(routing_weights, top_k, dim=-1)  # [batch, seq, top_k]

                # Process through selected experts
                expert_outputs = torch.zeros_like(hidden_states)
                for b in range(batch_size):
                    for s in range(seq_length):
                        for k in range(top_k):
                            expert_idx = indices[b, s, k].item()
                            expert = layer['mlp']['experts'][expert_idx]
                            weight = routing_weights[b, s, expert_idx]

                            # Apply expert
                            expert_input = hidden_states[b, s].unsqueeze(0)  # [1, hidden]
                            gate_output = F.gelu(expert['gate_proj'](expert_input))
                            expert_output = expert['down_proj'](gate_output)
                            expert_outputs[b, s] += weight * expert_output.squeeze(0)

                hidden_states = residual + expert_outputs
            else:
                # Standard MLP
                gate_output = F.gelu(layer['mlp']['gate_proj'](hidden_states))
                mlp_output = layer['mlp']['down_proj'](gate_output)
                hidden_states = residual + mlp_output

        # Apply final layer norm
        hidden_states = self.norm(hidden_states)

        # Get logits
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return CausalLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=hidden_states
        )

model = DeepZigMediumLM(config)

# 3. Load and prepare the dataset
print("Loading dataset...")
ds_full = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

# Use a larger subset of the dataset for better training
ds = ds_full.select(range(min(10000, len(ds_full))))
print(f"Using {len(ds)} samples from {len(ds_full)} total")

# Filter out very short texts
ds = ds.filter(lambda example: len(example["text"]) > 100)
print(f"After filtering: {len(ds)} samples")

# Create and train tokenizer first so we can use it for tokenization
print("Creating and training tokenizer...")
def create_tokenizer():
    # Create a basic BPE tokenizer
    tokenizer = Tokenizer(models.BPE())

    # Add pre-tokenization (split on whitespace and punctuation)
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.WhitespaceSplit(),
        pre_tokenizers.Punctuation()
    ])

    # Train on a sample of text
    trainer = trainers.BpeTrainer(
        vocab_size=cfg_dict["vocab_size"],
        special_tokens=["<unk>", "<s>", "</s>", "<pad>", "<mask>"]
    )

    # Use a larger sample of the dataset for tokenizer training
    files = ["tokenizer_training_data.txt"]
    with open(files[0], "w") as f:
        for i in range(min(20000, len(ds_full))):
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

# Create tokenizer
tokenizer = create_tokenizer()
out_dir = "deepzig-medium-demo"
os.makedirs(out_dir, exist_ok=True)
tokenizer.save(f"{out_dir}/tokenizer.json")
print("✅ Saved tokenizer to", f"{out_dir}/tokenizer.json")

# Now tokenize the dataset with our trained tokenizer
def tokenize_function(examples):
    max_length = 256  # Increased sequence length for better context
    results = {"input_ids": [], "labels": [], "attention_mask": []}

    for text in examples["text"]:
        # Skip empty texts
        if not text.strip():
            continue

        # Tokenize the text
        encoded = tokenizer.encode(text)
        tokens = encoded.ids

        # Create attention mask (1 for real tokens)
        attention_mask = [1] * len(tokens)

        # Truncate if too long
        if len(tokens) > max_length:
            tokens = tokens[:max_length]
            attention_mask = attention_mask[:max_length]

        # Pad if too short
        if len(tokens) < max_length:
            padding_length = max_length - len(tokens)
            tokens = tokens + [0] * padding_length
            attention_mask = attention_mask + [0] * padding_length

        # Make sure all sequences are exactly max_length
        assert len(tokens) == max_length, f"Token length {len(tokens)} != {max_length}"
        assert len(attention_mask) == max_length, f"Mask length {len(attention_mask)} != {max_length}"

        # Store the tokenized text
        results["input_ids"].append(tokens)
        results["labels"].append(tokens.copy())
        results["attention_mask"].append(attention_mask)

    return results

print("Tokenizing dataset...")
ds = ds.map(
    tokenize_function,
    batched=True,
    remove_columns=ds.column_names,
    desc="Tokenizing dataset"
)

# Define a custom data collator to handle padding
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

@dataclass
class CustomDataCollator:
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # First, pad all sequences to the same length
        max_length = 128

        # Initialize batch tensors
        batch = {}
        batch["input_ids"] = torch.zeros((len(features), max_length), dtype=torch.long)
        batch["labels"] = torch.zeros((len(features), max_length), dtype=torch.long)
        batch["attention_mask"] = torch.zeros((len(features), max_length), dtype=torch.long)

        # Fill batch tensors
        for i, feature in enumerate(features):
            input_ids = feature["input_ids"]
            labels = feature["labels"]
            attention_mask = feature.get("attention_mask", [1] * len(input_ids))

            # Ensure all are the same length by padding
            seq_length = min(len(input_ids), max_length)

            # Copy data to batch tensors
            batch["input_ids"][i, :seq_length] = torch.tensor(input_ids[:seq_length])
            batch["labels"][i, :seq_length] = torch.tensor(labels[:seq_length])
            batch["attention_mask"][i, :seq_length] = torch.tensor(attention_mask[:seq_length])

        return batch

args = TrainingArguments(
    "deepzig-medium-demo",  # Output directly to the current directory
    num_train_epochs          = 3,  # Fewer epochs for faster training
    per_device_train_batch_size= 8,  # Smaller batch size to avoid memory issues
    learning_rate             = 5e-5,  # Slightly higher learning rate for faster convergence
    logging_steps             = 1,  # Log every step for better feedback
    save_steps                = 50,  # Save more frequently
    save_total_limit          = 2,  # Keep the last 2 checkpoints
    gradient_accumulation_steps= 2,  # Smaller accumulation for faster updates
    fp16                      = False,  # Disable mixed precision to avoid potential issues
    report_to                 = "none",  # Don't report to wandb etc.
    warmup_steps              = 10,  # Fewer warmup steps
    weight_decay              = 0.01,  # Add weight decay for regularization
    disable_tqdm              = False,  # Ensure progress bars are shown
    logging_first_step        = True,  # Log the first step for immediate feedback
    save_safetensors          = False  # Disable safetensors to allow weight sharing
)
# Create trainer with custom data collator
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=ds,
    data_collator=CustomDataCollator()
)

# 5. Train the model
print("\nTraining the model...")
from transformers.trainer_callback import TrainerCallback

# Add progress callback for better visibility during training
class ProgressCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_world_process_zero and logs:
            print(f"Progress: {100 * state.global_step / state.max_steps:.1f}% complete")

# Fix for weight sharing error by disabling safe serialization
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=ds,
    data_collator=CustomDataCollator(),
    # Use existing optimizers if defined, otherwise let Trainer create them
    optimizers=(optimizer, lr_scheduler) if 'optimizer' in locals() else (None, None),
    callbacks=[ProgressCallback()]
)
trainer.train()

# Save the model
trainer.save_model(output_dir="deepzig-medium-demo")
print("✅ Model training complete")

# 6. Save model weights in Zig-compatible safetensors format
print("\nSaving model weights in Zig-compatible format...")

# Create output directory if it doesn't exist
os.makedirs("deepzig-medium-demo", exist_ok=True)

# Create a state dict with the model weights mapped to Zig-compatible names
state_dict = {}

# Add model weights to state dict with proper naming for Zig compatibility
for name, param in model.named_parameters():
    # Map Python model parameter names to Zig-compatible names
    if name == "embed.weight":
        state_dict["model.embed_tokens.weight"] = param.detach().cpu().numpy()
    elif name == "lm_head.weight":
        state_dict["lm_head.weight"] = param.detach().cpu().numpy()
    elif name == "norm.weight":
        state_dict["model.norm.weight"] = param.detach().cpu().numpy()
    elif name.startswith("layers."):
        # Handle transformer layers
        parts = name.split(".")
        layer_num = parts[1]

        if "attention" in name:
            # Handle attention components
            if "q_proj" in name:
                state_dict[f"model.layers.{layer_num}.self_attn.q_proj.weight"] = param.detach().cpu().numpy()
            elif "k_proj" in name:
                state_dict[f"model.layers.{layer_num}.self_attn.k_proj.weight"] = param.detach().cpu().numpy()
            elif "v_proj" in name:
                state_dict[f"model.layers.{layer_num}.self_attn.v_proj.weight"] = param.detach().cpu().numpy()
            elif "o_proj" in name:
                state_dict[f"model.layers.{layer_num}.self_attn.o_proj.weight"] = param.detach().cpu().numpy()
            elif "norm" in name:
                state_dict[f"model.layers.{layer_num}.input_layernorm.weight"] = param.detach().cpu().numpy()
        elif "mlp" in name:
            # Handle MLP components
            if "gate_proj" in name:
                state_dict[f"model.layers.{layer_num}.mlp.gate_proj.weight"] = param.detach().cpu().numpy()
            elif "down_proj" in name:
                state_dict[f"model.layers.{layer_num}.mlp.down_proj.weight"] = param.detach().cpu().numpy()
            elif "norm" in name:
                state_dict[f"model.layers.{layer_num}.post_attention_layernorm.weight"] = param.detach().cpu().numpy()
        else:
            # Use a generic mapping for other parameters
            state_dict[f"model.{name}"] = param.detach().cpu().numpy()
    else:
        # Use a generic mapping for other parameters
        state_dict[f"model.{name}"] = param.detach().cpu().numpy()

# Save to safetensors format
from safetensors.numpy import save_file
save_file(state_dict, "deepzig-medium-demo/model.safetensors")

# Save config
with open("deepzig-medium-demo/config.json", "w") as f:
    json.dump(cfg_dict, f, indent=2)

print("✅ Saved model, config, and tokenizer to deepzig-medium-demo")

# 7. Test the model with the tokenizer to ensure it generates text
print("\nTesting text generation with the trained model:")

# Move model to CPU for inference to avoid MPS device issues
model.to("cpu")
model.eval()

# Create an improved test function with enhanced sampling for better text generation
def generate_text(prompt, max_length=100, temperature=0.8, top_k=40, top_p=0.9, repetition_penalty=1.2):
    try:
        # Tokenize the prompt with proper BOS token
        if not prompt.startswith("<s>"):
            prompt = "<s> " + prompt

        encoded = tokenizer.encode(prompt)
        input_ids = torch.tensor([encoded.ids], device="cpu")
        prompt_length = len(encoded.ids)

        # Track generated tokens for repetition penalty
        generated_tokens = set(input_ids[0].tolist())

        # Generate text
        with torch.no_grad():
            for _ in range(max_length):
                # Get model output for current sequence
                outputs = model(input_ids)
                next_token_logits = outputs.logits[:, -1, :]

                # Apply repetition penalty to discourage repeating tokens
                for token_id in generated_tokens:
                    next_token_logits[0, token_id] /= repetition_penalty

                # Apply temperature scaling
                next_token_logits = next_token_logits / temperature

                # Apply top-k filtering
                if top_k > 0:
                    top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                    next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                    next_token_logits.scatter_(1, top_k_indices, top_k_logits)

                # Apply top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # Shift the indices to the right to keep also the first token above the threshold
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0

                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    next_token_logits[0, indices_to_remove] = float('-inf')

                # Sample from the filtered distribution
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).squeeze(1)

                # Add the new token to the sequence
                input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
                generated_tokens.add(next_token.item())

                # Stop if we predict EOS, </s>, or after generating several newlines
                if (next_token.item() == tokenizer.token_to_id("</s>") or
                    next_token.item() == tokenizer.token_to_id("\n") or
                    next_token.item() == tokenizer.token_to_id("<eos>") or
                    (len(input_ids[0]) > prompt_length + 10 and
                     input_ids[0][-3:].tolist().count(tokenizer.token_to_id("\n")) >= 2)):
                    break

        # Decode the generated text
        generated_ids = input_ids[0].tolist()

        # Get just the generated part (not the prompt)
        generated_part = generated_ids[prompt_length:]

        # Clean up the text
        full_text = tokenizer.decode(generated_ids)
        generated_text = tokenizer.decode(generated_part)

        # Clean up special tokens
        generated_text = generated_text.replace("<s>", "").replace("</s>", "").replace("<pad>", "")

        return f"{prompt.replace('<s> ', '')}{generated_text}"
    except Exception as e:
        print(f"Error during generation: {e}")
        return f"Error: {str(e)}"

# Test with a variety of prompts using different generation parameters
test_prompts = [
    "The quick brown fox jumps over",
    "In the beginning there was",
    "Once upon a time in a land far away",
    "Zig is a programming language designed for",
    "The history of artificial intelligence begins with",
    "The most important feature of DeepZig is",
    "To generate high-quality text, language models need"
]

print("\nGenerating with balanced parameters (temperature=0.7, top_k=50, top_p=0.92, repetition_penalty=1.2):")
for prompt in test_prompts:
    print(f"\nPrompt: {prompt}")
    generated = generate_text(prompt, max_length=150, temperature=0.7, top_k=50, top_p=0.92, repetition_penalty=1.2)
    print(f"Generated: {generated}")

print("\nGenerating with creative parameters (temperature=1.0, top_k=100, top_p=0.95, repetition_penalty=1.3):")
for prompt in test_prompts[:3]:  # Try a subset with different parameters
    print(f"\nPrompt: {prompt}")
    generated = generate_text(prompt, max_length=150, temperature=1.0, top_k=100, top_p=0.95, repetition_penalty=1.3)
    print(f"Generated: {generated}")

print("\nGenerating with focused parameters (temperature=0.5, top_k=20, top_p=0.85, repetition_penalty=1.5):")
for prompt in test_prompts[3:5]:  # Try technical prompts with more focused parameters
    print(f"\nPrompt: {prompt}")
    generated = generate_text(prompt, max_length=200, temperature=0.5, top_k=20, top_p=0.85, repetition_penalty=1.5)
    print(f"Generated: {generated}")

# 7. Save model weights in Zig-compatible safetensors format
print("\nSaving model weights in Zig-compatible format...")

# Create output directory if it doesn't exist
os.makedirs("deepzig-medium-demo", exist_ok=True)

# Create a state dict with the model weights mapped to Zig-compatible names
state_dict = {}

# Add model weights to state dict with proper naming for Zig compatibility
for name, param in model.named_parameters():
    # Map Python model parameter names to Zig-compatible names
    if name == "embed.weight":
        state_dict["model.embed_tokens.weight"] = param.detach().cpu().numpy()
    elif name == "lm_head.weight":
        state_dict["lm_head.weight"] = param.detach().cpu().numpy()
    elif name == "norm.weight":
        state_dict["model.norm.weight"] = param.detach().cpu().numpy()
    elif name.startswith("layers."):
        # Handle transformer layers
        layer_num = name.split(".")[1]
        if name.endswith(".0.weight") or name.endswith(".0.bias"):
            # First linear layer in MLP
            suffix = name.split(".")[-1]
            state_dict[f"model.layers.{layer_num}.mlp.gate_proj.{suffix}"] = param.detach().cpu().numpy()
        elif name.endswith(".2.weight") or name.endswith(".2.bias"):
            # Second linear layer in MLP
            suffix = name.split(".")[-1]
            state_dict[f"model.layers.{layer_num}.mlp.down_proj.{suffix}"] = param.detach().cpu().numpy()
        else:
            # Use a generic mapping for other parameters
            state_dict[f"model.{name}"] = param.detach().cpu().numpy()
    else:
        # Use a generic mapping for other parameters
        state_dict[f"model.{name}"] = param.detach().cpu().numpy()

# Save to safetensors format
from safetensors.numpy import save_file
save_file(state_dict, "deepzig-medium-demo/model.safetensors")

# Save config
with open("deepzig-medium-demo/config.json", "w") as f:
    json.dump(cfg_dict, f, indent=2)

# Save tokenizer files
tokenizer.save("deepzig-medium-demo/tokenizer.json")

print("✅ Saved model, config, and tokenizer to deepzig-medium-demo")
