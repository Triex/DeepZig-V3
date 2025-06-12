#!/usr/bin/env python3
"""
DeepZig Conversational Model Training Script
===========================================

A reference implementation for training small, efficient conversational language models
following 2024-2025 best practices. This script creates a DeepZig model optimized for:

- Fast inference on edge devices
- Tool calling and system prompt understanding
- Efficient memory usage with optional LoRA fine-tuning
- High-quality conversational capabilities

Key Features:
- RoPE (Rotary Position Embeddings) for better positional understanding
- RMS normalization for training stability
- Optimized small model architecture (512 hidden, 8 layers)
- High-quality conversational datasets with tool-calling examples
- Optional LoRA fine-tuning for parameter efficiency
- Proper generation configuration to eliminate warnings
- Zig-compatible safetensors export

Usage:
    python train_medium.py [--use-lora] [--eval-split 0.1] [--max-samples 20000]

Requirements:
    pip install torch transformers datasets safetensors accelerate tokenizers peft
"""

import argparse
import json
import os
import warnings
from typing import Dict, List, Optional, Union, Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.checkpoint import checkpoint

from transformers import (
    PretrainedConfig, PreTrainedModel, TrainingArguments, Trainer,
    GenerationConfig, TrainerCallback
)
from transformers.modeling_outputs import CausalLMOutput
from datasets import load_dataset, concatenate_datasets, Dataset
from tokenizers import Tokenizer, models, pre_tokenizers, trainers, processors
from safetensors.numpy import save_file

# Optional LoRA support
try:
    from peft import LoraConfig, get_peft_model, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    warnings.warn("PEFT not available. Install with: pip install peft")


# =============================================================================
# Model Configuration
# =============================================================================

class DeepZigConfig(PretrainedConfig):
    """Optimized configuration for small conversational language models."""

    model_type = "deepzig_conversational"

    def __init__(
        self,
        vocab_size: int = 32_000,
        hidden_size: int = 512,          # Optimized for efficiency
        intermediate_size: int = 1536,   # 3x hidden_size ratio
        num_hidden_layers: int = 8,      # Reduced for faster inference
        num_attention_heads: int = 8,
        num_key_value_heads: int = 8,    # Full attention for better quality
        max_position_embeddings: int = 4096,  # Support longer conversations
        rope_base: float = 10_000.0,          # RoPE base frequency
        rope_scaling: Optional[Dict] = None,  # For sequence length extension
        rms_norm_eps: float = 1e-6,
        attention_dropout: float = 0.0,
        hidden_dropout: float = 0.0,
        torch_dtype: str = "bfloat16",
        use_cache: bool = True,
        tie_word_embeddings: bool = True,     # Explicitly declare tied weights
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        **kwargs
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs
        )

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.max_position_embeddings = max_position_embeddings
        self.rope_base = rope_base
        self.rope_scaling = rope_scaling
        self.rms_norm_eps = rms_norm_eps
        self.attention_dropout = attention_dropout
        self.hidden_dropout = hidden_dropout
        self.torch_dtype = torch_dtype
        self.use_cache = use_cache
        self.tie_word_embeddings = tie_word_embeddings


# =============================================================================
# RoPE (Rotary Position Embedding) Implementation
# =============================================================================

class RotaryEmbedding(nn.Module):
    """
    Efficient Rotary Position Embedding implementation.

    RoPE provides better positional understanding than absolute position embeddings,
    crucial for conversational models that need to understand context relationships.
    """

    def __init__(self, dim: int, max_position_embeddings: int = 4096, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        # Precompute frequency tensor
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate cos and sin embeddings for the given sequence length."""
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(x.dtype), emb.sin().to(x.dtype)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embeddings to query and key tensors."""
    def rotate_half(x):
        x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# =============================================================================
# RMS Normalization
# =============================================================================

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

    RMSNorm is more stable and efficient than LayerNorm for language models,
    providing better training dynamics and faster inference.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states


# =============================================================================
# Attention Module with RoPE
# =============================================================================

class DeepZigAttention(nn.Module):
    """
    Multi-head attention with RoPE and optimizations for conversational models.
    """

    def __init__(self, config: DeepZigConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        if self.hidden_size % self.num_heads != 0:
            raise ValueError(f"hidden_size {self.hidden_size} not divisible by num_heads {self.num_heads}")

        # Linear projections
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        # RoPE
        self.rotary_emb = RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_base
        )

        # Dropout
        self.attention_dropout = nn.Dropout(config.attention_dropout)

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_length, _ = hidden_states.shape

        # Linear projections
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape for multi-head attention
        query_states = query_states.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_length, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_length, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        cos, sin = self.rotary_emb(value_states, seq_len=seq_length)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Expand key/value states for grouped query attention
        if self.num_key_value_groups > 1:
            key_states = key_states.unsqueeze(2).expand(-1, -1, self.num_key_value_groups, -1, -1).reshape(
                batch_size, self.num_heads, seq_length, self.head_dim
            )
            value_states = value_states.unsqueeze(2).expand(-1, -1, self.num_key_value_groups, -1, -1).reshape(
                batch_size, self.num_heads, seq_length, self.head_dim
            )

        # Scaled dot-product attention
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        # Apply causal mask
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # Causal mask for autoregressive generation
        causal_mask = torch.triu(torch.ones((seq_length, seq_length), dtype=torch.bool, device=hidden_states.device), diagonal=1)
        attn_weights = attn_weights.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = self.attention_dropout(attn_weights)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, value_states)

        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_length, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        return attn_output


# =============================================================================
# Feed-Forward Network
# =============================================================================

class DeepZigMLP(nn.Module):
    """
    Optimized feed-forward network with SwiGLU activation.

    SwiGLU provides better performance than standard ReLU/GELU for language models.
    """

    def __init__(self, config: DeepZigConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.dropout = nn.Dropout(config.hidden_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.gate_proj(x))  # SwiGLU activation
        up = self.up_proj(x)
        return self.dropout(self.down_proj(gate * up))


# =============================================================================
# Transformer Layer
# =============================================================================

class DeepZigLayer(nn.Module):
    """Single transformer layer with attention and feed-forward components."""

    def __init__(self, config: DeepZigConfig):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = DeepZigAttention(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = DeepZigMLP(config)

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention with residual connection
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, attention_mask)
        hidden_states = residual + hidden_states

        # Feed-forward with residual connection
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


# =============================================================================
# Main Model
# =============================================================================

class DeepZigConversationalModel(PreTrainedModel):
    """
    DeepZig Conversational Language Model

    A small, efficient transformer model optimized for conversational AI tasks,
    tool calling, and system prompt understanding.
    """

    config_class = DeepZigConfig

    def __init__(self, config: DeepZigConfig):
        super().__init__(config)
        self.config = config
        self.gradient_checkpointing = False

        # Embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        # Transformer layers
        self.layers = nn.ModuleList([DeepZigLayer(config) for _ in range(config.num_hidden_layers)])

        # Final normalization
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Language modeling head
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights first
        self.apply(self._init_weights)

        # Tie weights between embedding and lm_head for parameter efficiency
        self.tie_weights()

    def _init_weights(self, module):
        """Initialize weights following best practices for small language models."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def tie_weights(self):
        """Tie the weights between the input embeddings and the output embeddings."""
        self.lm_head.weight = self.embed_tokens.weight

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        **kwargs
    ):
        """Prepare inputs for generation (required by PEFT/LoRA)."""
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]

        # If attention_mask is provided, adjust it
        if attention_mask is not None and past_key_values is not None:
            attention_mask = attention_mask[:, -(input_ids.shape[-1] + past_key_values[0][0].shape[-2]):]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache", True),
        }

    def _set_gradient_checkpointing(self, module, value=False):
        """Enable or disable gradient checkpointing for the model."""
        if isinstance(module, DeepZigConversationalModel):
            module.gradient_checkpointing = value
        # Also set it for transformer layers
        for layer in self.layers:
            if hasattr(layer, 'gradient_checkpointing'):
                layer.gradient_checkpointing = value

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """Enable gradient checkpointing for the model."""
        if not hasattr(self, 'gradient_checkpointing'):
            self.gradient_checkpointing = True
        else:
            self.gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing for the model."""
        if hasattr(self, 'gradient_checkpointing'):
            self.gradient_checkpointing = False

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs
    ) -> CausalLMOutput:

        # Get embeddings
        hidden_states = self.embed_tokens(input_ids)

        # Create attention mask if not provided
        if attention_mask is not None:
            # Convert attention mask to bias format
            batch_size, seq_length = input_ids.shape
            attention_mask = (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)) * -10000.0

        # Pass through transformer layers
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                # Use gradient checkpointing for memory efficiency during training
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)
                    return custom_forward

                hidden_states = checkpoint(
                    create_custom_forward(layer),
                    hidden_states,
                    attention_mask,
                    use_reentrant=False
                )
            else:
                hidden_states = layer(hidden_states, attention_mask)

        # Final normalization
        hidden_states = self.norm(hidden_states)

        # Language modeling logits
        logits = self.lm_head(hidden_states)

        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            # Shift labels for causal language modeling
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # Calculate cross-entropy loss
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return CausalLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=hidden_states,
        )


# =============================================================================
# Data Processing
# =============================================================================

class ConversationalDataProcessor:
    """Handles loading and processing of conversational datasets with tool calling support."""

    SPECIAL_TOKENS = [
        "<unk>", "<s>", "</s>", "<pad>",
        "<user>", "</user>", "<assistant>", "</assistant>",
        "<system>", "</system>", "<tool>", "</tool>"
    ]

    @staticmethod
    def load_datasets(max_samples: int = 20000) -> Dataset:
        """Load and combine high-quality conversational datasets."""
        datasets_to_load = [
            ("databricks/databricks-dolly-15k", "train"),
            ("tatsu-lab/alpaca", "train"),
            ("sahil2801/CodeAlpaca-20k", "train"),
        ]

        all_datasets = []
        for dataset_name, split in datasets_to_load:
            try:
                print(f"📚 Loading {dataset_name}...")
                ds = load_dataset(dataset_name, split=split)
                all_datasets.append(ds)
                print(f"✅ Loaded {len(ds):,} examples from {dataset_name}")
            except Exception as e:
                print(f"⚠️ Failed to load {dataset_name}: {e}")

        if not all_datasets:
            raise ValueError("Failed to load any datasets")

        # Combine datasets
        combined_dataset = concatenate_datasets(all_datasets)

        # Add synthetic tool-calling examples
        tool_examples = ConversationalDataProcessor._create_tool_examples()
        tool_dataset = Dataset.from_list(tool_examples)
        combined_dataset = concatenate_datasets([combined_dataset, tool_dataset])

        # Sample subset for training
        max_samples = min(max_samples, len(combined_dataset))
        dataset = combined_dataset.select(range(max_samples))

        print(f"📊 Using {len(dataset):,} training examples")
        return dataset

    @staticmethod
    def _create_tool_examples() -> List[Dict]:
        """Create synthetic examples for tool calling training."""
        return [
            {
                "instruction": "What's the weather like in San Francisco?",
                "input": "",
                "response": "I'll check the current weather for you.\n\n<tool>get_weather</tool>\n{\"location\": \"San Francisco, CA\"}\n\nBased on the weather data, San Francisco is currently 68°F with partly cloudy skies and light winds from the west at 8 mph."
            },
            {
                "instruction": "Search for recent papers about transformer architectures",
                "input": "",
                "response": "I'll search for recent research papers on transformer architectures.\n\n<tool>search_papers</tool>\n{\"query\": \"transformer architecture\", \"date_range\": \"2024\", \"limit\": 10}\n\nI found several recent papers on transformer architectures:\n1. \"Efficient Attention Mechanisms in Large Language Models\" (2024)\n2. \"Scaling Laws for Neural Language Models\" (2024)\n3. \"RoPE: Rotary Position Embedding\" (2024)"
            },
            {
                "instruction": "Calculate the compound interest for $10,000 at 5% annual rate for 10 years",
                "input": "",
                "response": "I'll calculate the compound interest for you.\n\n<tool>calculate</tool>\n{\"expression\": \"10000 * (1 + 0.05)^10\"}\n\nUsing the compound interest formula A = P(1 + r)^t:\n- Principal (P): $10,000\n- Annual rate (r): 5% = 0.05\n- Time (t): 10 years\n\nResult: $16,288.95\nTotal interest earned: $6,288.95"
            }
        ]

    @staticmethod
    def create_tokenizer(dataset: Dataset, vocab_size: int = 32000) -> Tokenizer:
        """Create and train a conversation-aware tokenizer."""
        print("🔤 Creating and training tokenizer...")

        # Initialize BPE tokenizer
        tokenizer = Tokenizer(models.BPE())

        # Set up pre-tokenization for better handling of code and JSON
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.ByteLevel(add_prefix_space=False),
            pre_tokenizers.Punctuation(),
            pre_tokenizers.WhitespaceSplit()
        ])

        # Configure trainer
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=ConversationalDataProcessor.SPECIAL_TOKENS,
            min_frequency=2,
            continuing_subword_prefix="##",
            end_of_word_suffix="</w>"
        )

        # Prepare training data
        training_file = "tokenizer_training_data.txt"
        with open(training_file, "w", encoding="utf-8") as f:
            for item in dataset:
                if "instruction" in item and "response" in item:
                    # Format as conversation
                    f.write(f"<user>{item['instruction']}</user>\n")
                    if item.get("input"):
                        f.write(f"<system>{item['input']}</system>\n")
                    f.write(f"<assistant>{item['response']}</assistant>\n\n")
                elif "text" in item:
                    f.write(f"{item['text']}\n")

            # Add tool calling examples
            f.write("<tool>search</tool>\n{\"query\": \"example\"}\n")
            f.write("<tool>calculate</tool>\n{\"expression\": \"2+2\"}\n")

        # Train tokenizer
        tokenizer.train([training_file], trainer)

        # Set up post-processing
        tokenizer.post_processor = processors.TemplateProcessing(
            single="<s> $A </s>",
            pair="<s> $A </s> $B </s>",
            special_tokens=[("<s>", 1), ("</s>", 2)],
        )

        # Cleanup
        os.remove(training_file)

        print(f"✅ Tokenizer trained with {tokenizer.get_vocab_size():,} tokens")
        return tokenizer


class ConversationalDataCollator:
    """Efficient data collator for conversational training."""

    def __init__(self, tokenizer: Tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        batch_size = len(features)

        # Initialize batch tensors
        input_ids = torch.zeros((batch_size, self.max_length), dtype=torch.long)
        attention_mask = torch.zeros((batch_size, self.max_length), dtype=torch.long)
        labels = torch.full((batch_size, self.max_length), -100, dtype=torch.long)

        for i, feature in enumerate(features):
            ids = feature["input_ids"]
            seq_len = min(len(ids), self.max_length)

            input_ids[i, :seq_len] = torch.tensor(ids[:seq_len])
            attention_mask[i, :seq_len] = 1
            labels[i, :seq_len] = torch.tensor(ids[:seq_len])

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def tokenize_conversations(examples: Dict, tokenizer: Tokenizer, max_length: int = 512) -> Dict:
    """Tokenize conversational data with proper formatting."""
    results = {"input_ids": []}

    for i in range(len(examples["instruction"])):
        # Format conversation
        conversation = f"<user>{examples['instruction'][i]}</user>"
        if examples.get("input") and examples["input"][i]:
            conversation += f"<system>{examples['input'][i]}</system>"
        conversation += f"<assistant>{examples['response'][i]}</assistant>"

        # Tokenize
        encoded = tokenizer.encode(conversation)
        tokens = encoded.ids[:max_length]  # Truncate if needed

        results["input_ids"].append(tokens)

    return results


# =============================================================================
# Training Setup
# =============================================================================

class TrainingProgressCallback(TrainerCallback):
    """Enhanced training progress callback with metrics tracking."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_world_process_zero and logs:
            epoch = state.epoch
            step = state.global_step
            total_steps = state.max_steps
            progress = 100 * step / total_steps if total_steps > 0 else 0

            loss = logs.get("train_loss", 0)
            learning_rate = logs.get("learning_rate", 0)

            print(f"📈 Epoch {epoch:.2f} | Step {step}/{total_steps} ({progress:.1f}%) | Loss: {loss:.4f} | LR: {learning_rate:.2e}")


def setup_lora_training(model: DeepZigConversationalModel, rank: int = 16) -> DeepZigConversationalModel:
    """Set up LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning."""
    if not PEFT_AVAILABLE:
        raise ImportError("PEFT not available. Install with: pip install peft")

    print(f"🔧 Setting up LoRA training with rank {rank}")

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=rank,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )

    model = get_peft_model(model, lora_config)
    print(f"✅ LoRA enabled. Trainable parameters: {model.num_parameters():,}")
    return model


def create_training_args(output_dir: str, num_epochs: int = 3, batch_size: int = 4, has_eval: bool = True) -> TrainingArguments:
    """Create optimized training arguments following best practices."""

    # Base arguments
    args = {
        "output_dir": output_dir,

        # Training schedule
        "num_train_epochs": num_epochs,
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": batch_size,
        "gradient_accumulation_steps": 4,

        # Optimization
        "learning_rate": 2e-4,
        "weight_decay": 0.01,
        "warmup_ratio": 0.03,
        "lr_scheduler_type": "cosine",
        "max_grad_norm": 1.0,

        # Mixed precision and memory optimization
        "fp16": True,
        "gradient_checkpointing": True,   # Now properly implemented
        "dataloader_pin_memory": True,

        # Logging
        "logging_steps": 25,
        "save_steps": 500,
        "save_total_limit": 3,

        # Misc
        "remove_unused_columns": False,
        "report_to": "none",
        "disable_tqdm": False,
        "logging_first_step": True,
    }

    # Add evaluation-specific arguments only if we have an eval dataset
    if has_eval:
        args.update({
            "eval_steps": 500,
            "eval_strategy": "steps",
            "load_best_model_at_end": True,
            "metric_for_best_model": "eval_loss",
            "greater_is_better": False,
        })
    else:
        args.update({
            "eval_strategy": "no",
        })

    return TrainingArguments(**args)


# =============================================================================
# Model Export and Testing
# =============================================================================

def export_to_zig_format(model: DeepZigConversationalModel, tokenizer: Tokenizer, output_dir: str):
    """Export model in Zig-compatible format with proper naming conventions."""
    print("💾 Exporting model in Zig-compatible format...")

    os.makedirs(output_dir, exist_ok=True)

    # Prepare state dict with Zig-compatible naming
    state_dict = {}

    for name, param in model.named_parameters():
        if "lora" in name.lower():
            continue  # Skip LoRA weights for now

        tensor = param.detach().cpu().numpy()

        # Map to Zig-compatible names
        if name == "embed_tokens.weight":
            state_dict["model.embed_tokens.weight"] = tensor
        elif name == "lm_head.weight":
            state_dict["lm_head.weight"] = tensor
        elif name == "norm.weight":
            state_dict["model.norm.weight"] = tensor
        elif name.startswith("layers."):
            parts = name.split(".")
            layer_idx = parts[1]

            if "input_layernorm" in name:
                state_dict[f"model.layers.{layer_idx}.input_layernorm.weight"] = tensor
            elif "post_attention_layernorm" in name:
                state_dict[f"model.layers.{layer_idx}.post_attention_layernorm.weight"] = tensor
            elif "self_attn" in name:
                component = ".".join(parts[3:])
                state_dict[f"model.layers.{layer_idx}.self_attn.{component}"] = tensor
            elif "mlp" in name:
                component = ".".join(parts[3:])
                state_dict[f"model.layers.{layer_idx}.mlp.{component}"] = tensor
        else:
            state_dict[f"model.{name}"] = tensor

    # Save model weights
    save_file(state_dict, os.path.join(output_dir, "model.safetensors"))

    # Save configuration
    config_dict = {
        "model_type": "deepzig_conversational",
        "architectures": ["DeepZigConversationalModel"],
        "vocab_size": model.config.vocab_size,
        "hidden_size": model.config.hidden_size,
        "intermediate_size": model.config.intermediate_size,
        "num_hidden_layers": model.config.num_hidden_layers,
        "num_attention_heads": model.config.num_attention_heads,
        "num_key_value_heads": model.config.num_key_value_heads,
        "max_position_embeddings": model.config.max_position_embeddings,
        "rope_base": model.config.rope_base,
        "rms_norm_eps": model.config.rms_norm_eps,
        "torch_dtype": "bfloat16",
        "use_cache": True,
        "pad_token_id": tokenizer.token_to_id("<pad>"),
        "bos_token_id": tokenizer.token_to_id("<s>"),
        "eos_token_id": tokenizer.token_to_id("</s>"),
    }

    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config_dict, f, indent=2)

    # Save tokenizer
    tokenizer.save(os.path.join(output_dir, "tokenizer.json"))

    # Save generation config
    generation_config = GenerationConfig(
        max_length=2048,
        temperature=0.8,
        top_p=0.9,
        top_k=40,
        repetition_penalty=1.1,
        do_sample=True,
        pad_token_id=tokenizer.token_to_id("<pad>"),
        bos_token_id=tokenizer.token_to_id("<s>"),
        eos_token_id=tokenizer.token_to_id("</s>"),
    )
    generation_config.save_pretrained(output_dir)

    print(f"✅ Model exported to {output_dir}")


def test_model_generation(model: DeepZigConversationalModel, tokenizer: Tokenizer):
    """Test the trained model with conversation examples."""
    print("\n🧪 Testing model generation...")

    model.eval()
    test_prompts = [
        "<user>What is the weather like?</user><assistant>",
        "<user>Explain quantum computing</user><assistant>",
        "<user>Write a Python function to calculate fibonacci</user><assistant>",
    ]

    for prompt in test_prompts:
        print(f"\n💬 Prompt: {prompt}")

        # Tokenize
        encoded = tokenizer.encode(prompt)
        input_ids = torch.tensor([encoded.ids])

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_length=input_ids.shape[1] + 50,
                temperature=0.8,
                do_sample=True,
                pad_token_id=tokenizer.token_to_id("<pad>"),
                eos_token_id=tokenizer.token_to_id("</assistant>"),
            )

        # Decode
        generated_text = tokenizer.decode(outputs[0].tolist())
        print(f"🤖 Generated: {generated_text}")


# =============================================================================
# Main Training Function
# =============================================================================

def main():
    """Main training function with full pipeline."""
    parser = argparse.ArgumentParser(description="Train DeepZig Conversational Model")
    parser.add_argument("--use-lora", action="store_true", help="Use LoRA for parameter-efficient training")
    parser.add_argument("--lora-rank", type=int, default=16, help="LoRA rank (default: 16)")
    parser.add_argument("--max-samples", type=int, default=20000, help="Maximum training samples")
    parser.add_argument("--eval-split", type=float, default=0.1, help="Evaluation split ratio")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Training batch size")
    parser.add_argument("--output-dir", type=str, default="deepzig-conversational-model", help="Output directory")

    args = parser.parse_args()

    print("🚀 Starting DeepZig Conversational Model Training")
    print("=" * 60)

    # 1. Load and process data
    print("📚 Loading datasets...")
    dataset = ConversationalDataProcessor.load_datasets(max_samples=args.max_samples)

    # 2. Create tokenizer
    tokenizer = ConversationalDataProcessor.create_tokenizer(dataset)

    # 3. Tokenize dataset
    print("🔤 Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        lambda x: tokenize_conversations(x, tokenizer),
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing conversations"
    )

    # 4. Train/eval split
    if args.eval_split > 0:
        split_dataset = tokenized_dataset.train_test_split(test_size=args.eval_split, seed=42)
        train_dataset = split_dataset["train"]
        eval_dataset = split_dataset["test"]
    else:
        train_dataset = tokenized_dataset
        eval_dataset = None

    # 5. Create model
    print("🏗️ Creating model...")
    config = DeepZigConfig(vocab_size=tokenizer.get_vocab_size())
    model = DeepZigConversationalModel(config)

    print(f"📊 Model parameters: {model.num_parameters():,}")

    # 6. Setup LoRA if requested
    if args.use_lora:
        model = setup_lora_training(model, rank=args.lora_rank)

    # 7. Create data collator
    data_collator = ConversationalDataCollator(tokenizer)

    # 8. Setup training
    training_args = create_training_args(args.output_dir, args.epochs, args.batch_size, has_eval=(eval_dataset is not None))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=[TrainingProgressCallback()],
    )

    # 9. Train model
    print("🏃 Starting training...")
    trainer.train()

    # 10. Save model with safe serialization disabled for tied weights
    print("💾 Saving model...")
    trainer.save_model(safe_serialization=False)

    # 11. Export to Zig format
    export_to_zig_format(model, tokenizer, args.output_dir)

    # 12. Test generation
    test_model_generation(model, tokenizer)

    print("\n🎉 Training completed successfully!")
    print(f"📁 Model saved to: {args.output_dir}")
    print("🔧 Ready for Zig integration!")


if __name__ == "__main__":
    main()
