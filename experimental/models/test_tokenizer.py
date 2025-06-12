#!/usr/bin/env python3
"""
Quick tokenizer test to validate improvements before full training
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from tokenizers import Tokenizer, models, pre_tokenizers, trainers, processors

def test_tokenizer():
    """Test tokenizer functionality"""
    print("🔤 Testing tokenizer...")

    # Create simple test data
    test_data = [
        "<s><user>Hello</user><assistant>Hello! How can I help you?</assistant></s>",
        "<s><user>What is AI?</user><assistant>AI stands for Artificial Intelligence.</assistant></s>",
        "<s><user>Thanks</user><assistant>You're welcome!</assistant></s>",
    ]

    # Initialize tokenizer
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.WhitespaceSplit(),
        pre_tokenizers.Punctuation()
    ])

    # Special tokens
    special_tokens = [
        "<unk>", "<s>", "</s>", "<pad>",
        "<user>", "</user>", "<assistant>", "</assistant>",
        "<system>", "</system>", "<tool>", "</tool>"
    ]

    # Train tokenizer
    trainer = trainers.BpeTrainer(
        vocab_size=1000,  # Small for testing
        special_tokens=special_tokens,
        min_frequency=1,
        show_progress=True
    )

    # Create training file
    with open("test_data.txt", "w") as f:
        for text in test_data:
            f.write(text + "\n")

        # Add more varied content
        f.write("Hello world!\nHow are you today?\nI'm doing great, thanks for asking.\n")
        f.write("Python is a programming language.\nThe weather is nice today.\n")

    # Train
    tokenizer.train(["test_data.txt"], trainer)

    # FIXED: Remove post-processor since we manually add <s> and </s> tokens
    # This prevents duplication of BOS/EOS tokens
    tokenizer.post_processor = None

    print(f"✅ Tokenizer trained with {tokenizer.get_vocab_size()} tokens")

    # Test encoding/decoding
    for test_text in test_data:
        print(f"\n📝 Original: {test_text}")

        # Encode
        encoded = tokenizer.encode(test_text)
        print(f"🔢 Tokens: {encoded.ids}")
        print(f"📊 Length: {len(encoded.ids)}")

        # Decode
        try:
            decoded = tokenizer.decode(encoded.ids, skip_special_tokens=False)
            print(f"🔤 Decoded (raw): {decoded}")

            decoded_clean = tokenizer.decode(encoded.ids, skip_special_tokens=True)
            print(f"✨ Decoded (clean): {decoded_clean}")
        except Exception as e:
            print(f"❌ Decode error: {e}")

    # Test special tokens
    print(f"\n🔍 Special token IDs:")
    for token in special_tokens:
        token_id = tokenizer.token_to_id(token)
        print(f"  {token}: {token_id}")

    # Cleanup
    os.remove("test_data.txt")

    return tokenizer

if __name__ == "__main__":
    test_tokenizer()
