#!/usr/bin/env python3
"""
Compare GPT-2 and Qwen models for autocorrection.
"""

from gpt2_autocorrect import AutocorrectModel

def compare_models():
    """Compare GPT-2 and Qwen models."""
    print("🔍 Model Comparison: GPT-2 vs Qwen")
    print("=" * 60)
    
    # Test cases
    test_cases = [
        ("recieve", ["I", "will"]),
        ("accomodation", ["The", "hotel"]),
        ("beleive", ["I", "cannot"]),
        ("definately", ["That", "is"]),
        ("seperate", ["Please", "keep"]),
        ("tomrrow", ["See", "you"]),
        ("watever", ["I", "don't", "care"]),
        ("stting", ["I", "am"]),
    ]
    
    # Initialize both models
    print("🔄 Loading GPT-2 model...")
    gpt2_model = AutocorrectModel(model_name="gpt2", model_type="gpt2")
    
    print("🔄 Loading Qwen model...")
    qwen_model = AutocorrectModel(model_name="Qwen/Qwen2.5-0.5B-Instruct", model_type="qwen")
    
    print("\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)
    
    for word, context in test_cases:
        print(f"\n📝 Word: '{word}' (context: {' '.join(context)})")
        print("-" * 40)
        
        # Get GPT-2 results
        gpt2_result = gpt2_model.compare_models(word, context)
        print(f"🤖 GPT-2:")
        print(f"   Vocab candidates: {gpt2_result['vocab_candidates']}")
        print(f"   Model suggestions: {gpt2_result['model_suggestions']}")
        print(f"   Best scores: {dict(sorted(gpt2_result['all_scores'].items(), key=lambda x: x[1], reverse=True)[:3])}")
        
        # Get Qwen results
        qwen_result = qwen_model.compare_models(word, context)
        print(f"🧠 Qwen:")
        print(f"   Vocab candidates: {qwen_result['vocab_candidates']}")
        print(f"   Model suggestions: {qwen_result['model_suggestions']}")
        print(f"   Best scores: {dict(sorted(qwen_result['all_scores'].items(), key=lambda x: x[1], reverse=True)[:3])}")
        
        # Show final corrections
        gpt2_corrected = gpt2_model.autocorrect_word(word, context)
        qwen_corrected = qwen_model.autocorrect_word(word, context)
        
        print(f"✅ Final corrections:")
        print(f"   GPT-2: '{word}' → '{gpt2_corrected}'")
        print(f"   Qwen:  '{word}' → '{qwen_corrected}'")
        
        if gpt2_corrected != qwen_corrected:
            print("   🔄 DIFFERENT RESULTS!")
        else:
            print("   ⚖️  Same result")

if __name__ == "__main__":
    compare_models()
