#!/usr/bin/env python3
"""
Test the specific "stting" → "setting" correction fix.
"""

from gpt2_autocorrect import AutocorrectModel

def test_stting_correction():
    """Test that 'stting' gets corrected to 'setting'."""
    print("🧪 Testing 'stting' → 'setting' correction")
    print("=" * 50)
    
    # Test with GPT-2
    print("🔄 Testing with GPT-2...")
    gpt2_model = AutocorrectModel(model_name="gpt2", model_type="gpt2")
    
    test_cases = [
        ("stting", ["I", "am"]),
        ("stting", ["The", "chair"]),
        ("stting", []),
    ]
    
    for word, context in test_cases:
        print(f"\n📝 Word: '{word}' (context: {context})")
        
        # Get detailed analysis
        result = gpt2_model.compare_models(word, context)
        
        print(f"   Vocab candidates: {result['vocab_candidates']}")
        print(f"   Model suggestions: {result['model_suggestions']}")
        print(f"   All scores: {dict(sorted(result['all_scores'].items(), key=lambda x: x[1], reverse=True))}")
        
        # Get final correction
        corrected = gpt2_model.autocorrect_word(word, context)
        print(f"   ✅ Final correction: '{word}' → '{corrected}'")
        
        if corrected == "setting":
            print("   🎉 SUCCESS: Correctly chose 'setting'!")
        else:
            print(f"   ❌ FAILED: Expected 'setting', got '{corrected}'")

if __name__ == "__main__":
    test_stting_correction()
