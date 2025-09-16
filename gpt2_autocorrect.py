#!/usr/bin/env python3
"""
GPT-2 Autocorrect System
Combines NLTK vocabulary matching with GPT-2 context scoring for accurate spell correction.
"""

import os
import re
import difflib
import torch
import nltk
from collections import defaultdict
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Set HF_HOME to avoid permission issues
os.environ['HF_HOME'] = os.path.expanduser('~/.cache/huggingface')

# Download required NLTK data
try:
    nltk.download('words', quiet=True)
    nltk.download('gutenberg', quiet=True)
except:
    print("Warning: Could not download NLTK data. Some features may not work.")

class GPT2Autocorrect:
    """GPT-2 based autocorrect system using context-aware scoring."""
    
    def __init__(self, model_name: str = "gpt2"):
        """Initialize the autocorrect system."""
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.english_vocab = None
        self._load_model()
        self._load_vocabulary()
    
    def _load_model(self):
        """Load GPT-2 model and tokenizer."""
        print("🔄 Loading GPT-2 model...")
        cache_dir = os.path.expanduser('~/.cache/huggingface')
        
        self.tokenizer = GPT2Tokenizer.from_pretrained(
            self.model_name, 
            cache_dir=cache_dir,
            padding_side='left'
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = GPT2LMHeadModel.from_pretrained(
            self.model_name, 
            cache_dir=cache_dir
        )
        self.model.eval()
        print("✅ Model loaded successfully")
    
    def _load_vocabulary(self):
        """Load English vocabulary from NLTK."""
        print("🔄 Loading English vocabulary...")
        try:
            from nltk.corpus import words
            self.english_vocab = set(words.words())
            print(f"✅ Loaded {len(self.english_vocab)} English words")
        except:
            # Fallback vocabulary for common words
            self.english_vocab = {
                'accommodation', 'receive', 'separate', 'necessary', 'definitely',
                'occurrence', 'embarrass', 'maintenance', 'privilege', 'recommend',
                'accomplish', 'acknowledge', 'acquaintance', 'acquire', 'address',
                'apparent', 'argument', 'assistant', 'attendance', 'beginning',
                'believe', 'business', 'calendar', 'category', 'cemetery',
                'changeable', 'committee', 'conscience', 'conscious', 'definite',
                'desperate', 'disappear', 'disappoint', 'embarrass', 'environment',
                'existence', 'experience', 'familiar', 'fascinate', 'foreign',
                'government', 'grammar', 'guarantee', 'harass', 'height',
                'immediately', 'independent', 'intelligence', 'interrupt', 'irrelevant',
                'knowledge', 'library', 'license', 'lightning', 'maintenance',
                'mathematics', 'medicine', 'millennium', 'miniature', 'miscellaneous',
                'misspell', 'necessary', 'neighbor', 'noticeable', 'occasion',
                'occurrence', 'parallel', 'pastime', 'perceive', 'permanent',
                'perseverance', 'personnel', 'possession', 'precede', 'prejudice',
                'principal', 'privilege', 'proceed', 'pronunciation', 'questionnaire',
                'receive', 'recommend', 'reference', 'relevant', 'religious',
                'remember', 'restaurant', 'rhythm', 'ridiculous', 'sacrilegious',
                'secretary', 'separate', 'sophomore', 'successful', 'supersede',
                'surprise', 'temperament', 'tendency', 'thorough', 'tomorrow',
                'tragedy', 'truly', 'unfortunately', 'until', 'vacuum',
                'vegetable', 'vicious', 'weird', 'whether', 'writing'
            }
            print(f"✅ Using fallback vocabulary with {len(self.english_vocab)} words")
    
    def score_text(self, text: str) -> float:
        """Score text using GPT-2 perplexity (lower is better)."""
        if not text or not text.strip():
            return 0.0
        
        try:
            input_ids = self.tokenizer.encode(text, return_tensors='pt')
            
            # Ensure we have at least one token
            if input_ids.shape[1] == 0:
                return 0.0
            
            with torch.no_grad():
                outputs = self.model(input_ids, labels=input_ids)
            
            # Get the loss and check for valid values
            loss = outputs.loss.item()
            
            # Check for NaN or infinite values
            if torch.isnan(torch.tensor(loss)) or torch.isinf(torch.tensor(loss)):
                return 0.0
            
            # Return negative loss (higher is better)
            return -loss
            
        except Exception as e:
            print(f"Warning: Error scoring text '{text}': {e}")
            return 0.0
    
    def get_gpt2_suggestions(self, word: str, context_words: list = None, max_suggestions: int = 3) -> list:
        """Generate word suggestions using GPT-2."""
        if context_words is None:
            context_words = []
        
        # Create context for GPT-2
        context_text = " ".join(context_words[-3:]) + " " if context_words else ""
        prompt = context_text + word + " "
        
        # Tokenize and generate
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt')
        
        with torch.no_grad():
            # Generate multiple completions
            outputs = self.model.generate(
                input_ids,
                max_length=input_ids.shape[1] + 1,
                num_return_sequences=max_suggestions,
                num_beams=max_suggestions * 2,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        suggestions = []
        for output in outputs:
            # Decode the generated text
            generated_text = self.tokenizer.decode(output, skip_special_tokens=True)
            # Extract the next word after our input
            generated_words = generated_text[len(prompt):].strip().split()
            if generated_words:
                next_word = generated_words[0].lower()
                # Clean the word (remove punctuation)
                next_word = re.sub(r'[^\w]', '', next_word)
                if next_word and next_word not in suggestions:
                    suggestions.append(next_word)
        
        return suggestions[:max_suggestions]

    def get_candidates(self, word: str, context_words: list = None, max_candidates: int = 5) -> list:
        """Get candidate corrections for a misspelled word."""
        # Check if word is already correct
        if word.lower() in self.english_vocab:
            return [word]
        
        candidates = []
        
        # Get similar words from vocabulary
        vocab_candidates = difflib.get_close_matches(
            word.lower(), 
            self.english_vocab, 
            n=max_candidates, 
            cutoff=0.4
        )
        
        # If no close matches, try with lower cutoff
        if not vocab_candidates:
            vocab_candidates = difflib.get_close_matches(
                word.lower(), 
                self.english_vocab, 
                n=max_candidates, 
                cutoff=0.2
            )
        
        candidates.extend(vocab_candidates)
        
        # Add GPT-2 generated suggestions
        try:
            gpt2_suggestions = self.get_gpt2_suggestions(word, context_words, max_suggestions=3)
            for suggestion in gpt2_suggestions:
                if suggestion not in candidates and len(suggestion) > 1:
                    candidates.append(suggestion)
        except Exception as e:
            print(f"Warning: GPT-2 suggestions failed: {e}")
        
        # Always include original word as fallback
        if word not in candidates:
            candidates.append(word)
        
        # Limit to max_candidates
        return candidates[:max_candidates]
    
    def autocorrect_word(self, word: str, context_words: list = None, max_candidates: int = 5) -> str:
        """Autocorrect a single word using context."""
        if context_words is None:
            context_words = []
        
        # Get candidate corrections (now includes GPT-2 suggestions)
        candidates = self.get_candidates(word, context_words, max_candidates)
        
        if len(candidates) == 1:
            return candidates[0]
        
        # Score each candidate in context
        scores = {}
        context_text = " ".join(context_words[-5:]) + " " if context_words else ""
        
        for candidate in candidates:
            prompt = context_text + candidate
            scores[candidate] = self.score_text(prompt)
        
        # Return the best scoring candidate
        best_candidate = max(scores, key=scores.get)
        return best_candidate
    
    def autocorrect_word_with_scores(self, word: str, context_words: list = None, max_candidates: int = 5) -> tuple:
        """Autocorrect a word and return both the result and scoring details."""
        if context_words is None:
            context_words = []
        
        # Get candidate corrections (now includes GPT-2 suggestions)
        candidates = self.get_candidates(word, context_words, max_candidates)
        
        if len(candidates) == 1:
            return candidates[0], {candidates[0]: 0.0}
        
        # Score each candidate in context
        scores = {}
        context_text = " ".join(context_words[-5:]) + " " if context_words else ""
        
        for candidate in candidates:
            prompt = context_text + candidate
            score = self.score_text(prompt)
            scores[candidate] = score
            
            # Debug information
            if torch.isnan(torch.tensor(score)) or torch.isinf(torch.tensor(score)):
                print(f"Debug: Invalid score for '{candidate}' with prompt '{prompt}': {score}")
        
        # Return the best scoring candidate and all scores
        best_candidate = max(scores, key=scores.get)
        return best_candidate, scores
    
    def autocorrect_sentence(self, sentence: str) -> str:
        """Autocorrect an entire sentence."""
        # Clean and tokenize sentence
        words = re.findall(r'\b\w+\b', sentence.lower())
        corrected_words = []
        
        for i, word in enumerate(words):
            # Use already corrected words as context
            context = corrected_words[-5:] if corrected_words else []
            corrected_word = self.autocorrect_word(word, context)
            corrected_words.append(corrected_word)
        
        # Reconstruct sentence with original punctuation
        result = sentence
        for original, corrected in zip(words, corrected_words):
            if original != corrected:
                # Replace word while preserving case
                pattern = r'\b' + re.escape(original) + r'\b'
                result = re.sub(pattern, corrected, result, flags=re.IGNORECASE)
        
        return result
    
    def autocorrect_text(self, text: str) -> str:
        """Autocorrect multiple sentences in a text."""
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        corrected_sentences = []
        
        for sentence in sentences:
            if sentence.strip():
                corrected = self.autocorrect_sentence(sentence)
                corrected_sentences.append(corrected)
        
        return ' '.join(corrected_sentences)


def main():
    """Main function for interactive autocorrect."""
    print("🤖 GPT-2 Autocorrect System")
    print("=" * 50)
    
    # Initialize autocorrect system
    autocorrect = GPT2Autocorrect()
    
    print("\nEnter text to autocorrect (type 'quit' to exit):")
    print("=" * 50)
    
    while True:
        try:
            user_input = input("\nText: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("👋 Goodbye!")
                break
            
            if not user_input:
                continue
            
            # Autocorrect the input
            corrected = autocorrect.autocorrect_text(user_input)
            
            print(f"\nOriginal:  {user_input}")
            print(f"Corrected: {corrected}")
            
            # Show word-by-word changes with scoring details
            original_words = re.findall(r'\b\w+\b', user_input.lower())
            corrected_words = re.findall(r'\b\w+\b', corrected.lower())
            
            changes = []
            for orig, corr in zip(original_words, corrected_words):
                if orig != corr:
                    # Get scoring details for this word
                    try:
                        context_words = [w for w in original_words[:original_words.index(orig)]]
                        _, scores = autocorrect.autocorrect_word_with_scores(orig, context_words)
                        score = scores.get(corr, 0)
                        # Check if score is valid
                        if torch.isnan(torch.tensor(score)) or torch.isinf(torch.tensor(score)):
                            score_info = " (score: N/A)"
                        else:
                            score_info = f" (score: {score:.3f})"
                        changes.append(f"{orig} → {corr}{score_info}")
                    except Exception as e:
                        changes.append(f"{orig} → {corr} (error: {str(e)[:20]})")
            
            if changes:
                print(f"Changes:   {', '.join(changes)}")
            else:
                print("No changes needed.")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()