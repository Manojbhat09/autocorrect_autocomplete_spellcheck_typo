#!/usr/bin/env python3
"""
Advanced Autocorrect System
Combines NLTK vocabulary matching with language model context scoring for accurate spell correction.
Supports both GPT-2 and Qwen models.
"""

import os
import re
import difflib
import torch
import nltk
from collections import defaultdict
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoTokenizer, AutoModelForCausalLM

# Set HF_HOME to avoid permission issues
os.environ['HF_HOME'] = os.path.expanduser('~/.cache/huggingface')

# Download required NLTK data
try:
    nltk.download('words', quiet=True)
    nltk.download('gutenberg', quiet=True)
except:
    print("Warning: Could not download NLTK data. Some features may not work.")

class AutocorrectModel:
    """Advanced autocorrect system using context-aware scoring with multiple model support."""
    
    def __init__(self, model_name: str = "gpt2", model_type: str = "auto"):
        """Initialize the autocorrect system."""
        self.model_name = model_name
        self.model_type = model_type
        self.model = None
        self.tokenizer = None
        self.english_vocab = None
        
        # Auto-detect model type if not specified
        if model_type == "auto":
            if "qwen" in model_name.lower():
                self.model_type = "qwen"
            elif "gpt2" in model_name.lower():
                self.model_type = "gpt2"
            else:
                self.model_type = "gpt2"  # Default fallback
        
        self._load_model()
        self._load_vocabulary()
    
    def _load_model(self):
        """Load model and tokenizer based on model type."""
        print(f"🔄 Loading {self.model_type.upper()} model: {self.model_name}")
        cache_dir = os.path.expanduser('~/.cache/huggingface')
        
        try:
            if self.model_type == "qwen":
                # Load Qwen model
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    cache_dir=cache_dir,
                    trust_remote_code=True
                )
                
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    cache_dir=cache_dir,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto",
                    trust_remote_code=True
                )
                
                # Set pad token if not set
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                    
            else:  # GPT-2
                # Load GPT-2 model
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
            print(f"✅ {self.model_type.upper()} model loaded successfully")
            
        except Exception as e:
            print(f"❌ Failed to load {self.model_type} model: {e}")
            print("🔄 Falling back to GPT-2...")
            self.model_type = "gpt2"
            self.model_name = "gpt2"
            self._load_model()
    
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
        """Score text using model perplexity (lower is better)."""
        if not text or not text.strip():
            return 0.0
        
        try:
            # Tokenize text based on model type
            if self.model_type == "qwen":
                # For Qwen, use standard encoding without special tokens for scoring
                input_ids = self.tokenizer.encode(text, return_tensors='pt', add_special_tokens=False)
            else:
                # For GPT-2, use standard encoding
                input_ids = self.tokenizer.encode(text, return_tensors='pt')
            
            # Ensure we have at least one token
            if input_ids.shape[1] == 0:
                return 0.0
            
            # Move to model device
            input_ids = input_ids.to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model(input_ids, labels=input_ids)
            
            # Get the loss and check for valid values
            loss = outputs.loss.item()
            
            # Check for NaN or infinite values
            if torch.isnan(torch.tensor(loss)) or torch.isinf(torch.tensor(loss)):
                return 0.0
            
            # Apply model-specific scaling for better differentiation
            if self.model_type == "qwen":
                # Qwen tends to have different loss scales, so we normalize
                scaled_loss = loss * 0.5  # Scale down Qwen losses
            else:
                scaled_loss = loss
            
            # Return negative loss (higher is better)
            return -scaled_loss
            
        except Exception as e:
            print(f"Warning: Error scoring text '{text}': {e}")
            return 0.0
    
    def get_model_suggestions(self, word: str, context_words: list = None, max_suggestions: int = 3) -> list:
        """Generate word suggestions using the loaded model."""
        if context_words is None:
            context_words = []
        
        try:
            if self.model_type == "qwen":
                # Use Qwen's instruction-following capabilities with better prompting
                context_text = " ".join(context_words[-3:]) + " " if context_words else ""
                
                # Create a more focused instruction for Qwen
                instruction = f"Fix the spelling of '{word}'. Context: '{context_text.strip()}'. Give only the corrected word:"
                
                messages = [
                    {"role": "user", "content": instruction}
                ]
                
                # Apply chat template
                try:
                    input_text = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                except:
                    input_text = f"User: {instruction}\n\nAssistant:"
                
                input_ids = self.tokenizer.encode(input_text, return_tensors='pt', add_special_tokens=False)
                
            else:
                # GPT-2 approach - simple completion
                context_text = " ".join(context_words[-3:]) + " " if context_words else ""
                prompt = context_text + word + " "
                input_ids = self.tokenizer.encode(prompt, return_tensors='pt')
            
            input_ids = input_ids.to(self.model.device)
            
            with torch.no_grad():
                # Generate multiple completions with better parameters
                if self.model_type == "qwen":
                    # Qwen-specific parameters
                    outputs = self.model.generate(
                        input_ids,
                        max_length=input_ids.shape[1] + 10,  # Shorter for focused responses
                        num_return_sequences=max_suggestions,
                        do_sample=True,
                        temperature=0.3,  # Lower temperature for more focused corrections
                        top_p=0.8,
                        top_k=20,  # Limit vocabulary
                        pad_token_id=self.tokenizer.eos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        repetition_penalty=1.1
                    )
                else:
                    # GPT-2 parameters
                    outputs = self.model.generate(
                        input_ids,
                        max_length=input_ids.shape[1] + 5,
                        num_return_sequences=max_suggestions,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                        pad_token_id=self.tokenizer.eos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id
                    )
            
            suggestions = []
            for output in outputs:
                # Decode the generated text
                generated_text = self.tokenizer.decode(output, skip_special_tokens=True)
                
                if self.model_type == "qwen":
                    # Extract suggestions from Qwen's response
                    # Look for the word after the instruction
                    response_part = generated_text[len(input_text):].strip()
                    # Split by common delimiters and extract words
                    words = re.findall(r'\b[a-zA-Z]+\b', response_part.lower())
                    for w in words:
                        if w != word.lower() and len(w) > 1 and w not in suggestions:
                            suggestions.append(w)
                else:
                    # GPT-2 approach - extract next word
                    context_text = " ".join(context_words[-3:]) + " " if context_words else ""
                    prompt = context_text + word + " "
                    generated_words = generated_text[len(prompt):].strip().split()
                    if generated_words:
                        next_word = generated_words[0].lower()
                        next_word = re.sub(r'[^\w]', '', next_word)
                        if next_word and next_word not in suggestions:
                            suggestions.append(next_word)
            
            return suggestions[:max_suggestions]
            
        except Exception as e:
            print(f"Warning: Model suggestions failed: {e}")
            return []

    def get_candidates(self, word: str, context_words: list = None, max_candidates: int = 5) -> list:
        """Get candidate corrections for a misspelled word."""
        # Check if word is already correct
        if word.lower() in self.english_vocab:
            return [word]
        
        candidates = []
        
        # Get similar words from vocabulary (prioritize these)
        vocab_candidates = difflib.get_close_matches(
            word.lower(), 
            self.english_vocab, 
            n=max_candidates, 
            cutoff=0.6  # Higher cutoff for better matches
        )
        
        # If no close matches, try with lower cutoff
        if not vocab_candidates:
            vocab_candidates = difflib.get_close_matches(
                word.lower(), 
                self.english_vocab, 
                n=max_candidates, 
                cutoff=0.3
            )
        
        # Add vocabulary candidates first (they're more reliable)
        candidates.extend(vocab_candidates)
        
        # Add model generated suggestions (but filter them better)
        try:
            model_suggestions = self.get_model_suggestions(word, context_words, max_suggestions=2)
            for suggestion in model_suggestions:
                # Better filtering for model suggestions
                if (suggestion not in candidates and 
                    len(suggestion) > 1 and 
                    len(suggestion) <= len(word) + 2 and  # Not too much longer
                    suggestion.isalpha() and  # Only alphabetic
                    suggestion.lower() != word.lower()):  # Not the same word
                    candidates.append(suggestion)
        except Exception as e:
            print(f"Warning: Model suggestions failed: {e}")
        
        # Always include original word as fallback
        if word not in candidates:
            candidates.append(word)
        
        # Limit to max_candidates, prioritizing vocab candidates
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
            base_score = self.score_text(prompt)
            
            # Apply bias to favor vocabulary matches
            if candidate in self.english_vocab:
                # Boost vocabulary matches
                scores[candidate] = base_score + 1.0
            else:
                scores[candidate] = base_score
        
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
            base_score = self.score_text(prompt)
            
            # Apply bias to favor vocabulary matches
            if candidate in self.english_vocab:
                # Boost vocabulary matches
                final_score = base_score + 1.0
            else:
                final_score = base_score
                
            scores[candidate] = final_score
            
            # Debug information
            if torch.isnan(torch.tensor(final_score)) or torch.isinf(torch.tensor(final_score)):
                print(f"Debug: Invalid score for '{candidate}' with prompt '{prompt}': {final_score}")
        
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
    
    def compare_models(self, word: str, context_words: list = None) -> dict:
        """Compare suggestions from different model approaches."""
        if context_words is None:
            context_words = []
        
        # Get vocabulary candidates
        vocab_candidates = difflib.get_close_matches(
            word.lower(), 
            self.english_vocab, 
            n=5, 
            cutoff=0.4
        )
        
        # Get model suggestions
        model_suggestions = self.get_model_suggestions(word, context_words, max_suggestions=3)
        
        # Score all candidates
        all_candidates = list(set(vocab_candidates + model_suggestions + [word]))
        context_text = " ".join(context_words[-5:]) + " " if context_words else ""
        
        scores = {}
        for candidate in all_candidates:
            prompt = context_text + candidate
            scores[candidate] = self.score_text(prompt)
        
        return {
            'vocab_candidates': vocab_candidates,
            'model_suggestions': model_suggestions,
            'all_scores': scores,
            'model_type': self.model_type
        }


def main():
    """Main function for interactive autocorrect."""
    print("🤖 Advanced Autocorrect System")
    print("=" * 50)
    
    # Model selection
    print("Available models:")
    print("1. GPT-2 (default)")
    print("2. Qwen2.5-0.5B-Instruct")
    print("3. Custom model")
    
    choice = input("\nSelect model (1-3, or press Enter for GPT-2): ").strip()
    
    if choice == "2":
        model_name = "Qwen/Qwen2.5-0.5B-Instruct"
        model_type = "qwen"
    elif choice == "3":
        model_name = input("Enter model name/path: ").strip()
        model_type = "auto"
    else:
        model_name = "gpt2"
        model_type = "gpt2"
    
    print(f"\n🔄 Initializing {model_name}...")
    
    # Initialize autocorrect system
    autocorrect = AutocorrectModel(model_name=model_name, model_type=model_type)
    
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


# Backward compatibility alias
GPT2Autocorrect = AutocorrectModel

if __name__ == "__main__":
    main()