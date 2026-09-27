"""
Bigram model for the Module 3 FastAPI activity.

The functions below are moved from the Module 2 lecture notebook
(Module_2_Practical_2_Word_Sampling): simple_tokenizer, analyze_bigrams,
and generate_text. BigramModel wraps them so main.py can call
BigramModel(corpus) and bigram_model.generate_text(start_word, length).
"""

from collections import defaultdict, Counter
import random
import re


def simple_tokenizer(text, frequency_threshold=5):
    """Simple tokenizer that splits text into words."""
    # Convert to lowercase and extract words using regex
    tokens = re.findall(r"\b\w+\b", text.lower())
    if not frequency_threshold:
        return tokens
    # Count word frequencies
    word_counts = Counter(tokens)
    # Define a threshold for less frequent words (e.g., words appearing fewer than 5 times)
    filtered_tokens = [
        token for token in tokens if word_counts[token] >= frequency_threshold
    ]
    return filtered_tokens


def analyze_bigrams(text, frequency_threshold=None):
    """Analyze text to compute bigram probabilities."""
    words = simple_tokenizer(text, frequency_threshold)
    bigrams = list(zip(words[:-1], words[1:]))  # Create bigrams

    # Count bigram and unigram frequencies
    bigram_counts = Counter(bigrams)
    unigram_counts = Counter(words)

    # Compute bigram probabilities
    bigram_probs = defaultdict(dict)
    for (word1, word2), count in bigram_counts.items():
        bigram_probs[word1][word2] = count / unigram_counts[word1]

    return list(unigram_counts.keys()), bigram_probs


def generate_text(bigram_probs, start_word, num_words=20):
    """Generate text based on bigram probabilities."""
    current_word = start_word.lower()
    generated_words = [current_word]

    for _ in range(num_words - 1):
        next_words = bigram_probs.get(current_word)
        if not next_words:  # If no bigrams for the current word, stop generating
            break

        # Choose the next word based on probabilities
        next_word = random.choices(
            list(next_words.keys()), weights=next_words.values()
        )[0]
        generated_words.append(next_word)
        current_word = next_word  # Move to the next word

    return " ".join(generated_words)


class BigramModel:
    """Bigram text generator used by the FastAPI app."""

    def __init__(self, corpus, frequency_threshold=None):
        # main.py passes a list of sentences, so join them into one text
        text = " ".join(corpus)
        self.vocab, self.bigram_probs = analyze_bigrams(text, frequency_threshold)

    def generate_text(self, start_word, length=20):
        return generate_text(self.bigram_probs, start_word, num_words=length)
