from collections import Counter, defaultdict
import random
import re


class BigramModel:
    def __init__(self, corpus):
        words = re.findall(r"\b\w+\b", " ".join(corpus).lower())
        pairs = list(zip(words[:-1], words[1:]))
        counts = Counter(pairs)

        self.bigram_probs = defaultdict(dict)
        totals = Counter(first for first, _ in pairs)

        for (first, second), count in counts.items():
            self.bigram_probs[first][second] = count / totals[first]

    def generate_text(self, start_word, length=20):
        current = start_word.lower()
        generated = [current]

        for _ in range(length - 1):
            next_words = self.bigram_probs.get(current)
            if not next_words:
                break

            current = random.choices(
                list(next_words.keys()),
                weights=list(next_words.values()),
            )[0]
            generated.append(current)

        return " ".join(generated)