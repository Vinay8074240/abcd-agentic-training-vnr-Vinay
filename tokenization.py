from collections import defaultdict, Counter
import re

class BPE:
    def __init__(self, num_merges=50):
        self.num_merges = num_merges
        self.merges = []

    # Step 1: Build initial vocabulary
    def build_vocab(self, corpus):
        vocab = defaultdict(int)
        for word in corpus:
            word = " ".join(list(word)) + " </w>"
            vocab[word] += 1
        return vocab

    # Step 2: Get frequency of symbol pairs
    def get_stats(self, vocab):
        pairs = defaultdict(int)
        for word, freq in vocab.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i + 1])] += freq
        return pairs

    # Step 3: Merge the most frequent pair
    def merge_vocab(self, pair, vocab):
        new_vocab = {}
        bigram = re.escape(" ".join(pair))
        pattern = re.compile(r"(?<!\S)" + bigram + r"(?!\S)")
        replacement = "".join(pair)

        for word in vocab:
            new_word = pattern.sub(replacement, word)
            new_vocab[new_word] = vocab[word]

        return new_vocab

    # Step 4: Train BPE
    def train(self, corpus):
        vocab = self.build_vocab(corpus)

        for i in range(self.num_merges):
            pairs = self.get_stats(vocab)
            if not pairs:
                break

            best_pair = max(pairs, key=pairs.get)
            vocab = self.merge_vocab(best_pair, vocab)
            self.merges.append(best_pair)

            print(f"Merge {i + 1}: {best_pair}")

        self.vocab = vocab

    # Step 5: Tokenize new words
    def tokenize(self, word):
        tokens = list(word) + ["</w>"]

        for pair in self.merges:
            i = 0
            while i < len(tokens) - 1:
                if tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
                    tokens[i:i + 2] = ["".join(pair)]
                else:
                    i += 1

        return tokens[:-1]  # remove </w>


# -------------------------------
# Example Usage
# -------------------------------

if __name__ == "__main__":
    corpus = [
        "low",
        "lower",
        "lowest",
        "newer",
        "wider"
    ]

    bpe = BPE(num_merges=20)
    bpe.train(corpus)

    print("\nFinal Vocabulary:")
    for word in bpe.vocab:
        print(word)

    print("\nTokenization Examples:")
    words = ["lowest", "newest", "widest"]
    for w in words:
        print(w, "→", bpe.tokenize(w))
