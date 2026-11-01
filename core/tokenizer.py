"""
OpenFoundation-Engine: Custom BPE Tokenizer Engine
Implements Byte-Pair Encoding (BPE) tokenization with:
- GPT-4 / LLaMA style regex pre-tokenization
- Merge rule learning from raw multi-lingual corpus (English, Arabic, Code)
- Compression ratio benchmarking (bytes per token, chars per token)
- Special tokens handling (<pad>, <bos>, <eos>, <unk>, <sep>)
- Serialization & Deserialization
"""

import json
import re
from typing import Dict, List, Tuple, Optional


class FoundationTokenizer:
    """
    Subword Byte-Pair Encoding Tokenizer trained for multi-lingual foundation models.
    """
    # Fast standard regex pattern matching words, numbers, and punctuation
    PRETOKENIZE_PATTERN = re.compile(
        r"""'(?:[sdmt]|ll|ve|re)|[^\r\n\w]?\w+|\d{1,3}| ?[^\s\w]+[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+""",
        re.UNICODE
    )
    FALLBACK_PATTERN = re.compile(r"""\w+|[^\w\s]|\s+""", re.UNICODE)

    def __init__(self):
        self.special_tokens = ["<pad>", "<bos>", "<eos>", "<unk>", "<sep>"]
        self.vocab: Dict[str, int] = {tok: idx for idx, tok in enumerate(self.special_tokens)}
        self.inv_vocab: Dict[int, str] = {idx: tok for tok, idx in self.vocab.items()}
        self.merges: List[Tuple[str, str]] = []
        self._ensure_byte_vocabulary()

    def _ensure_byte_vocabulary(self):
        """Ensures all standard individual ASCII/Unicode single characters exist in vocab."""
        for char in " abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?:;-_'\"/\\()[]{}@#$%^&*+=<>~`|\n\t":
            if char not in self.vocab:
                idx = len(self.vocab)
                self.vocab[char] = idx
                self.inv_vocab[idx] = char

    def _tokenize_text(self, text: str) -> List[str]:
        """Pre-tokenizes text into words/chunks using unicode regex."""
        try:
            tokens = self.FALLBACK_PATTERN.findall(text)
        except Exception:
            tokens = list(text)
        return tokens

    def train(self, corpus: List[str], target_vocab_size: int = 500, min_freq: int = 2):
        """
        Learns BPE merge rules iteratively from corpus until target_vocab_size is reached.
        """
        # Split corpus into character representations
        word_freqs: Dict[Tuple[str, ...], int] = {}
        for doc in corpus:
            raw_tokens = self._tokenize_text(doc)
            for tok in raw_tokens:
                # Add characters as individual initial units
                char_tuple = tuple(list(tok))
                word_freqs[char_tuple] = word_freqs.get(char_tuple, 0) + 1

        while len(self.vocab) < target_vocab_size:
            # Count adjacent pairs
            pair_counts: Dict[Tuple[str, str], int] = {}
            for word_tuple, freq in word_freqs.items():
                for i in range(len(word_tuple) - 1):
                    pair = (word_tuple[i], word_tuple[i + 1])
                    pair_counts[pair] = pair_counts.get(pair, 0) + freq

            if not pair_counts:
                break

            # Find most frequent pair
            best_pair, best_freq = max(pair_counts.items(), key=lambda x: x[1])
            if best_freq < min_freq:
                break

            merged_tok = "".join(best_pair)
            self.merges.append(best_pair)
            idx = len(self.vocab)
            self.vocab[merged_tok] = idx
            self.inv_vocab[idx] = merged_tok

            # Update word representations with the newly merged token
            new_word_freqs = {}
            for word_tuple, freq in word_freqs.items():
                new_word = []
                i = 0
                while i < len(word_tuple):
                    if i < len(word_tuple) - 1 and (word_tuple[i], word_tuple[i + 1]) == best_pair:
                        new_word.append(merged_tok)
                        i += 2
                    else:
                        new_word.append(word_tuple[i])
                        i += 1
                new_word_freqs[tuple(new_word)] = freq
            word_freqs = new_word_freqs

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = True) -> List[int]:
        """Encodes raw text into a list of token IDs using learned merge rules."""
        raw_tokens = self._tokenize_text(text)
        token_ids = []

        if add_bos:
            token_ids.append(self.vocab["<bos>"])

        for tok in raw_tokens:
            symbols = list(tok)
            # Apply merges in order
            for pair in self.merges:
                merged = "".join(pair)
                new_symbols = []
                i = 0
                while i < len(symbols):
                    if i < len(symbols) - 1 and symbols[i] == pair[0] and symbols[i + 1] == pair[1]:
                        new_symbols.append(merged)
                        i += 2
                    else:
                        new_symbols.append(symbols[i])
                        i += 1
                symbols = new_symbols

            for s in symbols:
                if s in self.vocab:
                    token_ids.append(self.vocab[s])
                else:
                    # Character by character fallback or <unk>
                    for char in s:
                        token_ids.append(self.vocab.get(char, self.vocab["<unk>"]))

        if add_eos:
            token_ids.append(self.vocab["<eos>"])

        return token_ids

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decodes token IDs back to a readable string."""
        chars = []
        for tid in token_ids:
            tok = self.inv_vocab.get(tid, "<unk>")
            if skip_special_tokens and tok in self.special_tokens:
                continue
            chars.append(tok)
        return "".join(chars)

    def analyze_compression(self, text: str) -> Dict[str, float]:
        """
        Computes foundation model tokenization metrics:
        - Total characters and UTF-8 bytes
        - Total generated tokens
        - Characters Per Token (CPT)
        - Bytes Per Token (BPT) - key for model efficiency
        - Compression Ratio vs raw bytes
        """
        utf8_bytes = len(text.encode("utf-8"))
        char_count = len(text)
        tokens = self.encode(text, add_bos=False, add_eos=False)
        token_count = max(len(tokens), 1)

        return {
            "char_count": char_count,
            "byte_count": utf8_bytes,
            "token_count": token_count,
            "chars_per_token": round(char_count / token_count, 3),
            "bytes_per_token": round(utf8_bytes / token_count, 3),
            "compression_ratio": round(utf8_bytes / (token_count * 2), 3)  # assuming int16 token storage
        }

    def save(self, filepath: str):
        data = {
            "special_tokens": self.special_tokens,
            "vocab": self.vocab,
            "merges": self.merges
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def load(self, filepath: str):
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.special_tokens = data["special_tokens"]
        self.vocab = data["vocab"]
        self.inv_vocab = {int(v): k for k, v in self.vocab.items()}
        self.merges = [tuple(p) for p in data["merges"]]
