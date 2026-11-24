class Vocabulary:
    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"
    PAD_ID = 0
    UNK_ID = 1

    def __init__(self, tokenizer, min_freq=2):
        self.tokenizer = tokenizer
        self.min_freq = min_freq
        self.token_to_id = {}
        self.id_to_token = {0: self.PAD_TOKEN, 1: self.UNK_TOKEN}
        self.token_freq = {}
        self._is_built = False

    def build(self, texts):
        if self._is_built:
            raise ValueError("Vocabulary has already been built.")
        
        for text in texts:
            tokens = self.tokenizer.tokenize(text)  # Use the provided tokenizer
            for token in tokens:
                if token in self.token_freq:
                    self.token_freq[token] += 1
                else:
                    self.token_freq[token] = 1

        # Filter tokens by min frequency and assign IDs
        for token, freq in self.token_freq.items():
            if freq >= self.min_freq:
                token_id = len(self.id_to_token)
                self.token_to_id[token] = token_id
                self.id_to_token[token_id] = token

        self._is_built = True

        return self         # It allows chaining: Vocabulary().build(texts).encode(text)

    def encode(self, text):     # Takes a string and encodes it using the vocabulary, returning a list of token IDs
        if not self._is_built:
            raise ValueError("Vocabulary has not been built yet.")

        tokens = self.tokenizer.tokenize(text)
        return [self.token_to_id.get(token, self.UNK_ID) for token in tokens]

    def decode(self, token_ids, skip_special_tokens=False):    # Takes a list of token IDs and decodes it back to a string using the vocabulary, skips special tokens if specified
        if not self._is_built:
                    raise ValueError("Vocabulary has not been built yet.")
        
        tokens = [
            self.id_to_token.get(token_id, self.UNK_TOKEN)
            for token_id in token_ids
        ]

        if skip_special_tokens:
            tokens = [
                token for token in tokens
                if token not in (self.PAD_TOKEN, self.UNK_TOKEN)
            ]

        return " ".join(tokens)

    def __len__(self):
        return len(self.id_to_token)