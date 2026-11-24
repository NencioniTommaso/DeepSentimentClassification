class Tokenizer:
    def tokenize(self, text):
        ...

class WhitespaceTokenizer(Tokenizer):
    def tokenize(self, text):
        return text.split()  # Simple whitespace tokenization