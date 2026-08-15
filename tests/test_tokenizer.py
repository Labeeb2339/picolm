"""Tokenizer tests: char round-trip, BPE losslessness and compression."""

from picolm.tokenizer import BPETokenizer, CharTokenizer


def test_char_tokenizer_roundtrip():
    text = "hello world, this is a test!"
    tok = CharTokenizer.fit(text)
    ids = tok.encode(text)
    assert tok.decode(ids) == text
    assert tok.vocab_size == len(set(text))


def test_char_tokenizer_save_load(tmp_path):
    tok = CharTokenizer.fit("abcdefg")
    tok.save(tmp_path / "tok.json")
    loaded = CharTokenizer.load(tmp_path / "tok.json")
    assert loaded.chars == tok.chars


def test_bpe_roundtrip_lossless():
    text = "the quick brown fox jumps over the lazy dog. " * 50
    tok = BPETokenizer()
    tok.train(text, vocab_size=300)
    assert tok.vocab_size == 300
    ids = tok.encode(text)
    assert tok.decode(ids) == text  # lossless by construction


def test_bpe_compresses():
    text = "abracadabra abracadabra abracadabra " * 20
    tok = BPETokenizer()
    tok.train(text, vocab_size=280)
    ids = tok.encode(text)
    assert len(ids) < len(text.encode("utf-8"))


def test_bpe_handles_unicode():
    text = "héllo wörld — café ☕ " * 15
    tok = BPETokenizer()
    tok.train(text, vocab_size=280)
    assert tok.decode(tok.encode(text)) == text


def test_bpe_save_load(tmp_path):
    text = "some training text for the tokenizer. " * 40
    tok = BPETokenizer()
    tok.train(text, vocab_size=270)
    tok.save(tmp_path / "bpe.json")
    loaded = BPETokenizer.load(tmp_path / "bpe.json")
    assert loaded.merges == tok.merges
    assert loaded.decode(loaded.encode(text)) == text
