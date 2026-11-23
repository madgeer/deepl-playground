import json
import re
import os
from collections import Counter
from typing import List, Dict, Optional


class Tokenizer:
    """
    Tokenizer level kata (word-level) sederhana untuk NLP dengan PyTorch.

    Fitur:
        - Build vocabulary dari daftar teks
        - Special tokens: [PAD], [UNK], [CLS], [SEP], [MASK]
        - Encode teks -> daftar token ID
        - Decode token ID -> teks
        - Padding dan truncation ke panjang fixed
        - Simpan dan muat vocabulary ke/from file JSON

    Parameters
    ----------
    vocab : dict, optional
        Mapping token -> ID. Jika None, kosong dan harus dibangun dengan fit().
    special_tokens : dict, optional
        Mapping nama special token -> string token.
        Default: {'pad': '[PAD]', 'unk': '[UNK]', 'cls': '[CLS]',
                  'sep': '[SEP]', 'mask': '[MASK]'}
    """

    def __init__(self,
                 vocab: Optional[Dict[str, int]] = None,
                 special_tokens: Optional[Dict[str, str]] = None):
        if special_tokens is None:
            special_tokens = {
                'pad': '[PAD]',
                'unk': '[UNK]',
                'cls': '[CLS]',
                'sep': '[SEP]',
                'mask': '[MASK]',
            }
        self.special_tokens = special_tokens

        # Inisialisasi vocab dengan special tokens
        self.vocab: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}

        if vocab is not None:
            self.vocab = dict(vocab)
            self.id_to_token = {v: k for k, v in self.vocab.items()}
        else:
            self._init_special_tokens()

        self._next_id = max(self.vocab.values()) + 1 if self.vocab else 0

    def _init_special_tokens(self) -> None:
        """Masukkan special tokens ke vocab di awal."""
        for name, token in self.special_tokens.items():
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab)
                self.id_to_token[self.vocab[token]] = token

    @property
    def pad_token_id(self) -> int:
        return self.vocab[self.special_tokens['pad']]

    @property
    def unk_token_id(self) -> int:
        return self.vocab[self.special_tokens['unk']]

    @property
    def cls_token_id(self) -> int:
        return self.vocab[self.special_tokens['cls']]

    @property
    def sep_token_id(self) -> int:
        return self.vocab[self.special_tokens['sep']]

    @property
    def mask_token_id(self) -> int:
        return self.vocab[self.special_tokens['mask']]

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def _tokenize(self, text: str) -> List[str]:
        """
        Pisahkan teks menjadi daftar token (kata) sederhana.
        """
        # Lowercase, pisahkan spasi dan tanda baca sederhana
        text = text.lower()
        text = re.sub(r"([.,!?;:\"'()\[\]{}])", r" \1 ", text)
        tokens = text.split()
        return tokens

    def fit(self, texts: List[str], min_freq: int = 1,
            max_vocab: Optional[int] = None) -> None:
        """
        Bangun vocabulary dari daftar teks.

        Parameters
        ----------
        texts : list of str
            Daftar teks untuk membangun vocab.
        min_freq : int
            Frekuensi minimum token untuk masuk ke vocab.
        max_vocab : int, optional
            Ukuran maksimal vocab (termasuk special tokens).
        """
        counter = Counter()
        for text in texts:
            tokens = self._tokenize(text)
            counter.update(tokens)

        # Urutkan berdasarkan frekuensi menurun
        sorted_tokens = [tok for tok, freq in counter.most_common()
                         if freq >= min_freq]

        if max_vocab is not None:
            # Reserve tempat untuk special tokens
            reserved = len(self.special_tokens)
            sorted_tokens = sorted_tokens[:max_vocab - reserved]

        for token in sorted_tokens:
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab)
                self.id_to_token[self.vocab[token]] = token

    def encode(self,
               text: str,
               max_length: Optional[int] = None,
               add_special_tokens: bool = True,
               pad: bool = True,
               truncation: bool = True) -> List[int]:
        """
        Ubah teks menjadi daftar token ID.

        Parameters
        ----------
        text : str
            Teks input.
        max_length : int, optional
            Panjang output setelah padding/truncation.
        add_special_tokens : bool
            Tambahkan [CLS] di awal dan [SEP] di akhir.
        pad : bool
            Pad ke max_length jika diperlukan.
        truncation : bool
            Potong ke max_length jika melebihi batas.

        Returns
        -------
        list of int
            Daftar token ID.
        """
        tokens = self._tokenize(text)

        if add_special_tokens:
            tokens = [self.special_tokens['cls']] + tokens + \
                [self.special_tokens['sep']]

        ids = [self.vocab.get(tok, self.unk_token_id) for tok in tokens]

        if truncation and max_length is not None:
            ids = ids[:max_length]

        if pad and max_length is not None:
            pad_id = self.pad_token_id
            ids = ids + [pad_id] * (max_length - len(ids))

        return ids

    def decode(self, ids: List[int],
               skip_special_tokens: bool = False) -> str:
        """
        Ubah daftar token ID kembali menjadi teks.

        Parameters
        ----------
        ids : list of int
            Daftar token ID.
        skip_special_tokens : bool
            Lewati special tokens saat decode.

        Returns
        -------
        str
            Teks hasil decode.
        """
        tokens = []
        for i in ids:
            token = self.id_to_token.get(i, self.special_tokens['unk'])
            if skip_special_tokens and token in self.special_tokens.values():
                continue
            tokens.append(token)

        # Gabungkan kembali, hapus spasi di depan tanda baca sederhana
        text = " ".join(tokens)
        text = re.sub(r"\s+([.,!?;:\"'\)\]\}])", r"\1", text)
        text = re.sub(r"\(|\{|\[", lambda m: m.group(0), text)
        return text.strip()

    def save(self, path: str) -> None:
        """
        Simpan vocabulary dan special tokens ke file JSON.
        """
        data = {
            "vocab": self.vocab,
            "special_tokens": self.special_tokens,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Vocabulary disimpan ke {path} ({self.vocab_size} token)")

    def load(self, path: str) -> None:
        """
        Muat vocabulary dan special tokens dari file JSON.
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.special_tokens = data.get("special_tokens", self.special_tokens)
        self.vocab = data.get("vocab", {})
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        self._next_id = max(self.vocab.values()) + 1 if self.vocab else 0
        print(f"Vocabulary dimuat dari {path} ({self.vocab_size} token)")


def create_tokenizer() -> Tokenizer:
    """
    Fungsi pembantu untuk membuat instance Tokenizer baru.

    Returns
    -------
    Tokenizer
        Instance Tokenizer kosong (belum ada vocab kata asli).
    """
    return Tokenizer()


def load_tokenizer(path: str) -> Tokenizer:
    """
    Muat Tokenizer dari file JSON.

    Returns
    -------
    Tokenizer
        Instance Tokenizer dengan vocab yang sudah dimuat.
    """
    tokenizer = Tokenizer()
    tokenizer.load(path)
    return tokenizer
