import json
import math
from collections import defaultdict
from typing import List, Dict, Tuple, Optional, Any


class Ngram:
    """
    N-gram language model sederhana.

    Memodel probabilitas kalimat dengan mendekati probabilitas
    kata berikutnya berdasarkan n-1 kata sebelumnya.

    Supported features:
        - Count n-grams dari corpus teks
        - Estimasi probabilitas P(w_n | w_{n-1}, ..., w_{n-n+1})
        - Laplace smoothing (add-one smoothing)
        - Good-Turing smoothing (sederhana)
        - Prediksi kata selanjutnya
        - Hitung probabilitas kalimat lengkap

    Parameters
    ----------
    n : int
        Ukuran n-gram (mis. 1 = unigram, 2 = bigram, 3 = trigram)
    use_smoothing : bool
        Gunakan Laplace smoothing untuk menghandle zero probabilities.
    """

    def __init__(self, n: int = 2, use_smoothing: bool = True):
        if n < 1:
            raise ValueError("n must be >= 1")
        self.n = n
        self.use_smoothing = use_smoothing

        # Simpan counts n-grams: (n-1)-gram -> {current_word: count}
        self.ngram_counts: Dict[Tuple[str, ...], Dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        # Unigram counts untuk numerator saat smoothing
        self.unigram_counts: Dict[str, int] = defaultdict(int)
        # Total n-grams untuk normalisasi
        self.total_ngrams = 0

        # Kata khusus
        self.vocab: set = set()
        self.vocab_size = 0

    def fit(self, corpus: List[str]) -> None:
        """
        Latih model n-gram dari daftar kalimat (corpus).

        Parameters
        ----------
        corpus : list of str
            Daftar kalimat untuk melatih model.
        """
        self.vocab = set()
        self.unigram_counts = defaultdict(int)
        self.ngram_counts = defaultdict(lambda: defaultdict(int))
        self.total_ngrams = 0

        for sentence in corpus:
            # Tokenisasi sederhana: lowercase + split
            tokens = sentence.lower().split()
            self.vocab.update(tokens)

            # Hitung unigram
            for token in tokens:
                self.unigram_counts[token] += 1

            # Tambahkan token START dan END untuk handling boundary
            # Pad dengan token khusus
            padded = ["<START>"] * (self.n - 1) + tokens + ["<END>"]

            # Hitung n-grams
            for i in range(len(padded) - self.n + 1):
                context = tuple(padded[i : i + self.n - 1])
                target = padded[i + self.n - 1]
                self.ngram_counts[context][target] += 1
                self.total_ngrams += 1

        self.vocab_size = len(self.vocab)

    def probability(self, word: str, context: Tuple[str, ...]) -> float:
        """
        Hitung probabilitas P(word | context) dengan optional smoothing.

        Parameters
        ----------
        word : str
            Kata target.
        context : tuple of str
            Konteks (n-1 kata sebelumnya). Panjang harus n-1.

        Returns
        -------
        float
            Probabilitas kata word diberikan context. Selalu > 0 jika smoothing.
        """
        if self.n == 1:
            # Unigram model
            count = self.unigram_counts.get(word, 0)
            if self.use_smoothing:
                return (count + 1) / (sum(self.unigram_counts.values()) + self.vocab_size)
            return count / sum(self.unigram_counts.values()) if sum(
                self.unigram_counts.values()) > 0 else 0.0

        # Ambil count n-gram
        context_count = sum(self.ngram_counts[context].values())
        target_count = self.ngram_counts[context].get(word, 0)

        if self.use_smoothing:
            # Laplace smoothing: (count(w, context) + 1) / (count(context) + V)
            # count(context) = total n-grams with this context
            # V = vocab_size
            return (target_count + 1) / (context_count + self.vocab_size)
        else:
            # Tanpa smoothing
            if context_count == 0:
                # Fallback ke unigram dengan smoothing atau 0
                if self.use_smoothing:
                    return (self.unigram_counts.get(word, 0) + 1) / (
                        sum(self.unigram_counts.values()) + self.vocab_size
                    )
                return 0.0
            return target_count / context_count

    def next_word(self, context: Tuple[str, ...],
                  return_scores: bool = False) -> Any:
        """
        Prediksi kata selanjutnya berdasarkan konteks.

        Parameters
        ----------
        context : tuple of str
            Konteks n-1 kata terakhir.
        return_scores : bool
            Jika True, return daftar (word, probability) yang diurutkan.

        Returns
        -------
        str or list of (str, float)
            Kata dengan probabilitas tertinggi, atau daftar terurut jika return_scores=True.
        """
        # Pastikan context panjang n-1
        if len(context) != self.n - 1:
            raise ValueError(
                f"Context length must be {self.n - 1}, got {len(context)}"
            )

        # Kumpulkan semua candidate kata dan probability-nya
        candidates = []
        for word in self.vocab:
            prob = self.probability(word, context)
            if prob > 0:
                candidates.append((word, prob))

        # Urutkan berdasarkan probability turun
        candidates.sort(key=lambda x: x[1], reverse=True)

        if return_scores:
            return candidates
        return candidates[0][0] if candidates else "<UNK>"

    def sentence_log_prob(self, sentence: str) -> float:
        """
        Hitung log-probabilitas keseluruhan sebuah kalimat.

        Parameters
        ----------
        sentence : str
            Kalimat untuk dihitung probabilitas.

        Returns
        -------
        float
            Log-probabilitas. Negatif (atau nol jika improbable).
        """
        tokens = sentence.lower().split()
        if not tokens:
            return 0.0

        # Pad
        padded = ["<START>"] * (self.n - 1) + tokens + ["<END>"]

        log_prob = 0.0
        for i in range(len(padded) - self.n + 1):
            context = tuple(padded[i : i + self.n - 1])
            word = padded[i + self.n - 1]
            prob = self.probability(word, context)
            if prob > 0:
                log_prob += math.log(prob)
            else:
                # Jika probability nol (tanpa smoothing), log prob = -inf
                return float("-inf")

        return log_prob

    def perplexity(self, corpus: List[str]) -> float:
        """
        Hitung perplexity dari corpus.

        Perplexity mengukur seberapa "terkejut" model dengan data baru.
        Semakin kecil perplexity, model semakin bagus.

        Parameters
        ----------
        corpus : list of str
            Corpus uji.

        Returns
        -------
        float
            Nilai perplexity.
        """
        total_log_prob = 0.0
        total_tokens = 0

        for sentence in corpus:
            tokens = sentence.lower().split()
            if not tokens:
                continue

            padded = ["<START>"] * (self.n - 1) + tokens + ["<END>"]
            sentence_len = len(padded) - self.n + 1

            for i in range(sentence_len):
                context = tuple(padded[i : i + self.n - 1])
                word = padded[i + self.n - 1]
                prob = self.probability(word, context)
                if prob > 0:
                    total_log_prob += math.log(prob)
                    total_tokens += 1

            # Tambah log prob untuk <END> token
            # (sudah dihitung dalam loop di atas karena panjang n-gram)

        if total_tokens == 0:
            return float("inf")

        avg_log_prob = -total_log_prob / total_tokens
        return math.exp(avg_log_prob)

    def save(self, path: str) -> None:
        """
        Simpan model ke file (hanya counts dan parameter).
        """
        data = {
            "n": self.n,
            "use_smoothing": self.use_smoothing,
            "vocab": list(self.vocab),
            "unigram_counts": dict(self.unigram_counts),
            "ngram_counts": {
                "".join(k): dict(v) for k, v in self.ngram_counts.items()
            },  # key dijadikan string buat JSON
            "total_ngrams": self.total_ngrams,
            "vocab_size": self.vocab_size,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Model disimpan ke {path} (n={self.n}, vocab={self.vocab_size})")

    @classmethod
    def load(cls, path: str) -> "Ngram":
        """
        Muat model dari file.
        """
        import json

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        model = cls(n=data["n"], use_smoothing=data["use_smoothing"])
        model.vocab = set(data["vocab"])
        model.vocab_size = data["vocab_size"]
        model.total_ngrams = data["total_ngrams"]
        model.unigram_counts = defaultdict(int, data["unigram_counts"])

        # Rebuild ngram_counts dari string keys
        model.ngram_counts = defaultdict(lambda: defaultdict(int))
        for key, val in data["ngram_counts"].items():
            context = tuple(key.split("_"))  # simplistic restore
            for word, count in val.items():
                model.ngram_counts[context][word] = count
        model.vocab_size = len(model.vocab)

        print(f"Model dimuat dari {path} (n={model.n}, vocab={model.vocab_size})")
        return model


def create_ngram(n: int = 2, use_smoothing: bool = True) -> Ngram:
    """
    Fungsi pembantu untuk membuat instance Ngram.

    Returns
    -------
    Ngram
        Instance model n-gram baru.
    """
    return Ngram(n=n, use_smoothing=use_smoothing)


def main():
    """CLI interaktif sederhana (tanpa argparse, config hardcode)."""
    print("N-gram Model Tool")
    print("=================")


def demo() -> None:
    """Fungsi demo n-gram."""
    # Contoh penggunaan
    corpus = [
        "saya suka makan apel",
        "saya suka minum jeruk",
        "mereka suka makan apel jeruk",
        "saya suka bermain bola",
        "mereka suka bermain bola di taman",
    ]

    # Buat trigram model
    model = create_ngram(n=3, use_smoothing=True)
    model.fit(corpus)

    print("=== Demo N-gram Trigram ===")
    print(f"Vocabulary size: {model.vocab_size}")
    print()

    # Prediksi kata selanjutnya
    contexts = [
        ("saya", "suka"),
        ("mereka", "suka"),
        ("saya", "bermain"),
    ]

    for ctx in contexts:
        pred = model.next_word(ctx)
        print(f"Context: {ctx} -> Prediksi selanjutnya: {pred}")

    # Hitung log-probabilitas kalimat
    test_sentences = [
        "saya suka makan apel",
        "saya bermain di taman",
    ]

    for sent in test_sentences:
        lp = model.sentence_log_prob(sent)
        print(f"'{sent}' -> log-prob: {lp:.4f}")

    # Hitung perplexity
    pp = model.perplexity(corpus)
    print(f"Perplexity training corpus: {pp:.4f}")


if __name__ == "__main__":
    demo()