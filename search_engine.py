"""search-engine — inverted index full-text search with BM25 ranking.

Zero-dependency implementation of the classic search-engine pipeline:

1. ``tokenize`` — unicode word extraction, lowercasing, stopword removal
   and light suffix stemming.
2. ``InvertedIndex`` — postings lists with term frequencies and token
   positions, persisted as JSON (no pickle, no third-party formats).
3. Ranking — Okapi BM25 with configurable ``k1`` / ``b`` parameters.
4. Query language — bare terms, ``"exact phrases"`` matched by positional
   intersection, and boolean ``AND`` / ``OR`` / ``NOT`` with parentheses.
   Precedence is ``NOT`` > ``AND`` > ``OR``; bare juxtaposition means
   ``AND``; keywords are case-insensitive.

CLI::

    python search_engine.py index <dir> [--index FILE] [--ext .txt]
    python search_engine.py query <expr> [--index FILE] [--top N]
"""

import argparse
import json
import math
import os
import re
import sys

__all__ = [
    "STOPWORDS",
    "QuerySyntaxError",
    "InvertedIndex",
    "build_parser",
    "main",
    "parse_query",
    "stem",
    "tokenize",
]

DEFAULT_INDEX = "search_index.json"

# Unicode-aware word finder: sequences of word characters that are not
# underscores (so CJK, accented Latin and digits all count as words).
_WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)

# English function words stripped before indexing (54 entries).
STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "can",
    "could", "did", "do", "for", "from", "had", "has", "have", "he",
    "her", "his", "i", "if", "in", "into", "is", "it", "no", "not",
    "of", "on", "or", "our", "should", "she", "such", "that", "the",
    "their", "then", "there", "these", "they", "this", "to", "us",
    "was", "we", "were", "will", "with", "would", "you", "your",
})

_VOWELS = "aeiou"


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

def _undouble(word: str) -> str:
    """Undo a doubled final consonant (``running`` -> ``runn`` -> ``run``)."""
    if (len(word) >= 3 and word[-1] == word[-2] and word[-1] not in _VOWELS):
        return word[:-1]
    return word


def stem(word: str) -> str:
    """Light suffix stemming (documented rules, applied in order).

    ``cats`` -> ``cat``, ``boxes`` -> ``box``, ``watches`` -> ``watch``,
    ``running`` -> ``run``, ``walked`` -> ``walk``,
    ``companies`` -> ``company``.  Words of three characters or fewer and
    words that match no rule are returned unchanged.
    """
    if len(word) <= 3:
        return word
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith(("ches", "shes", "xes", "zes", "ses")):
        return word[:-2]
    if word.endswith("ing") and len(word) > 5:
        return _undouble(word[:-3])
    if word.endswith("ed") and len(word) > 4:
        return _undouble(word[:-2])
    if word.endswith("s") and not word.endswith(("ss", "us")):
        return word[:-1]
    return word


def _token_spans(text):
    """Return ``[(term, position), ...]`` for *text*.

    Positions index the raw lowercased word stream *before* stopword
    removal, so phrase adjacency still reflects the original text: in
    ``"quick and brown"`` the words ``quick`` and ``brown`` sit at
    positions 0 and 2 and are therefore not adjacent.
    """
    spans = []
    for pos, word in enumerate(_WORD_RE.findall(text.lower())):
        if word in STOPWORDS:
            continue
        spans.append((stem(word), pos))
    return spans


def tokenize(text: str) -> list:
    """Lowercase *text*, drop stopwords, stem: the indexed term list."""
    return [term for term, _ in _token_spans(text)]


# ---------------------------------------------------------------------------
# Query language
# ---------------------------------------------------------------------------

class QuerySyntaxError(ValueError):
    """Raised for malformed query expressions."""


_QUERY_RE = re.compile(r'\s*(?:"([^"]*)"|(\()|(\))|([^\s()"]+))')
_KEYWORDS = frozenset({"AND", "OR", "NOT"})


def _query_tokens(expr: str) -> list:
    """Scan *expr* into ``("LP"|"RP"|"KW"|"TERM"|"PHRASE", value)`` tuples."""
    tokens = []
    pos = 0
    while pos < len(expr):
        m = _QUERY_RE.match(expr, pos)
        if m is None:
            rest = expr[pos:].strip()
            if not rest:
                break
            if expr[pos] == '"':
                raise QuerySyntaxError("unterminated quote in query")
            raise QuerySyntaxError("cannot tokenise %r" % expr[pos:])
        pos = m.end()
        phrase, lp, rp, chunk = m.groups()
        if phrase is not None:
            terms = []
            for i, word in enumerate(_WORD_RE.findall(phrase.lower())):
                if word in STOPWORDS:
                    continue
                terms.append((stem(word), i))
            tokens.append(("PHRASE", terms))
        elif lp is not None:
            tokens.append(("LP", "("))
        elif rp is not None:
            tokens.append(("RP", ")"))
        elif chunk.upper() in _KEYWORDS:
            tokens.append(("KW", chunk.upper()))
        else:
            for word in _WORD_RE.findall(chunk.lower()):
                if word in STOPWORDS:
                    continue
                tokens.append(("TERM", stem(word)))
    return tokens


class _QueryParser:
    """Recursive-descent parser: OR over AND over NOT over primaries."""

    def __init__(self, tokens):
        self.tokens = tokens
        self.i = 0

    def _peek(self):
        if self.i < len(self.tokens):
            return self.tokens[self.i]
        return (None, None)

    def _at(self, kind, value=None):
        k, v = self._peek()
        return k == kind and (value is None or v == value)

    def _starts_primary(self):
        k, v = self._peek()
        return k in ("TERM", "PHRASE", "LP") or (k == "KW" and v == "NOT")

    def parse(self):
        node = self._or()
        if self.i < len(self.tokens):
            kind, value = self._peek()
            if kind == "RP":
                raise QuerySyntaxError("unbalanced ')'")
            raise QuerySyntaxError("unexpected token %r" % (value,))
        return node

    def _or(self):
        node = self._and()
        while self._at("KW", "OR"):
            self.i += 1
            node = ("or", node, self._and())
        return node

    def _and(self):
        node = self._not()
        while True:
            if self._at("KW", "AND"):
                self.i += 1
                node = ("and", node, self._not())
            elif self._starts_primary():
                # bare juxtaposition: "cat dog" means "cat AND dog"
                node = ("and", node, self._not())
            else:
                break
        return node

    def _not(self):
        if self._at("KW", "NOT"):
            self.i += 1
            return ("not", self._not())
        return self._primary()

    def _primary(self):
        kind, value = self._peek()
        if kind == "LP":
            self.i += 1
            node = self._or()
            if not self._at("RP"):
                raise QuerySyntaxError("missing ')'")
            self.i += 1
            return node
        if kind == "TERM":
            self.i += 1
            return ("term", value)
        if kind == "PHRASE":
            self.i += 1
            return ("phrase", value)
        if kind is None:
            raise QuerySyntaxError("unexpected end of query")
        raise QuerySyntaxError("unexpected token %r" % (value,))


def parse_query(expr: str):
    """Parse *expr* into an AST, or ``None`` when it holds no query terms."""
    tokens = _query_tokens(expr)
    if not tokens:
        return None
    return _QueryParser(tokens).parse()


def _collect_terms(node, out=None):
    """Ordered, de-duplicated query terms (phrase words included)."""
    if out is None:
        out = []
    kind = node[0]
    if kind == "term":
        if node[1] not in out:
            out.append(node[1])
    elif kind == "phrase":
        for term, _ in node[1]:
            if term not in out:
                out.append(term)
    elif kind in ("and", "or"):
        _collect_terms(node[1], out)
        _collect_terms(node[2], out)
    elif kind == "not":
        _collect_terms(node[1], out)
    return out


# ---------------------------------------------------------------------------
# Inverted index
# ---------------------------------------------------------------------------

class InvertedIndex:
    """Postings lists, document lengths and BM25 retrieval.

    ``_postings`` maps ``term -> {doc_id: [tf, [positions]]}``;
    ``_doc_len`` maps ``doc_id -> number of indexed tokens``.
    """

    def __init__(self, k1: float = 1.2, b: float = 0.75):
        self.k1 = float(k1)
        self.b = float(b)
        self._postings = {}
        self._doc_len = {}

    # -- construction -----------------------------------------------------

    @property
    def documents(self) -> tuple:
        """Document ids in insertion order."""
        return tuple(self._doc_len)

    @property
    def doc_count(self) -> int:
        return len(self._doc_len)

    @property
    def avg_doc_length(self) -> float:
        if not self._doc_len:
            return 0.0
        return sum(self._doc_len.values()) / len(self._doc_len)

    def df(self, term: str) -> int:
        """Number of documents containing *term*."""
        return len(self._postings.get(term, {}))

    def idf(self, term: str) -> float:
        """BM25 inverse document frequency: ``ln(1 + (N-df+0.5)/(df+0.5))``."""
        n = len(self._doc_len)
        if n == 0:
            return 0.0
        df = self.df(term)
        return math.log(1.0 + (n - df + 0.5) / (df + 0.5))

    def add_document(self, doc_id: str, text: str) -> None:
        """Index *text* under *doc_id* (raises on duplicate ids)."""
        if doc_id in self._doc_len:
            raise ValueError("duplicate document id: %r" % (doc_id,))
        spans = _token_spans(text)
        self._doc_len[doc_id] = len(spans)
        by_term = {}
        for term, pos in spans:
            by_term.setdefault(term, []).append(pos)
        for term, positions in by_term.items():
            self._postings.setdefault(term, {})[doc_id] = [len(positions),
                                                           positions]

    # -- persistence ------------------------------------------------------

    def save(self, path: str) -> None:
        """Write the index as JSON (deterministic key order)."""
        payload = {
            "version": 1,
            "k1": self.k1,
            "b": self.b,
            # Explicit insertion order: sort_keys() below reorders the
            # doc_lens mapping on disk, so the original sequence must be
            # recorded for roundtrips to reproduce ``documents`` exactly.
            "doc_order": list(self._doc_len),
            "doc_lens": dict(self._doc_len),
            "postings": {
                term: {doc: [tf, positions]
                       for doc, (tf, positions) in docs.items()}
                for term, docs in self._postings.items()
            },
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, sort_keys=True, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "InvertedIndex":
        """Reload an index written by :meth:`save`."""
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        if payload.get("version") != 1:
            raise ValueError("unsupported index version: %r"
                             % (payload.get("version"),))
        index = cls(k1=payload["k1"], b=payload["b"])
        doc_lens = {doc: int(n)
                    for doc, n in payload["doc_lens"].items()}
        order = payload.get("doc_order")
        if order is not None:
            unknown = [doc for doc in order if doc not in doc_lens]
            if unknown:
                raise ValueError(
                    "doc_order references unknown documents: %r" % (unknown,))
            doc_lens = {doc: doc_lens[doc] for doc in order}
        index._doc_len = doc_lens
        index._postings = {
            term: {doc: [int(tf), [int(p) for p in positions]]
                   for doc, (tf, positions) in docs.items()}
            for term, docs in payload["postings"].items()
        }
        return index

    # -- retrieval --------------------------------------------------------

    def _phrase_docs(self, terms) -> set:
        """Documents where the phrase's words appear at consecutive positions."""
        if not terms:
            return set()
        candidates = None
        for term, _ in terms:
            docs = set(self._postings.get(term, {}))
            candidates = docs if candidates is None else candidates & docs
        if not candidates:
            return set()
        hits = set()
        for doc in candidates:
            positions = {term: set(self._postings[term][doc][1])
                         for term, _ in terms}
            for base in positions[terms[0][0]]:
                if all((base + offset) in positions[term]
                       for term, offset in terms):
                    hits.add(doc)
                    break
        return hits

    def _eval(self, node) -> set:
        kind = node[0]
        if kind == "term":
            return set(self._postings.get(node[1], {}))
        if kind == "phrase":
            return self._phrase_docs(node[1])
        if kind == "and":
            return self._eval(node[1]) & self._eval(node[2])
        if kind == "or":
            return self._eval(node[1]) | self._eval(node[2])
        if kind == "not":
            return set(self._doc_len) - self._eval(node[1])
        raise QuerySyntaxError("unknown query node %r" % (kind,))

    def _bm25_scores(self, terms) -> dict:
        n = len(self._doc_len)
        if n == 0:
            return {}
        avgdl = self.avg_doc_length or 1.0
        scores = {}
        for term in terms:
            postings = self._postings.get(term)
            if not postings:
                continue
            idf = self.idf(term)
            for doc, (tf, _positions) in postings.items():
                dl = self._doc_len[doc]
                denom = tf + self.k1 * (1.0 - self.b + self.b * dl / avgdl)
                scores[doc] = (scores.get(doc, 0.0)
                               + idf * tf * (self.k1 + 1.0) / denom)
        return scores

    def search(self, query: str, limit=None) -> list:
        """Evaluate *query*, returning ``[(score, doc_id), ...]`` best first.

        Matches are ranked with BM25 over every term in the expression;
        documents reached only through ``NOT`` score 0.0.  Ties break on
        document id so output is deterministic.
        """
        node = parse_query(query)
        if node is None:
            return []
        matched = self._eval(node)
        scores = self._bm25_scores(_collect_terms(node))
        results = [(scores.get(doc, 0.0), doc) for doc in matched]
        results.sort(key=lambda item: (-item[0], item[1]))
        if limit is not None:
            results = results[:limit]
        return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="search_engine",
        description="Inverted index full-text search with BM25 ranking.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_index = sub.add_parser("index", help="build the index from a directory")
    p_index.add_argument("dir", help="directory of text files to index")
    p_index.add_argument("--index", default=DEFAULT_INDEX,
                         help="index file to write (default: %(default)s)")
    p_index.add_argument("--ext", default=".txt",
                         help="file extension to index (default: %(default)s)")

    p_query = sub.add_parser("query", help="run a query against the index")
    p_query.add_argument("expr", help='terms, "phrases", AND/OR/NOT')
    p_query.add_argument("--index", default=DEFAULT_INDEX,
                         help="index file to read (default: %(default)s)")
    p_query.add_argument("--top", type=int, default=10,
                         help="maximum results (default: %(default)s)")
    return parser


def _cmd_index(args) -> int:
    if not os.path.isdir(args.dir):
        print("not a directory: %s" % args.dir, file=sys.stderr)
        return 1
    index = InvertedIndex()
    count = 0
    for root, dirs, files in os.walk(args.dir):
        dirs.sort()
        for name in sorted(files):
            if not name.lower().endswith(args.ext):
                continue
            full = os.path.join(root, name)
            rel = os.path.relpath(full, args.dir).replace(os.sep, "/")
            with open(full, "r", encoding="utf-8", errors="replace") as fh:
                index.add_document(rel, fh.read())
            count += 1
    index.save(args.index)
    print("indexed %d documents -> %s" % (count, args.index))
    return 0


def _cmd_query(args) -> int:
    if not os.path.exists(args.index):
        print("no index at %s (run 'index' first)" % args.index,
              file=sys.stderr)
        return 1
    index = InvertedIndex.load(args.index)
    try:
        results = index.search(args.expr, limit=args.top)
    except QuerySyntaxError as exc:
        print("bad query: %s" % exc, file=sys.stderr)
        return 1
    if not results:
        print("no results")
        return 0
    for score, doc in results:
        print("%.4f\t%s" % (score, doc))
    return 0


def main(argv=None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:  # --help, missing/bad subcommand
        return 0 if exc.code is None else exc.code
    if args.command == "index":
        return _cmd_index(args)
    return _cmd_query(args)


if __name__ == "__main__":
    sys.exit(main())
