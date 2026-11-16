# search-engine

A dependency-free full-text search engine: inverted index, BM25 ranking,
phrase queries, and boolean expressions — in one Python file, standard
library only.

## Why it's special

- **Positional postings.** Every term stores per-document term frequency
  *and* exact token positions, which is what makes real phrase queries
  (`"python services"`) possible instead of the usual bag-of-words fake.
- **A real query language.** `AND` / `OR` / `NOT` with correct precedence,
  parentheses, and quoted phrases, parsed into an AST — not string
  splitting.
- **Honest BM25.** `idf = ln(1 + (N - df + 0.5)/(df + 0.5))`, document
  length normalization, and deterministic tie-breaking on document id.
  Shorter documents win tf/idf ties, exactly as the formula intends.
- **Self-describing index file.** One JSON document with an explicit
  `doc_order`, so save → load reproduces the in-memory index exactly,
  byte-for-byte deterministic across runs.

## Architecture

```
tokenize (spans + stopwords + suffix-stemming)
    → postings { term: { doc_id: [tf, positions] } }
    → query parser → AST → boolean evaluation over doc-id sets
    → BM25 scoring over matched docs → sorted results
```

Index files carry `version`, `k1`, `b`, `doc_order`, `doc_lens`, and
`postings`, so parameters survive roundtrips.

## Usage

```console
$ python search_engine.py index ./docs --index idx.json
indexed 2 documents -> idx.json

$ python search_engine.py query "python AND analytics" --index idx.json
0.9061	guide.txt

$ python search_engine.py query '"python services"' --index idx.json
0.9172	ops.txt
```

Query forms: bare terms, `"quoted phrases"`, `AND`, `OR`, `NOT`,
parentheses. Options: `--ext` (index extension filter), `--top N`.

## Tests

```console
$ python -m unittest discover -s tests -v
```

35 tests cover the tokenizer, BM25 math (monotonicity, idf formula),
phrase adjacency and ordering, boolean precedence, persistence roundtrips,
deterministic saves, and CLI end-to-end behavior.

## License

MIT — see `LICENSE`.
