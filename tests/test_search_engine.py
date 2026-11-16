"""Tests for the search-engine project (entry 010)."""

import contextlib
import io
import os
import shutil
import tempfile
import unittest

import search_engine
from search_engine import (
    InvertedIndex,
    QuerySyntaxError,
    main,
    parse_query,
    stem,
    tokenize,
)


def build_index(documents, **kwargs):
    index = InvertedIndex(**kwargs)
    for doc_id, text in documents:
        index.add_document(doc_id, text)
    return index


def score_map(results):
    """Turn ``[(score, doc), ...]`` into ``{doc: score}``."""
    return {doc: score for score, doc in results}


# A hand-built corpus with a known winner: "python" appears in three
# documents with different frequencies, and "kubernetes" only in one.
CORPUS = [
    ("guide.txt", "python python python data structures and algorithms"),
    ("intro.txt", "python data data analysis with python basics"),
    ("ops.txt", "kubernetes deployment guides for containers"),
    ("misc.txt", "cooking recipes and gardening tips"),
]


class TokenizerTests(unittest.TestCase):
    def test_punctuation_is_split(self):
        self.assertEqual(tokenize("hello, world! (test); done."),
                         ["hello", "world", "test", "done"])

    def test_unicode_words_preserved(self):
        self.assertEqual(tokenize("Café naïve 東京"), ["café", "naïve", "東京"])

    def test_lowercase_and_stopwords(self):
        self.assertEqual(tokenize("The Quick Brown Fox and the lazy Dog"),
                         ["quick", "brown", "fox", "lazy", "dog"])

    def test_empty_and_symbol_only_documents(self):
        self.assertEqual(tokenize(""), [])
        self.assertEqual(tokenize("  \n\t  "), [])
        self.assertEqual(tokenize("!!! ... ---"), [])

    def test_numbers_are_tokens(self):
        self.assertEqual(tokenize("top 10 tips"), ["top", "10", "tip"])


class StemmingTests(unittest.TestCase):
    def test_suffix_rules(self):
        self.assertEqual(stem("cats"), "cat")
        self.assertEqual(stem("running"), "run")
        self.assertEqual(stem("walked"), "walk")
        self.assertEqual(stem("companies"), "company")
        self.assertEqual(stem("boxes"), "box")

    def test_short_and_unruled_words_unchanged(self):
        self.assertEqual(stem("dog"), "dog")
        self.assertEqual(stem("class"), "class")
        self.assertEqual(stem("data"), "data")

    def test_stemming_unifies_query_and_index(self):
        index = build_index([("a", "the runner runs racing"), ("b", "cats")])
        self.assertEqual([d for _, d in index.search("run")], ["a"])
        self.assertEqual([d for _, d in index.search("cat")], ["b"])


class RankingTests(unittest.TestCase):
    def setUp(self):
        self.index = build_index(CORPUS)

    def test_known_winner_ranks_first(self):
        results = self.index.search("python")
        self.assertEqual([doc for _, doc in results],
                         ["guide.txt", "intro.txt"])

    def test_bm25_increases_with_term_frequency(self):
        # Same term, two documents: the higher-tf document scores higher.
        index = build_index([("a", "python python python"),
                             ("b", "python")])
        scores = score_map(index.search("python"))
        self.assertGreater(scores["a"], scores["b"])

    def test_score_monotonic_in_tf(self):
        # tf = 1..4, all else equal: scores must strictly increase.
        docs = [("d%d" % i, "python " * (i + 1)) for i in range(4)]
        index = build_index(docs)
        scores = score_map(index.search("python"))
        values = [scores["d%d" % i] for i in range(4)]
        self.assertEqual(values, sorted(values))
        for earlier, later in zip(values, values[1:]):
            self.assertLess(earlier, later)

    def test_rarer_term_scores_higher(self):
        # Same document, same tf: the idf of a rare term dominates.
        index = build_index([
            ("a", "zebra common"),
            ("b", "common words here"),
            ("c", "more common words"),
        ])
        scores = score_map(index.search("zebra"))
        common_scores = score_map(index.search("common"))
        self.assertGreater(scores["a"], common_scores["a"])

    def test_idf_formula_and_monotonicity(self):
        index = build_index(CORPUS)
        self.assertEqual(index.df("python"), 2)
        # "kubernetes" is stemmed at index time; ask with the same stem.
        self.assertEqual(index.df(stem("kubernetes")), 1)
        self.assertGreater(index.idf("kubernetes"), index.idf("python"))
        # Any term present in the corpus has a positive idf.
        self.assertGreater(index.idf(stem("cooking")), 0)

    def test_unknown_term_and_empty_query(self):
        self.assertEqual(self.index.search("quantum"), [])
        self.assertEqual(self.index.search(""), [])


class PhraseTests(unittest.TestCase):
    def setUp(self):
        self.index = build_index([
            ("adjacent.txt", "the quick brown fox jumps"),
            ("split.txt", "the quick red brown fox jumps"),
            ("rev.txt", "brown quick elsewhere"),
        ])

    def test_phrase_matches_adjacent(self):
        self.assertEqual([d for _, d in self.index.search('"quick brown"')],
                         ["adjacent.txt"])

    def test_phrase_excludes_non_adjacent(self):
        # "split.txt" contains both words but not next to each other.
        self.assertEqual([d for _, d in self.index.search('"quick brown"')],
                         ["adjacent.txt"])
        self.assertNotIn("split.txt",
                         [d for _, d in self.index.search('"quick brown"')])

    def test_phrase_order_matters(self):
        self.assertEqual([d for _, d in self.index.search('"brown quick"')],
                         ["rev.txt"])

    def test_phrase_combines_with_boolean(self):
        results = self.index.search('"quick brown" AND NOT red')
        self.assertEqual([d for _, d in results], ["adjacent.txt"])


class BooleanTests(unittest.TestCase):
    def setUp(self):
        self.index = build_index([
            ("red.txt", "red apples"),
            ("blue.txt", "blue sky"),
            ("green.txt", "green apples sky"),
        ])

    def docs(self, query):
        return {doc for _, doc in self.index.search(query)}

    def test_and_intersection(self):
        self.assertEqual(self.docs("apples AND sky"), {"green.txt"})
        self.assertEqual(self.docs("apples sky"), {"green.txt"})  # implicit

    def test_or_union(self):
        self.assertEqual(self.docs("red OR blue"), {"red.txt", "blue.txt"})

    def test_not_complement(self):
        # Only blue.txt lacks "apples" (red.txt and green.txt have it).
        self.assertEqual(self.docs("NOT apples"), {"blue.txt"})

    def test_precedence_not_and_or(self):
        # NOT binds tightest: NOT red OR blue == (NOT red) OR blue
        self.assertEqual(self.docs("NOT red OR blue"),
                         {"blue.txt", "green.txt"})

    def test_parentheses_override_precedence(self):
        self.assertEqual(self.docs("NOT (red OR blue)"), {"green.txt"})

    def test_unknown_query_raises(self):
        with self.assertRaises(QuerySyntaxError):
            parse_query("apples AND")
        with self.assertRaises(QuerySyntaxError):
            parse_query("(apples")
        with self.assertRaises(QuerySyntaxError):
            parse_query('apples "unterminated')


class PersistenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="search-engine-test-")
        self.path = os.path.join(self.tmp, "idx.json")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_roundtrip_preserves_results(self):
        index = build_index(CORPUS)
        index.save(self.path)
        loaded = InvertedIndex.load(self.path)
        self.assertEqual(loaded.search("python"), index.search("python"))
        self.assertEqual(loaded.search('"python data"'),
                         index.search('"python data"'))
        self.assertEqual(loaded.search("python OR cooking"),
                         index.search("python OR cooking"))
        self.assertEqual(loaded.documents, index.documents)

    def test_roundtrip_preserves_parameters_and_positions(self):
        index = build_index(CORPUS, k1=2.0, b=0.5)
        index.save(self.path)
        loaded = InvertedIndex.load(self.path)
        self.assertEqual(loaded.k1, 2.0)
        self.assertEqual(loaded.b, 0.5)
        self.assertEqual(loaded._postings, index._postings)

    def test_saved_file_is_deterministic(self):
        index = build_index(CORPUS)
        index.save(self.path)
        with open(self.path, "rb") as fh:
            first = fh.read()
        index.save(self.path)
        with open(self.path, "rb") as fh:
            second = fh.read()
        self.assertEqual(first, second)


class CliTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="search-engine-cli-")
        self.corpus_dir = os.path.join(self.tmp, "docs")
        os.makedirs(self.corpus_dir)
        with open(os.path.join(self.corpus_dir, "a.txt"), "w",
                  encoding="utf-8") as fh:
            fh.write("python python data analysis")
        with open(os.path.join(self.corpus_dir, "b.txt"), "w",
                  encoding="utf-8") as fh:
            fh.write("rust systems programming")
        with open(os.path.join(self.corpus_dir, "skip.log"), "w",
                  encoding="utf-8") as fh:
            fh.write("python ignored because of the extension")
        self.index_path = os.path.join(self.tmp, "idx.json")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def run_main(self, argv):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = main(argv)
        return code, out.getvalue(), err.getvalue()

    def test_index_then_query_end_to_end(self):
        code, out, err = self.run_main(
            ["index", self.corpus_dir, "--index", self.index_path])
        self.assertEqual(code, 0)
        self.assertIn("indexed 2 documents", out)
        self.assertTrue(os.path.exists(self.index_path))

        code, out, _ = self.run_main(
            ["query", "python", "--index", self.index_path])
        self.assertEqual(code, 0)
        lines = out.rstrip("\n").split("\n")
        self.assertEqual(len(lines), 1)
        score, doc = lines[0].split("\t")
        self.assertEqual(doc, "a.txt")
        self.assertGreater(float(score), 0.0)

    def test_query_ranking_prints_best_first(self):
        self.run_main(["index", self.corpus_dir, "--index", self.index_path])
        code, out, _ = self.run_main(
            ["query", "data OR systems", "--index", self.index_path])
        self.assertEqual(code, 0)
        lines = out.rstrip("\n").split("\n")
        docs = [line.split("\t")[1] for line in lines]
        scores = [float(line.split("\t")[0]) for line in lines]
        # Both docs match exactly one term with tf=1 and identical idf, so
        # BM25's document-length normalization breaks the tie: b.txt holds
        # 3 tokens against a.txt's 4, so the shorter document ranks first.
        self.assertEqual(docs, ["b.txt", "a.txt"])
        self.assertGreater(scores[0], scores[1])

    def test_empty_results_reported(self):
        self.run_main(["index", self.corpus_dir, "--index", self.index_path])
        code, out, _ = self.run_main(
            ["query", "quantum", "--index", self.index_path])
        self.assertEqual(code, 0)
        self.assertEqual(out.strip(), "no results")

    def test_missing_index_fails(self):
        code, _, err = self.run_main(
            ["query", "python", "--index", os.path.join(self.tmp, "nope.json")])
        self.assertNotEqual(code, 0)
        self.assertIn("no index", err)

    def test_bad_directory_fails(self):
        code, _, err = self.run_main(
            ["index", os.path.join(self.tmp, "ghost"), "--index",
             self.index_path])
        self.assertNotEqual(code, 0)
        self.assertIn("not a directory", err)

    def test_bad_query_fails(self):
        self.run_main(["index", self.corpus_dir, "--index", self.index_path])
        code, _, err = self.run_main(
            ["query", "data AND", "--index", self.index_path])
        self.assertNotEqual(code, 0)
        self.assertIn("bad query", err)

    def test_no_arguments_fails(self):
        code, _, _ = self.run_main([])
        self.assertNotEqual(code, 0)


class DuplicateDocumentTests(unittest.TestCase):
    def test_duplicate_id_rejected(self):
        index = InvertedIndex()
        index.add_document("a", "hello")
        with self.assertRaises(ValueError):
            index.add_document("a", "world")


if __name__ == "__main__":
    unittest.main()
