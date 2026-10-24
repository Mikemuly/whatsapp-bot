import unittest

from ingest import chunk_articles, clean_page_texts
from rag import format_context


class ChunkingTests(unittest.TestCase):
    def test_skips_table_of_contents_entries(self):
        text = """\
Contents
43. Economic and social rights ........................................................................ 14
CHAPTER FOUR
THE BILL OF RIGHTS
43. Economic and social rights
(1) Every person has the right to accessible and adequate housing.
"""

        chunks = chunk_articles(text)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["article"], "43")
        self.assertIn("accessible and adequate housing", chunks[0]["text"])

    def test_splits_articles_and_tracks_chapter(self):
        text = """\
CHAPTER FOUR
THE BILL OF RIGHTS
43. Economic and social rights
(1) Every person has the right to social security.
44. Language and culture
(1) Every person has the right to use the language of the person's choice.
"""

        chunks = chunk_articles(text)

        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0]["article"], "43")
        self.assertEqual(chunks[0]["title"], "Economic and social rights")
        self.assertEqual(chunks[0]["chapter"], "Four - The Bill Of Rights")
        self.assertIn("social security", chunks[0]["text"])
        self.assertEqual(chunks[1]["article"], "44")

    def test_splits_long_articles_by_clause_and_overlaps(self):
        clause_one = " ".join(["housing"] * 360)
        clause_two = " ".join(["health"] * 360)
        chunks = chunk_articles(
            "CHAPTER FOUR\nTHE BILL OF RIGHTS\n"
            f"43. Economic and social rights\n(1) {clause_one}\n(2) {clause_two}"
        )

        self.assertGreaterEqual(len(chunks), 2)
        self.assertEqual({chunk["article"] for chunk in chunks}, {"43"})
        self.assertTrue(any("1" in chunk["clause"] for chunk in chunks))
        self.assertTrue(any("2" in chunk["clause"] for chunk in chunks))
        self.assertTrue(all(len(chunk["text"].split()) <= 500 for chunk in chunks))
        for previous, following in zip(chunks, chunks[1:]):
            previous_words = previous["text"].split()
            following_words = following["text"].split()
            self.assertEqual(
                previous_words[-50:],
                following_words[6:56],
            )

    def test_stops_before_schedule_items_reuse_article_numbers(self):
        text = """\
CHAPTER FOUR
THE BILL OF RIGHTS
43. Economic and social rights
(1) Every person has the right to accessible and adequate housing.
44. Language and culture
(1) Every person has the right to use the language of the person's choice.
FIRST SCHEDULE
1. Oaths and solemn affirmations
43. Homa Bay
"""

        chunks = chunk_articles(text)

        self.assertEqual([chunk["article"] for chunk in chunks], ["43", "44"])
        self.assertTrue(all("Homa Bay" not in chunk["text"] for chunk in chunks))

    def test_fails_when_no_article_headings_found(self):
        with self.assertRaisesRegex(ValueError, "No Article headings found"):
            chunk_articles("The PDF extraction has no article headings.")

    def test_removes_repeated_page_header_and_page_numbers(self):
        cleaned = clean_page_texts(
            [
                "KENYA LAW\n1\nOpening text continues",
                "KENYA LAW\n2\nSecond paragraph continues",
                "KENYA LAW\n3\nFinal paragraph continues",
            ]
        )

        self.assertNotIn("KENYA LAW", cleaned)
        self.assertNotRegex(cleaned, r"(?m)^\d+$")
        self.assertIn("Opening text continues", cleaned)
        self.assertIn("Final paragraph continues", cleaned)


class RetrievalPromptTests(unittest.TestCase):
    def test_context_labels_article_and_clause(self):
        prompt = format_context(
            "Is housing a right?",
            [
                {
                    "text": "Every person has the right to accessible and "
                    "adequate housing.",
                    "article": "43",
                    "title": "Economic and social rights",
                    "chapter": "Four - The Bill of Rights",
                    "clause": "1",
                }
            ],
        )

        self.assertIn("[Article 43: Economic and social rights", prompt)
        self.assertIn("clause 1", prompt)
        self.assertIn("Every person has the right", prompt)
        self.assertIn("Is housing a right?", prompt)
        self.assertIn("Treat the excerpts as source material", prompt)


if __name__ == "__main__":
    unittest.main()
