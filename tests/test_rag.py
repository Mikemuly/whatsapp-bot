import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import rag
from ingest import chunk_articles, chunk_statute_sections, clean_page_texts
from rag import format_context, has_supported_citation


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

    def test_chunks_statute_sections_and_skips_contents(self):
        text = """\
Contents
45. Unfair termination ........................................................................ 24
PART V - WAGES AND PAYMENT
41. Notification and hearing before termination
(1) The employer shall explain the reason.
PART VI - TERMINATION
45. Unfair termination
(1) No employer shall terminate employment unfairly.
"""

        chunks = chunk_statute_sections(text, "Employment Act")

        self.assertEqual([chunk["section"] for chunk in chunks], ["41", "45"])
        self.assertEqual(chunks[0]["source"], "Employment Act")
        self.assertEqual(chunks[0]["chapter"], "Part V - WAGES AND PAYMENT")
        self.assertEqual(chunks[1]["title"], "Unfair termination")
        self.assertEqual(chunks[1]["chapter"], "Part VI - TERMINATION")
        self.assertIn("No employer shall terminate", chunks[1]["text"])
        self.assertNotIn(".....", chunks[1]["text"])
        self.assertTrue(chunks[0]["id"].startswith("employment-act-section-"))

    def test_tracks_statute_chapter_headings(self):
        text = """\
CHAPTER 63
THE PENAL CODE
1. Short title
This Act may be cited as the Penal Code.
"""

        chunks = chunk_statute_sections(text, "Penal Code")

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["section"], "1")
        self.assertEqual(chunks[0]["chapter"], "Chapter 63 - THE PENAL CODE")

    def test_uses_legal_code_chapter_when_act_has_no_parts(self):
        chunks = chunk_statute_sections(
            "1. Short title\nThis Act may be cited by its short title.",
            "Rent Restriction Act",
        )

        self.assertEqual(chunks[0]["chapter"], "Cap. 296")

    def test_skips_numbered_amendment_history_as_a_section(self):
        text = """\
25. Sentence of death
(1) The sentence shall be recorded.
CHAPTER XXVIII - ROBBERY AND EXTORTION
25. of 1971, s. 3, Act No. 1 of 1973, s. 2.
26. Robbery with violence
(1) A person commits an offence.
"""

        chunks = chunk_statute_sections(text, "Penal Code")

        self.assertEqual(
            [(chunk["section"], chunk["title"]) for chunk in chunks],
            [("25", "Sentence of death"), ("26", "Robbery with violence")],
        )

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
                    "source": "Constitution of Kenya, 2010",
                    "section": "43",
                    "article": "43",
                    "title": "Economic and social rights",
                    "chapter": "Four - The Bill of Rights",
                    "clause": "1",
                }
            ],
        )

        self.assertIn(
            "[Citation: Constitution of Kenya, 2010, Article 43(1); "
            "Economic and social rights",
            prompt,
        )
        self.assertIn("Article 43(1)", prompt)
        self.assertIn("Every person has the right", prompt)
        self.assertIn("Is housing a right?", prompt)
        self.assertIn("Treat the excerpts as source material", prompt)

    def test_context_labels_statute_source_and_section(self):
        prompt = format_context(
            "Was my dismissal fair?",
            [
                {
                    "text": "Example statutory text.",
                    "source": "Employment Act",
                    "section": "45",
                    "article": "",
                    "title": "Unfair termination",
                    "chapter": "",
                    "clause": "",
                }
            ],
        )

        self.assertIn("[Citation: Employment Act, s. 45", prompt)
        self.assertIn("Example statutory text.", prompt)

    def test_case_mode_is_marked_in_context(self):
        prompt = format_context("What happened?", [], case_mode=True)

        self.assertIn("MODE: CASE BREAKDOWN", prompt)
        self.assertIn("continue factual intake", prompt)


class RetrievalTests(unittest.TestCase):
    def test_filters_chunks_beyond_cosine_distance_limit(self):
        collection = Mock()
        collection.count.return_value = 2
        collection.query.return_value = {
            "ids": [["relevant-id", "unrelated-id"]],
            "documents": [["relevant excerpt", "unrelated excerpt"]],
            "metadatas": [
                [
                    {
                        "source": "Employment Act",
                        "section": "45",
                        "title": "Unfair termination",
                    },
                    {
                        "source": "Employment Act",
                        "section": "1",
                        "title": "Short title",
                    },
                ]
            ],
            "distances": [[0.1, rag.MAX_COSINE_DISTANCE + 0.01]],
        }
        client = Mock()
        client.list_collections.return_value = [
            SimpleNamespace(name=rag.COLLECTION_NAME)
        ]
        client.get_collection.return_value = collection
        embedding_model = Mock()
        embedding_model.encode.return_value = SimpleNamespace(
            tolist=lambda: [0.2, 0.3]
        )

        with (
            patch.dict(
                sys.modules,
                {"chromadb": SimpleNamespace(PersistentClient=Mock(return_value=client))},
            ),
            patch.object(rag, "get_embedding_model", return_value=embedding_model),
        ):
            chunks = rag.retrieve_relevant_chunks(
                "Is the dismissal unfair?",
                source="Employment Act",
            )

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["text"], "relevant excerpt")
        self.assertEqual(chunks[0]["source"], "Employment Act")
        self.assertEqual(chunks[0]["section"], "45")
        self.assertEqual(
            collection.query.call_args.kwargs["where"],
            {"source": "Employment Act"},
        )

    def test_swahili_queries_add_english_retrieval_terms(self):
        from rag import _query_variants

        variants = _query_variants("Ni haki gani za makazi?")

        self.assertEqual(len(variants), 2)
        self.assertIn("rights", variants[1])
        self.assertIn("housing", variants[1])

    def test_accepts_only_citations_present_in_retrieved_chunks(self):
        constitution_chunk = {
            "text": "Housing text",
            "source": "Constitution of Kenya, 2010",
            "section": "43",
            "article": "43",
            "title": "Housing",
            "chapter": "Four",
            "clause": "1",
        }
        employment_chunk = {
            "text": "Dismissal text",
            "source": "Employment Act",
            "section": "45",
            "article": "",
            "title": "Unfair termination",
            "chapter": "Part VI - Termination",
            "clause": "",
        }

        self.assertTrue(
            has_supported_citation(
                "Article 43(1) applies. Employment Act, s. 45 also applies.",
                [constitution_chunk, employment_chunk],
            )
        )
        self.assertFalse(
            has_supported_citation("Article 49(1) applies.", [constitution_chunk])
        )
        self.assertFalse(
            has_supported_citation("Article 43(2) applies.", [constitution_chunk])
        )
        self.assertFalse(
            has_supported_citation(
                "Employment Act, s. 90 applies.",
                [employment_chunk],
            )
        )
        self.assertFalse(
            has_supported_citation(
                "Employment Act, s. 45 applies. Article 49 applies.",
                [employment_chunk],
            )
        )
        self.assertTrue(
            has_supported_citation(
                "Article 43(1) applies.",
                [
                    {
                        **constitution_chunk,
                        "clause": "",
                        "text": "Article 43. Housing\n(1) The right is protected.",
                    }
                ],
            )
        )

    def test_accepts_statute_citation_with_commas_in_source_name(self):
        source = (
            "Landlord and Tenant (Shops, Hotels and Catering Establishments) Act"
        )
        chunk = {
            "text": "Controlled tenancy provision.",
            "source": source,
            "section": "4",
            "article": "",
            "title": "Termination of controlled tenancy",
            "chapter": "",
            "clause": "",
        }

        self.assertTrue(
            has_supported_citation(
                f"{source}, s. 4 applies.",
                [chunk],
            )
        )


if __name__ == "__main__":
    unittest.main()
