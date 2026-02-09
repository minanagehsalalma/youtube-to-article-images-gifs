import re
import unittest

import main as app


class MainHelpersTests(unittest.TestCase):
    def test_parse_num_handles_digits_and_words(self):
        self.assertEqual(app.parse_num("7"), 7)
        self.assertEqual(app.parse_num("twenty one"), 21)
        self.assertEqual(app.parse_num("forty-two"), 42)
        self.assertIsNone(app.parse_num("hundred"))

    def test_mmss_and_safe_slug(self):
        self.assertEqual(app.mmss(0), "00:00")
        self.assertEqual(app.mmss(65.9), "01:05")
        self.assertEqual(app.safe_slug("  ChatGPT Bulk Delete!  "), "chatgpt_bulk_delete")
        self.assertEqual(app.safe_slug(""), "item")


class TranscriptSegmentationTests(unittest.TestCase):
    def test_build_transcript_items_uses_numbered_markers_with_gap_fill(self):
        transcript = [
            {"start": 0.0, "duration": 2.0, "text": "Number one extension manager helps control plugins."},
            {"start": 10.0, "duration": 2.0, "text": "Number three vertical tabs improve organization."},
            {"start": 20.0, "duration": 2.0, "text": "Number four dark mode tools reduce eye strain."},
            {"start": 30.0, "duration": 2.0, "text": "Number five quick screenshot extension."},
        ]

        items, strategy = app.build_transcript_items(transcript, max_items=10, min_sections=3)
        numbers = [int(it["n"]) for it in items]

        self.assertEqual(strategy, "numbered_markers")
        self.assertEqual(numbers, [1, 2, 3, 4, 5])
        self.assertTrue(any(it["name"] == "Interpolated item 2" for it in items))

    def test_build_transcript_items_falls_back_to_time_chunks(self):
        transcript = [
            {"start": 0.0, "duration": 8.0, "text": "Welcome to the overview of browser productivity."},
            {"start": 8.0, "duration": 8.0, "text": "We discuss search tricks and clean tab workflows."},
            {"start": 16.0, "duration": 8.0, "text": "Then we cover reading mode and dark themes."},
            {"start": 24.0, "duration": 8.0, "text": "After that we compare screenshot utilities."},
            {"start": 32.0, "duration": 8.0, "text": "Finally we wrap up and summarize the stack."},
        ]

        items, strategy = app.build_transcript_items(transcript, max_items=10, min_sections=3)
        self.assertEqual(strategy, "time_chunks")
        self.assertGreaterEqual(len(items), 1)

    def test_attach_section_text_populates_raw_text(self):
        transcript = [
            {"start": 0.0, "duration": 5.0, "text": "Number one extension manager setup."},
            {"start": 5.0, "duration": 5.0, "text": "More details about extension manager."},
            {"start": 12.0, "duration": 5.0, "text": "Number two bulk delete and queueing for chat gpt."},
            {"start": 18.0, "duration": 5.0, "text": "More details about bulk delete."},
        ]
        items = [
            {"n": 2, "name": "Bulk delete", "start": 12.0, "seg_i": 2},
            {"n": 1, "name": "Extension manager", "start": 0.0, "seg_i": 0},
        ]

        attached = app.attach_section_text(transcript, items)

        self.assertEqual([it["n"] for it in attached], [1, 2])
        self.assertIn("extension manager", attached[0]["raw_text"].lower())
        self.assertIn("bulk delete", attached[1]["raw_text"].lower())


class ArticleRenderingTests(unittest.TestCase):
    def test_build_transcript_article_emits_placeholders(self):
        items = [
            {"n": 1, "name": "Autotoma", "start": 862.0, "raw_text": "Number one autotoma automates actions."},
            {"n": 2, "name": "Bulk delete", "start": 896.0, "raw_text": "Number two bulk delete for chatgpt."},
        ]

        md = app.build_transcript_article(items, strategy="numbered_markers")
        placeholders = re.findall(app.PLACEHOLDER_LINE_RE.pattern, md, flags=re.MULTILINE)

        self.assertIn("## 1. Autotoma", md)
        self.assertIn("## 2. Bulk delete", md)
        self.assertIn("timestamp 14:22", md)
        self.assertIn("timestamp 14:56", md)
        self.assertEqual(len(placeholders), 2)

    def test_md_to_html_basic_renders_markdown_elements(self):
        md = (
            "# Title\n\n"
            "## Subtitle\n\n"
            "Paragraph text.\n\n"
            "![Alt](images/example.jpg)\n\n"
            "---\n"
        )
        html = app.md_to_html_basic(md)

        self.assertIn("<h1>Title</h1>", html)
        self.assertIn("<h2>Subtitle</h2>", html)
        self.assertIn("<p>Paragraph text.</p>", html)
        self.assertIn('<img src="images/example.jpg" alt="Alt"/>', html)
        self.assertIn("<hr/>", html)

    def test_md_to_html_basic_supports_style_presets(self):
        md = "# Title\n\nParagraph text."
        html_basic = app.md_to_html_basic(md, style="basic")
        html_article = app.md_to_html_basic(md, style="article")

        self.assertIn("font-family: system-ui", html_basic)
        self.assertIn("--content-max: 1180px", html_article)


if __name__ == "__main__":
    unittest.main()
