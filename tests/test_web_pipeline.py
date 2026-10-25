"""
tests/test_web_pipeline.py
Unit tests for web source tiering, URL cleaning, deduplication, and article extraction.
"""

import unittest

from web.extraction import extract_article_content
from web.search import deduplicate_and_cluster
from web.sources import clean_url, get_domain, get_source_quality


class TestWebPipeline(unittest.TestCase):
    def test_source_quality_tiering(self):
        # Tier 1 wire / official
        q1, tier1, _ = get_source_quality("https://www.reuters.com/world/article", "Reuters")
        self.assertEqual(tier1, 1)
        self.assertEqual(q1, 1.0)

        q_bbc, tier_bbc, _ = get_source_quality("https://www.bbc.com/news/123", "BBC News")
        self.assertEqual(tier_bbc, 1)

        # Tier 2 major publication
        q2, tier2, _ = get_source_quality("https://www.thehindu.com/news/national/article", "The Hindu")
        self.assertEqual(tier2, 2)
        self.assertGreaterEqual(q2, 0.8)

        # Tier 3 encyclopedia / magazine
        q3, tier3, _ = get_source_quality("https://en.wikipedia.org/wiki/Machine_learning", "Wikipedia")
        self.assertEqual(tier3, 3)

        # Tier 4 unknown blog
        q4, tier4, _ = get_source_quality("https://random-unknown-blog-123.com/post", "Unknown Blog")
        self.assertEqual(tier4, 4)
        self.assertLessEqual(q4, 0.5)

    def test_clean_url(self):
        url = "https://www.thehindu.com/news/article.ece?utm_source=rss&utm_medium=feed&oc=5"
        cleaned = clean_url(url)
        self.assertNotIn("utm_source", cleaned)
        self.assertNotIn("oc=5", cleaned)
        self.assertIn("article.ece", cleaned)

    def test_deduplicate_and_cluster(self):
        candidates = [
            {
                "title": "Global Leaders Meet in Geneva for AI Safety Summit",
                "url": "https://www.reuters.com/article-1",
                "publisher": "Reuters",
                "snippet": "Leaders met in Geneva."
            },
            {
                "title": "Global Leaders Meet in Geneva for AI Safety Summit",
                "url": "https://www.unknownblog.com/repost",
                "publisher": "Unknown Blog",
                "snippet": "Leaders met in Geneva."
            },
            {
                "title": "Unrelated Sports Update on Champions League",
                "url": "https://www.espn.com/sports-1",
                "publisher": "ESPN",
                "snippet": "Champions league match results."
            }
        ]
        deduped = deduplicate_and_cluster(candidates)
        # Should deduplicate the two identical AI Safety articles and retain Reuters (Tier 1)
        self.assertEqual(len(deduped), 2)
        publishers = [d["publisher"] for d in deduped]
        self.assertIn("Reuters", publishers)
        self.assertIn("ESPN", publishers)


if __name__ == "__main__":
    unittest.main()
