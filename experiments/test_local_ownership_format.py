import unittest
from diagnose_local_ownership_format import format_summary


class LocalOwnershipFormatTests(unittest.TestCase):
    def test_thinking_truncation_is_not_semantic_abstention(self):
        result = format_summary([dict(response='', thinking='internal work', done_reason='length', eval_count=16)])
        self.assertEqual(result['empty'], 1)
        self.assertEqual(result['exact_unclear'], 0)
        self.assertEqual(result['thinking_nonempty'], 1)
        self.assertEqual(result['done_reasons'], {'length':1})

    def test_complete_abstention_and_complete_label_are_distinct(self):
        result = format_summary([dict(response='unclear.', done_reason='stop', eval_count=3),
                                 dict(response='home', done_reason='stop', eval_count=2)])
        self.assertEqual(result['empty'], 0)
        self.assertEqual(result['exact_unclear'], 1)
        self.assertEqual(result['thinking_nonempty'], 0)
        self.assertEqual(result['done_reasons'], {'stop':2})

    def test_narrative_with_one_keyword_is_a_syntax_rejection(self):
        result = format_summary([dict(response='The home team', done_reason='stop', eval_count=4)])
        self.assertEqual(result['syntax_rejections'], 1)


if __name__ == '__main__':
    unittest.main()
