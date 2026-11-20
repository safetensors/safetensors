import unittest

from safetensors.convert_pr import discussion_continues_revision


class _Commit:
    def __init__(self, commit_id):
        self.commit_id = commit_id


class DiscussionContinuesRevisionTest(unittest.TestCase):
    def test_one_commit_history_is_not_a_match(self):
        commits = [_Commit("only")]
        self.assertFalse(discussion_continues_revision(commits, "base"))

    def test_empty_history_is_not_a_match(self):
        self.assertFalse(discussion_continues_revision([], "base"))

    def test_parent_of_tip_matches_the_revision(self):
        commits = [_Commit("tip"), _Commit("base")]
        self.assertTrue(discussion_continues_revision(commits, "base"))

    def test_different_parent_does_not_match(self):
        commits = [_Commit("tip"), _Commit("other")]
        self.assertFalse(discussion_continues_revision(commits, "base"))
