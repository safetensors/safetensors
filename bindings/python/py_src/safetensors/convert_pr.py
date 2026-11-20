"""Helpers for the weight-conversion script that do not import torch."""


def discussion_continues_revision(commits, revision_commit) -> bool:
    """Return whether a PR branch is exactly one commit ahead of ``revision_commit``.

    ``list_repo_commits`` is newest-first, so the parent of the tip is ``commits[1]``.
    A history shorter than two commits is not that PR; indexing it used to raise
    ``IndexError`` out of ``previous_pr`` and abort the conversion.
    """
    try:
        parent = commits[1]
    except IndexError:
        return False
    return revision_commit == parent.commit_id
