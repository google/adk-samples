"""Deliberately tiny module for a reviewer test. Not a recipe.

The point of this file is that it is the ONLY thing the pull request changes.
If a review of this PR ever mentions a path outside this directory, the
reviewer is reading a diff that is not the PR's.
"""


def add(a: int, b: int) -> int:
    return a + b


def divide(a: int, b: int) -> float:
    # An obvious defect, left in on purpose: a review that is working should
    # find this, and a review that is reading the wrong diff will not.
    return a / b
