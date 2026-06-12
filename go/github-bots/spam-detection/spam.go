// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package main

import (
	"fmt"
	"strings"
)

// botAlertSignature is the leading text of the comment the bot posts when it
// flags an issue. It must stay in sync with the body written by
// buildAlertComment so the bot can recognize its own past alerts and avoid
// posting a duplicate on a later run.
const botAlertSignature = "**Automated spam detection:**"

// maxSnippetRunes bounds how much of any single piece of user text (the issue
// body or one comment) is forwarded to the model. Long text is truncated; this
// keeps the prompt small and the run cheap.
const maxSnippetRunes = 1500

// Comment is the normalized view of a single issue comment.
type Comment struct {
	Author string
	Body   string
}

// Issue is the normalized view of a GitHub issue used for spam review. It is
// deliberately small: only the fields needed to decide whether the content is
// spam.
type Issue struct {
	Number   int
	Title    string
	Body     string
	Author   string
	Labels   []string
	Comments []Comment
}

// maintainerSet builds a lowercased lookup set of maintainer logins. GitHub
// logins are case-insensitive, so normalizing here lets a maintainer configured
// as "Alice" match the API's "alice".
func maintainerSet(logins []string) map[string]bool {
	m := make(map[string]bool, len(logins))
	for _, l := range logins {
		if l = strings.ToLower(strings.TrimSpace(l)); l != "" {
			m[l] = true
		}
	}
	return m
}

// isIgnoredAuthor reports whether content from this login should be skipped when
// looking for spam: empty authors, any "[bot]" account, the bot's own identity,
// and trusted maintainers. Their text is never sent to the model.
func isIgnoredAuthor(login, selfLogin string, maintainers map[string]bool) bool {
	if login == "" || strings.HasSuffix(login, "[bot]") {
		return true
	}
	if isSelfAuthor(login, selfLogin) {
		return true
	}
	// GitHub logins are case-insensitive; the set is lowercased to match.
	return maintainers[strings.ToLower(login)]
}

// isSelfAuthor reports whether a login is the bot's own resolved identity
// (case-insensitive). Used to authenticate the bot's own alert comments.
//
// It deliberately does NOT trust the generic "[bot]" suffix: anyone can create a
// GitHub App whose login ends in "[bot]", so trusting the suffix would let an
// attacker post a comment carrying botAlertSignature to make hasBotAlert treat
// the issue as already handled and suppress moderation. When the identity could
// not be resolved (selfLogin == ""), this returns false and idempotency falls
// back to the spam label (the primary guard).
func isSelfAuthor(login, selfLogin string) bool {
	return selfLogin != "" && strings.EqualFold(login, selfLogin)
}

// hasLabel reports whether labels contains target (case-insensitive).
func hasLabel(labels []string, target string) bool {
	for _, l := range labels {
		if strings.EqualFold(strings.TrimSpace(l), target) {
			return true
		}
	}
	return false
}

// hasBotAlert reports whether the bot has already posted its alert comment on
// this issue. It only counts comments authored by the bot, so a spammer cannot
// suppress detection by pasting the signature into their own comment.
func hasBotAlert(iss Issue, selfLogin string) bool {
	for _, c := range iss.Comments {
		if isSelfAuthor(c.Author, selfLogin) && strings.Contains(c.Body, botAlertSignature) {
			return true
		}
	}
	return false
}

// alreadyHandled reports whether the issue has already been actioned and should
// be skipped before the model is invoked: it already carries the spam label, or
// the bot has already alerted on it. This is the idempotency gate that prevents
// duplicate labels and comments across runs (and in single-issue mode, where the
// search-time -label:spam filter does not apply).
func alreadyHandled(iss Issue, selfLogin, spamLabel string) bool {
	return hasLabel(iss.Labels, spamLabel) || hasBotAlert(iss, selfLogin)
}

// clean normalizes a piece of user text for review: it trims surrounding
// whitespace and truncates to maxRunes.
//
// It deliberately does NOT strip fenced code blocks (the Python original did):
// spam hidden inside a ``` fence would then never be reviewed. Keeping the text
// and bounding it with truncation closes that bypass while still capping tokens.
func clean(s string, maxRunes int) string {
	return truncateRunes(strings.TrimSpace(s), maxRunes)
}

// truncateRunes shortens s to at most n runes, appending a marker when it trims.
func truncateRunes(s string, n int) string {
	r := []rune(s)
	if len(r) <= n {
		return s
	}
	return string(r[:n]) + " …[truncated]"
}

// assembleSuspectText builds the text handed to the model for one issue: the
// issue's own title/body (when its author is reviewable) followed by each
// reviewable comment, with long text truncated. It returns "" when there is
// nothing to review (e.g. every author is a maintainer or a bot), which lets the
// caller skip the issue without invoking the model.
//
// It is pure so it can be exhaustively table-tested.
func assembleSuspectText(iss Issue, selfLogin string, maintainers map[string]bool, maxRunes int) string {
	var sections []string

	if !isIgnoredAuthor(iss.Author, selfLogin, maintainers) {
		body := clean(iss.Body, maxRunes)
		header := fmt.Sprintf("Issue #%d opened by @%s", iss.Number, iss.Author)
		if title := strings.TrimSpace(iss.Title); title != "" {
			header += fmt.Sprintf("\nTitle: %s", title)
		}
		if body != "" {
			sections = append(sections, header+"\nBody:\n"+body)
		} else if strings.TrimSpace(iss.Title) != "" {
			// A title with an empty body is still worth reviewing (spam titles).
			sections = append(sections, header)
		}
	}

	for _, c := range iss.Comments {
		if isIgnoredAuthor(c.Author, selfLogin, maintainers) {
			continue
		}
		if body := clean(c.Body, maxRunes); body != "" {
			sections = append(sections, fmt.Sprintf("Comment by @%s:\n%s", c.Author, body))
		}
	}

	return strings.Join(sections, "\n\n---\n\n")
}

// buildAlertComment renders the maintainer-facing comment the bot posts when it
// flags an issue. It always begins with botAlertSignature (so the bot can
// recognize its own alerts) and embeds the model's reason as an inert fenced
// block so the reason text cannot break the comment's Markdown.
func buildAlertComment(reason string) string {
	// Neutralize any fences in the (model-authored) reason so it cannot escape
	// the code block below.
	safe := strings.ReplaceAll(strings.TrimSpace(reason), "```", "'''")
	if safe == "" {
		safe = "(no reason provided)"
	}
	return fmt.Sprintf(
		"%s a suspected spam comment was detected in this thread. "+
			"Maintainers, please review.\n\nReason:\n```text\n%s\n```",
		botAlertSignature, safe,
	)
}
