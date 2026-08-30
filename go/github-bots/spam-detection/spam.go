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
	// Association is the commenter's GitHub author association (e.g.
	// FIRST_TIME_CONTRIBUTOR, NONE, MEMBER). It is a spam-likelihood prior fed to
	// the model, not a filter.
	Association string
	Body        string
}

// Issue is the normalized view of a GitHub issue used for spam review. It is
// deliberately small: only the fields needed to decide whether the content is
// spam.
type Issue struct {
	Number int
	Title  string
	Body   string
	Author string
	// Association is the issue author's GitHub author association (see Comment).
	Association string
	Labels      []string
	Comments    []Comment
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
// looking for spam: empty authors, any botSuffix account, the bot's own identity,
// and trusted maintainers. Their text is never sent to the model.
func isIgnoredAuthor(login, selfLogin string, maintainers map[string]bool) bool {
	if login == "" || strings.HasSuffix(login, botSuffix) {
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
// It deliberately does NOT trust the generic botSuffix suffix: anyone can create a
// GitHub App whose login ends in botSuffix, so trusting the suffix would let an
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
// Trust boundary: the authorship/association headers are TRUSTED scaffolding
// generated here from GitHub API metadata and are emitted OUTSIDE the fence.
// Each user-controlled blob (title+body, or a comment body) is wrapped in its
// own [UNTRUSTED:nonce] ... [/UNTRUSTED:nonce] fence. Because the nonce is
// unguessable, a spammer cannot close the fence to escape it, and because the
// headers live outside the fence they cannot forge a "Comment by @owner
// [author association: OWNER]" line inside their own text — any such attempt
// stays trapped inside the fence as inert data.
//
// It is pure so it can be exhaustively table-tested.
func assembleSuspectText(iss Issue, selfLogin string, maintainers map[string]bool, maxRunes int, nonce string) string {
	open, closeTag := "[UNTRUSTED:"+nonce+"]", "[/UNTRUSTED:"+nonce+"]"
	fence := func(s string) string { return open + "\n" + s + "\n" + closeTag }

	var sections []string

	if !isIgnoredAuthor(iss.Author, selfLogin, maintainers) {
		var content strings.Builder
		if title := clean(iss.Title, maxRunes); title != "" {
			content.WriteString("Title: " + title)
		}
		if body := clean(iss.Body, maxRunes); body != "" {
			if content.Len() > 0 {
				content.WriteString("\n")
			}
			content.WriteString("Body:\n" + body)
		}
		if content.Len() > 0 {
			header := fmt.Sprintf("Issue #%d opened by @%s%s", iss.Number, iss.Author, assocNote(iss.Association))
			sections = append(sections, header+"\n"+fence(content.String()))
		}
	}

	for _, c := range iss.Comments {
		if isIgnoredAuthor(c.Author, selfLogin, maintainers) {
			continue
		}
		if body := clean(c.Body, maxRunes); body != "" {
			header := fmt.Sprintf("Comment by @%s%s:", c.Author, assocNote(c.Association))
			sections = append(sections, header+"\n"+fence(body))
		}
	}

	return strings.Join(sections, "\n\n---\n\n")
}

// assocNote renders an author-association annotation for the prompt, e.g.
// " [author association: FIRST_TIME_CONTRIBUTOR]". It returns "" when the
// association is unknown so the prompt stays clean.
func assocNote(association string) string {
	if a := strings.TrimSpace(association); a != "" {
		return fmt.Sprintf(" [author association: %s]", a)
	}
	return ""
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
