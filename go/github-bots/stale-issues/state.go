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
	"slices"
	"sort"
	"strings"
	"time"
)

// botAlertSignature is the leading text of the comment the bot posts when it
// detects a "silent" description edit. It must stay in sync with the body
// written by alertMaintainerOfEdit so the bot can recognize its own alerts and
// avoid spamming the thread.
const botAlertSignature = "**Notification:** The author has updated the issue description"

// Role classifies the last human actor on an issue.
type Role string

const (
	roleAuthor     Role = "author"
	roleMaintainer Role = "maintainer"
	roleOther      Role = "other_user"
)

// eventType is the kind of a normalized history event.
type eventType string

const (
	eventCreated      eventType = "created"
	eventCommented    eventType = "commented"
	eventEditedDesc   eventType = "edited_description"
	eventRenamedTitle eventType = "renamed_title"
	eventReopened     eventType = "reopened"
)

// historyEvent is a single normalized, human-attributed event on the issue
// timeline.
type historyEvent struct {
	Type  eventType
	Actor string
	Time  time.Time
	Body  string // populated for comments only
}

// --- Raw GraphQL shapes -----------------------------------------------------
//
// These mirror the GraphQL response so the GitHub client can decode directly
// into them, while the pure functions below operate on this typed input
// (keeping them trivially unit-testable with struct literals).

type rawActor struct {
	Login string `json:"login"`
}

type rawComment struct {
	Author       *rawActor  `json:"author"`
	Body         string     `json:"body"`
	CreatedAt    time.Time  `json:"createdAt"`
	LastEditedAt *time.Time `json:"lastEditedAt"`
}

type rawEdit struct {
	Editor   *rawActor `json:"editor"`
	EditedAt time.Time `json:"editedAt"`
}

type rawTimelineItem struct {
	Typename  string    `json:"__typename"`
	CreatedAt time.Time `json:"createdAt"`
	Actor     *rawActor `json:"actor"`
	Label     *struct {
		Name string `json:"name"`
	} `json:"label"`
}

type rawIssue struct {
	Author    *rawActor `json:"author"`
	CreatedAt time.Time `json:"createdAt"`
	Labels    struct {
		Nodes []struct {
			Name string `json:"name"`
		} `json:"nodes"`
	} `json:"labels"`
	Comments struct {
		Nodes []rawComment `json:"nodes"`
	} `json:"comments"`
	UserContentEdits struct {
		Nodes []rawEdit `json:"nodes"`
	} `json:"userContentEdits"`
	TimelineItems struct {
		Nodes []rawTimelineItem `json:"nodes"`
	} `json:"timelineItems"`
}

// IssueState is the structured summary handed to the LLM. The JSON field names
// match the keys referenced by the prompt's decision tree.
type IssueState struct {
	Status                string   `json:"status"`
	LastActionRole        string   `json:"last_action_role"`
	LastActionType        string   `json:"last_action_type"`
	LastActorName         string   `json:"last_actor_name"`
	MaintainerAlertNeeded bool     `json:"maintainer_alert_needed"`
	IsStale               bool     `json:"is_stale"`
	DaysSinceActivity     float64  `json:"days_since_activity"`
	DaysSinceStaleLabel   float64  `json:"days_since_stale_label"`
	LastCommentText       string   `json:"last_comment_text"`
	CurrentLabels         []string `json:"current_labels"`
	StaleThresholdDays    float64  `json:"stale_threshold_days"`
	CloseThresholdDays    float64  `json:"close_threshold_days"`
	Maintainers           []string `json:"maintainers"`
	IssueAuthor           string   `json:"issue_author"`
}

// isIgnoredActor reports whether events from this login should be ignored:
// empty actors, any "[bot]" account, and the bot's own identity.
func isIgnoredActor(login, selfLogin string) bool {
	return login == "" || strings.HasSuffix(login, "[bot]") || (selfLogin != "" && login == selfLogin)
}

// buildTimeline normalizes the raw GraphQL data into a chronologically sorted
// list of human events. It also returns the times the stale label was applied
// and the most recent time the bot posted a silent-edit alert (used for spam
// prevention).
func buildTimeline(raw *rawIssue, selfLogin, staleLabel string) (events []historyEvent, staleLabelTimes []time.Time, lastBotAlert time.Time) {
	author := actorLogin(raw.Author)

	// Baseline: issue creation.
	events = append(events, historyEvent{Type: eventCreated, Actor: author, Time: raw.CreatedAt})

	// Comments.
	for _, c := range raw.Comments.Nodes {
		actor := actorLogin(c.Author)
		// Track the bot's own silent-edit alerts; never add them to history.
		if strings.Contains(c.Body, botAlertSignature) {
			if lastBotAlert.IsZero() || c.CreatedAt.After(lastBotAlert) {
				lastBotAlert = c.CreatedAt
			}
			continue
		}
		if isIgnoredActor(actor, selfLogin) {
			continue
		}
		// Prefer the edit time when a comment was later edited.
		when := c.CreatedAt
		if c.LastEditedAt != nil && !c.LastEditedAt.IsZero() {
			when = *c.LastEditedAt
		}
		events = append(events, historyEvent{Type: eventCommented, Actor: actor, Time: when, Body: c.Body})
	}

	// Description edits ("ghost edits").
	for _, e := range raw.UserContentEdits.Nodes {
		actor := actorLogin(e.Editor)
		if isIgnoredActor(actor, selfLogin) {
			continue
		}
		events = append(events, historyEvent{Type: eventEditedDesc, Actor: actor, Time: e.EditedAt})
	}

	// Timeline events: stale-label applications, title renames, reopens.
	for _, t := range raw.TimelineItems.Nodes {
		switch t.Typename {
		case "LabeledEvent":
			if t.Label != nil && t.Label.Name == staleLabel {
				staleLabelTimes = append(staleLabelTimes, t.CreatedAt)
			}
			continue
		}
		actor := actorLogin(t.Actor)
		if isIgnoredActor(actor, selfLogin) {
			continue
		}
		et := eventReopened
		if t.Typename == "RenamedTitleEvent" {
			et = eventRenamedTitle
		}
		events = append(events, historyEvent{Type: et, Actor: actor, Time: t.CreatedAt})
	}

	sort.SliceStable(events, func(i, j int) bool { return events[i].Time.Before(events[j].Time) })
	return events, staleLabelTimes, lastBotAlert
}

// replayResult captures the outcome of replaying the event history.
type replayResult struct {
	LastActorRole   Role
	LastActivity    time.Time
	LastActionType  eventType
	LastActorName   string
	LastCommentText string
	LastCommentBy   string // login of LastCommentText's author
}

// replay walks the sorted history to find the last human actor and their role.
//
// LastCommentText retains the most recent comment even when a later non-comment
// event (e.g. a title rename) becomes the last action, so the maintainer-intent
// analysis still has text to work with. LastCommentBy records who wrote it, so
// computeIssueState can avoid attributing one person's comment to another.
func replay(events []historyEvent, maintainers map[string]bool, author string) replayResult {
	st := replayResult{LastActorRole: roleAuthor, LastActionType: eventCreated, LastActorName: author}
	if len(events) > 0 {
		st.LastActivity = events[0].Time
	}
	for _, e := range events {
		st.LastActorRole = classify(e.Actor, author, maintainers)
		st.LastActivity = e.Time
		st.LastActionType = e.Type
		st.LastActorName = e.Actor
		if e.Type == eventCommented {
			st.LastCommentText = e.Body
			st.LastCommentBy = e.Actor
		}
	}
	return st
}

func classify(actor, author string, maintainers map[string]bool) Role {
	switch {
	case actor == author:
		return roleAuthor
	case maintainers[actor]:
		return roleMaintainer
	default:
		return roleOther
	}
}

// computeIssueState orchestrates timeline construction, replay, and the final
// staleness/alert calculations. It is pure: all inputs are explicit (including
// now) so it can be exhaustively unit-tested.
func computeIssueState(raw *rawIssue, selfLogin string, maintainers []string, staleLabel string, staleAfter, closeAfter time.Duration, now time.Time) IssueState {
	author := actorLogin(raw.Author)
	labels := make([]string, 0, len(raw.Labels.Nodes))
	for _, l := range raw.Labels.Nodes {
		labels = append(labels, l.Name)
	}

	events, staleLabelTimes, lastBotAlert := buildTimeline(raw, selfLogin, staleLabel)
	st := replay(events, toSet(maintainers), author)

	daysSinceActivity := now.Sub(st.LastActivity).Hours() / 24

	isStale := slices.Contains(labels, staleLabel)
	daysSinceStaleLabel := 0.0
	if isStale {
		if len(staleLabelTimes) > 0 {
			daysSinceStaleLabel = now.Sub(latest(staleLabelTimes)).Hours() / 24
		} else {
			// The stale LabeledEvent can scroll out of the bounded timeline
			// window on very active issues. Fall back to time since the last
			// activity so a stale, inactive issue can still be closed rather
			// than lingering open forever.
			daysSinceStaleLabel = daysSinceActivity
		}
	}

	// Surface the last comment only when its author is also the last actor, so
	// the maintainer-intent step never analyzes someone else's words (e.g. the
	// author's question) as if the maintainer had written them.
	lastCommentText := ""
	if st.LastCommentBy == st.LastActorName {
		lastCommentText = st.LastCommentText
	}

	// Silent-edit alert: the author/user edited the description without
	// commenting, and we have not already alerted about an edit since then.
	alertNeeded := false
	if (st.LastActorRole == roleAuthor || st.LastActorRole == roleOther) && st.LastActionType == eventEditedDesc {
		if lastBotAlert.IsZero() || !lastBotAlert.After(st.LastActivity) {
			alertNeeded = true
		}
	}

	return IssueState{
		Status:                "success",
		LastActionRole:        string(st.LastActorRole),
		LastActionType:        string(st.LastActionType),
		LastActorName:         st.LastActorName,
		MaintainerAlertNeeded: alertNeeded,
		IsStale:               isStale,
		DaysSinceActivity:     daysSinceActivity,
		DaysSinceStaleLabel:   daysSinceStaleLabel,
		LastCommentText:       lastCommentText,
		CurrentLabels:         labels,
		StaleThresholdDays:    staleAfter.Hours() / 24,
		CloseThresholdDays:    closeAfter.Hours() / 24,
		Maintainers:           maintainers,
		IssueAuthor:           author,
	}
}

func actorLogin(a *rawActor) string {
	if a == nil {
		return ""
	}
	return a.Login
}

func toSet(xs []string) map[string]bool {
	m := make(map[string]bool, len(xs))
	for _, x := range xs {
		m[x] = true
	}
	return m
}

func latest(ts []time.Time) time.Time {
	var newest time.Time
	for _, t := range ts {
		if t.After(newest) {
			newest = t
		}
	}
	return newest
}
