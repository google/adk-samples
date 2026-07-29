/**
 * Copyright 2026 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

const FORM_BASE_URL = 'https://docs.google.com/forms/d/e/1FAIpQLSfze7bTlmlC8a4pFpY26VsMnfmlpRyRFZkmsgEC7uVwuLskKA/viewform'

const ENTRY_IDS = {
  feedbackType: 'entry.214913093',
  capability: 'entry.632720233',
  whatHappened: 'entry.519231100',
  errorMessage: 'entry.10556972',
  inputFiles: 'entry.139467209',
  outputFiles: 'entry.1150184131',
  email: 'entry.1989095905',
}

interface FeedbackParams {
  feedbackType?: 'Bug / Error' | 'Quality Issue' | 'Feature Request' | 'Other'
  capability?: string
  errorMessage?: string
}

export function openFeedbackForm(params: FeedbackParams = {}) {
  const query = new URLSearchParams()
  query.set('usp', 'pp_url')

  if (params.feedbackType) {
    query.set(ENTRY_IDS.feedbackType, params.feedbackType)
  }
  if (params.capability) {
    query.set(ENTRY_IDS.capability, params.capability)
  }
  if (params.errorMessage) {
    query.set(ENTRY_IDS.errorMessage, params.errorMessage)
  }

  window.open(`${FORM_BASE_URL}?${query.toString()}`, '_blank')
}

interface FeedbackButtonProps {
  capability?: string
}

function FeedbackButton({ capability }: FeedbackButtonProps) {
  return (
    <button
      onClick={() => openFeedbackForm({ capability })}
      className="fixed bottom-6 right-6 z-50 flex items-center gap-2 px-4 py-2.5 rounded-full bg-white dark:bg-white/10 border border-black/[0.08] dark:border-white/[0.1] shadow-level-2 hover:shadow-level-3 text-button font-medium text-gm-text-secondary-light dark:text-gm-text-secondary hover:text-gm-text-primary-light dark:hover:text-gm-text-primary transition-all duration-200 backdrop-blur-sm group"
      aria-label="Send feedback"
    >
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" className="opacity-60 group-hover:opacity-100 transition-opacity">
        <path d="M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2v10z" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
      </svg>
      Feedback
    </button>
  )
}

export default FeedbackButton
