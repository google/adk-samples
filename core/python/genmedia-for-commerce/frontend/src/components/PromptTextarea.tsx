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

import { ChangeEvent, useState } from 'react'

interface PromptTextareaProps {
  value: string
  onChange: (value: string) => void
  placeholder?: string
  className?: string
  showGeminiButton?: boolean
  onGeminiClick?: () => void
  isEnhancing?: boolean
  readOnly?: boolean
  disabled?: boolean
}

function PromptTextarea({
  value,
  onChange,
  placeholder = 'Describe your prompt here...',
  className = '',
  showGeminiButton = true,
  onGeminiClick,
  isEnhancing = false,
  readOnly = false,
  disabled = false
}: PromptTextareaProps) {
  const [isFocused, setIsFocused] = useState(false)

  const handleChange = (e: ChangeEvent<HTMLTextAreaElement>) => {
    onChange(e.target.value)
  }

  const charCount = value.length
  const maxChars = 500

  if (readOnly) {
    return (
      <div className="relative group">
        <div className={`input-glass w-full min-h-[140px] p-4 text-body leading-relaxed rounded-xl ${className}`}>
          {value || <span className="text-gm-text-tertiary-light dark:text-gm-text-tertiary">{placeholder}</span>}
        </div>

        {/* Locked indicator */}
        <div className="absolute top-3 left-3 flex items-center gap-1.5 text-gm-text-tertiary-light dark:text-gm-text-tertiary">
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" className="opacity-60">
            <rect x="3" y="11" width="18" height="11" rx="2" stroke="currentColor" strokeWidth="2"/>
            <path d="M7 11V7a5 5 0 0110 0v4" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
          </svg>
          <span className="text-[10px] font-medium uppercase tracking-wider">Locked</span>
        </div>

        {showGeminiButton && (
          <button
            onClick={onGeminiClick}
            disabled={isEnhancing}
            className="absolute bottom-3 right-3 w-9 h-9 bg-black/[0.04] hover:bg-gm-accent/10 dark:bg-white/[0.06] dark:hover:bg-gm-accent/20 rounded-lg flex items-center justify-center transition-all duration-200 hover:scale-105 group/btn disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:scale-100"
            title="Enhance with Gemini"
          >
            {isEnhancing ? (
              <svg className="animate-spin h-5 w-5 text-gm-accent" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none"/>
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"/>
              </svg>
            ) : (
              <img src="/assets/branding/gemini.png" alt="Gemini AI" className="w-5 h-5 transition-transform group-hover/btn:scale-110" />
            )}
          </button>
        )}
      </div>
    )
  }

  return (
    <div className={`relative group ${isFocused ? 'ring-4 ring-gm-accent/10 rounded-xl' : ''}`}>
      <textarea
        value={value}
        onChange={handleChange}
        onFocus={() => setIsFocused(true)}
        onBlur={() => setIsFocused(false)}
        placeholder={placeholder}
        maxLength={maxChars}
        disabled={disabled}
        className={`input-glass w-full h-36 resize-none rounded-xl pr-12 ${className} ${
          isFocused ? 'border-gm-accent/50' : ''
        } ${disabled ? 'opacity-50 cursor-not-allowed' : ''}`}
      />

      {/* Character counter */}
      <div className={`absolute bottom-3 left-4 text-[10px] font-medium transition-opacity duration-200 ${
        isFocused || charCount > 0 ? 'opacity-100' : 'opacity-0'
      } ${charCount > maxChars * 0.9 ? 'text-gm-warning' : 'text-gm-text-tertiary-light dark:text-gm-text-tertiary'}`}>
        {charCount}/{maxChars}
      </div>

      {showGeminiButton && (
        <button
          onClick={onGeminiClick}
          disabled={isEnhancing}
          className="absolute bottom-3 right-3 w-9 h-9 bg-black/[0.04] hover:bg-gm-accent/10 dark:bg-white/[0.06] dark:hover:bg-gm-accent/20 rounded-lg flex items-center justify-center transition-all duration-200 hover:scale-105 group/btn disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:scale-100"
          title="Enhance with Gemini"
        >
          {isEnhancing ? (
            <svg className="animate-spin h-5 w-5 text-gm-accent" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none"/>
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"/>
            </svg>
          ) : (
            <img src="/assets/branding/gemini.png" alt="Gemini AI" className="w-5 h-5 transition-transform group-hover/btn:scale-110" />
          )}
        </button>
      )}

      {/* Focus glow effect */}
      {isFocused && (
        <div className="absolute -inset-px rounded-xl bg-gradient-to-r from-gm-accent/20 via-gm-accent/10 to-gm-accent/20 -z-10 blur-sm opacity-50" />
      )}
    </div>
  )
}

export default PromptTextarea
