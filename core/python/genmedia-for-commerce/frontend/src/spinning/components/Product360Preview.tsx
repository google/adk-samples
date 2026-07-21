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

import { downloadVideo } from '../services/shoesSpinningApi'
import BackgroundVideo from '../../components/BackgroundVideo'
import { openFeedbackForm } from '../../components/FeedbackButton'

interface Product360PreviewProps {
  className?: string
  showVideo?: boolean
  videoUrl?: string | null
  videoBase64?: string | null
  isLoading?: boolean
  error?: string | null
  flow?: string
}

function Product360Preview({
  className = '',
  showVideo = true,
  videoUrl,
  videoBase64,
  isLoading = false,
  error,
  flow
}: Product360PreviewProps) {
  // When showVideo is false, use transparent background to let persistent video show through
  const panelClass = showVideo
    ? 'glass-panel rounded-lg'
    : 'rounded-lg border border-black/[0.08] dark:border-white/10'

  const handleDownload = () => {
    if (videoBase64) {
      downloadVideo(videoBase64, `spinning-360-${Date.now()}.mp4`)
    }
  }

  // Show error state
  if (error) {
    return (
      <div className={`${panelClass} flex items-center justify-center min-h-[600px] overflow-hidden relative ${className}`}>
        <div className="text-center px-8 relative z-10">
          <div className="w-16 h-16 rounded-full bg-red-500/10 flex items-center justify-center mx-auto mb-4">
            <svg width="32" height="32" viewBox="0 0 24 24" fill="none" className="text-red-500">
              <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="2"/>
              <path d="M12 8v4M12 16h.01" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
            </svg>
          </div>
          <h2 className="text-xl font-semibold mb-2 text-red-500">Generation Failed</h2>
          <p className="text-sm text-gm-text-secondary-light dark:text-gm-text-secondary max-w-md mb-4">
            {error}
          </p>
          <button
            onClick={() => openFeedbackForm({ feedbackType: 'Bug / Error', capability: flow ? `Product 360 - ${flow}` : 'Product 360', errorMessage: error })}
            className="text-sm text-gm-accent hover:text-gm-accent-hover transition-colors underline underline-offset-2"
          >
            Report this issue
          </button>
        </div>
      </div>
    )
  }

  // Show loading state
  if (isLoading) {
    return (
      <div className={`${panelClass} flex items-center justify-center min-h-[600px] overflow-hidden relative ${className}`}>
        {/* Animated background */}
        {showVideo && <BackgroundVideo opacity="opacity-30" />}

        {/* Gradient Overlay */}
        <div className="absolute inset-0 bg-gradient-to-t from-gm-bg-light/80 via-gm-bg-light/40 to-transparent dark:from-gm-bg/80 dark:via-gm-bg/40"></div>

        {/* Loading Content */}
        <div className="text-center px-8 relative z-10">
          <div className="relative w-20 h-20 mx-auto mb-6">
            {/* Spinning ring */}
            <div className="absolute inset-0 rounded-full border-4 border-gm-accent/20"></div>
            <div className="absolute inset-0 rounded-full border-4 border-transparent border-t-gm-accent animate-spin"></div>
            {/* Inner icon */}
            <div className="absolute inset-0 flex items-center justify-center">
              <svg width="28" height="28" viewBox="0 0 24 24" fill="none" className="text-gm-accent">
                <circle cx="12" cy="12" r="9" stroke="currentColor" strokeWidth="2"/>
                <path d="M12 3a15.3 15.3 0 014 9 15.3 15.3 0 01-4 9 15.3 15.3 0 01-4-9 15.3 15.3 0 014-9z" stroke="currentColor" strokeWidth="2"/>
                <path d="M3 12h18" stroke="currentColor" strokeWidth="2"/>
              </svg>
            </div>
          </div>
          <h2 className="text-2xl font-bold mb-2">
            Generating <span className="text-gm-accent">360° Spin</span>
          </h2>
          <p className="text-sm text-gm-text-secondary-light dark:text-gm-text-secondary">
            Processing your images and creating the video...
          </p>
          <p className="text-xs text-gm-text-tertiary-light dark:text-gm-text-tertiary mt-2">
            This may take a few minutes
          </p>
        </div>
      </div>
    )
  }

  // Show generated video
  if (videoUrl) {
    return (
      <div className={`${panelClass} flex flex-col min-h-[600px] overflow-hidden relative ${className}`}>
        {/* Video Player */}
        <div className="flex-1 flex items-center justify-center bg-black/5 dark:bg-black/20 relative">
          <video
            src={videoUrl}
            autoPlay
            loop
            muted
            playsInline
            controls
            className="max-w-full max-h-full w-auto h-auto object-contain"
          />
        </div>

        {/* Controls */}
        <div className="p-4 border-t border-black/[0.08] dark:border-white/10 flex items-center justify-between bg-white/50 dark:bg-black/20 backdrop-blur-sm">
          <div>
            <p className="text-sm font-medium text-gm-text-primary-light dark:text-gm-text-primary">
              360° Spin Generated
            </p>
            <p className="text-xs text-gm-text-tertiary-light dark:text-gm-text-tertiary">
              Video ready for download
            </p>
          </div>
          <button
            onClick={handleDownload}
            className="btn-primary px-4 py-2 rounded-lg text-sm font-medium flex items-center gap-2"
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
              <path d="M12 3v12M12 15l-4-4M12 15l4-4" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
              <path d="M2 17v2a2 2 0 002 2h16a2 2 0 002-2v-2" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
            </svg>
            Download
          </button>
        </div>
      </div>
    )
  }

  // Default state - ready to create
  return (
    <div className={`${panelClass} flex items-center justify-center min-h-[600px] overflow-hidden relative ${className}`}>
      {/* Video Background - Only render if showVideo is true */}
      {showVideo && <BackgroundVideo />}

      {/* Gradient Overlay */}
      <div className="absolute inset-0 bg-gradient-to-t from-gm-bg-light/50 via-gm-bg-light/20 to-transparent dark:from-gm-bg/50 dark:via-gm-bg/20"></div>

      {/* Content */}
      <div className="text-center px-8 relative z-10 max-w-lg animate-fade-in-up">
        <div className="w-16 h-16 mx-auto mb-6 rounded-2xl bg-gm-accent/10 dark:bg-gm-accent/20 flex items-center justify-center glow-accent">
          <svg width="28" height="28" viewBox="0 0 24 24" fill="none" className="text-gm-accent">
            <path d="M4 12a8 8 0 0116 0" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
            <path d="M20 12a8 8 0 01-16 0" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeDasharray="2 3" />
            <path d="M12 2v2M12 20v2M22 12h-2M4 12H2" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
          </svg>
        </div>

        <h2 className="text-[40px] lg:text-[48px] font-display font-bold mb-4 leading-tight tracking-tight">
          Product{' '}
          <span className="text-gm-accent" style={{ textShadow: '0 0 40px rgba(138, 180, 248, 0.4), 0 0 80px rgba(138, 180, 248, 0.2)' }}>
            360° Spin
          </span>
        </h2>
        <p className="text-body-lg text-gm-text-secondary-light dark:text-gm-text-secondary leading-relaxed">
          Upload product images to generate a smooth 360° spinning video.
        </p>

        <div className="flex flex-wrap items-center justify-center gap-2 mt-6">
          <span className="px-3 py-1.5 rounded-full bg-black/[0.08] dark:bg-white/[0.12] text-caption text-gm-text-secondary-light dark:text-gm-text-secondary">
            Veo 3.1
          </span>
          <span className="px-3 py-1.5 rounded-full bg-black/[0.08] dark:bg-white/[0.12] text-caption text-gm-text-secondary-light dark:text-gm-text-secondary">
            360° Rotation
          </span>
          <span className="px-3 py-1.5 rounded-full bg-black/[0.08] dark:bg-white/[0.12] text-caption text-gm-text-secondary-light dark:text-gm-text-secondary">
            Multi-Angle
          </span>
        </div>
      </div>
    </div>
  )
}

export default Product360Preview
