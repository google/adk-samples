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

import { useState, useEffect } from 'react'
import BackgroundVideo from '../../components/BackgroundVideo'
import { openFeedbackForm } from '../../components/FeedbackButton'
import type { FittingPipelineResult, VideoResult, FittingValidation } from './productFittingApi'

interface ProductFittingPreviewProps {
  className?: string
  showVideo?: boolean
  isLoading?: boolean
  error?: string | null
  result?: FittingPipelineResult | null
  onAnimate?: () => void
  isVideoLoading?: boolean
  videoResult?: VideoResult | null
  videoError?: string | null
}

function ProductFittingPreview({
  className = '',
  showVideo = true,
  isLoading = false,
  error,
  result,
  onAnimate,
  isVideoLoading = false,
  videoResult,
  videoError,
}: ProductFittingPreviewProps) {
  const panelClass = showVideo
    ? 'glass-panel rounded-lg'
    : 'rounded-lg border border-black/[0.08] dark:border-white/10'

  const [videoUrl, setVideoUrl] = useState<string | null>(null)
  const [selectedImage, setSelectedImage] = useState<string | null>(null)

  useEffect(() => {
    if (videoResult && videoResult.videos && videoResult.videos.length > 0) {
      const base64Data = videoResult.videos[0]
      const byteCharacters = atob(base64Data)
      const byteNumbers = new Array(byteCharacters.length)
      for (let i = 0; i < byteCharacters.length; i++) {
        byteNumbers[i] = byteCharacters.charCodeAt(i)
      }
      const byteArray = new Uint8Array(byteNumbers)
      const blob = new Blob([byteArray], { type: 'video/mp4' })
      const url = URL.createObjectURL(blob)
      setVideoUrl(url)

      return () => {
        URL.revokeObjectURL(url)
      }
    } else {
      setVideoUrl(null)
    }
  }, [videoResult])

  // Error state
  if (error) {
    return (
      <div className={`${panelClass} flex items-center justify-center min-h-[600px] overflow-hidden relative sticky top-24 self-start ${className}`}>
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
            onClick={() => openFeedbackForm({ feedbackType: 'Bug / Error', capability: 'Product Fitting', errorMessage: error })}
            className="text-sm text-gm-accent hover:text-gm-accent-hover transition-colors underline underline-offset-2"
          >
            Report this issue
          </button>
        </div>
      </div>
    )
  }

  // Loading state
  if (isLoading) {
    return (
      <div className={`${panelClass} flex items-center justify-center min-h-[600px] overflow-hidden relative sticky top-24 self-start ${className}`}>
        {showVideo && <BackgroundVideo opacity="opacity-30" />}
        <div className="absolute inset-0 bg-gradient-to-t from-gm-bg-light/80 via-gm-bg-light/40 to-transparent dark:from-gm-bg/80 dark:via-gm-bg/40"></div>

        <div className="text-center px-8 relative z-10">
          <div className="relative w-20 h-20 mx-auto mb-6">
            <div className="absolute inset-0 rounded-full border-4 border-gm-accent/20"></div>
            <div className="absolute inset-0 rounded-full border-4 border-transparent border-t-gm-accent animate-spin"></div>
            <div className="absolute inset-0 flex items-center justify-center">
              <svg width="28" height="28" viewBox="0 0 24 24" fill="none" className="text-gm-accent">
                <path d="M20 21v-2a4 4 0 00-4-4H8a4 4 0 00-4 4v2" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                <circle cx="12" cy="7" r="4" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
              </svg>
            </div>
          </div>
          <h2 className="text-2xl font-bold mb-2">
            Generating <span className="text-gm-accent">Product Fitting</span>
          </h2>
          <p className="text-sm text-gm-text-secondary-light dark:text-gm-text-secondary">
            Fitting garments onto model and evaluating results...
          </p>
        </div>
      </div>
    )
  }

  // Results state
  if (result && (result.front || result.back)) {
    const sides = [
      result.front ? { label: 'Front', ...result.front } : null,
      result.back ? { label: 'Back', ...result.back } : null,
    ].filter(Boolean) as Array<{ label: string; imageUrl: string; status: string; totalAttempts: number; validation: FittingValidation }>

    return (
      <div className={`${panelClass} flex flex-col min-h-[600px] overflow-hidden relative sticky top-24 self-start ${className}`}>
        <div className="flex-1 p-4 overflow-y-auto flex flex-col gap-4">
          {/* 1. Video Playback Section (Shown prominently at the top when ready) */}
          {videoResult && videoResult.videos && videoResult.videos.length > 0 && videoUrl && (
            <div className="flex flex-col items-center gap-3 bg-black/[0.02] dark:bg-white/[0.02] p-4 rounded-xl border border-black/[0.05] dark:border-white/[0.05]">
              <span className="text-[11px] font-semibold text-gm-accent uppercase tracking-widest">
                Video Try-On Result
              </span>
              <div className="w-full max-w-[280px] rounded-lg overflow-hidden bg-black/5 dark:bg-black/20 flex justify-center shadow-md">
                <video
                  src={videoUrl}
                  controls
                  autoPlay
                  loop
                  playsInline
                  className="w-full aspect-[9/16] object-cover"
                />
              </div>
              <button
                onClick={onAnimate}
                disabled={isVideoLoading}
                className="btn-primary w-full max-w-[280px] py-3 rounded-xl text-button font-medium flex items-center justify-center gap-2 disabled:opacity-50 mt-1"
              >
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M21.5 2v6h-6M21.34 15.57a10 10 0 1 1-.57-8.38l5.67-5.67"/>
                </svg>
                {isVideoLoading ? 'Regenerating...' : 'Regenerate Video'}
              </button>
            </div>
          )}

          {/* 2. Original Reference Images (Shown below the video) */}
          <div>
            {videoResult && (
              <div className="text-center mb-3">
                <span className="text-[11px] font-medium text-gm-text-secondary-light dark:text-gm-text-secondary uppercase tracking-wider">
                  Generated Reference Views
                </span>
              </div>
            )}
            <div className={`grid ${sides.length === 2 ? 'grid-cols-2' : 'grid-cols-1'} ${videoResult ? 'gap-2 max-w-[280px]' : 'gap-4 max-w-[420px]'} mx-auto transition-all`}>
              {sides.map((side) => (
                <div 
                  key={side.label} 
                  onClick={() => setSelectedImage(side.imageUrl)}
                  className="relative rounded-lg overflow-hidden bg-black/5 dark:bg-black/20 cursor-pointer group transition-transform hover:scale-[1.02]"
                >
                  <img
                    src={side.imageUrl}
                    alt={`${side.label} fitting result`}
                    className="w-full aspect-[3/4] object-cover"
                  />
                  <div className="absolute inset-0 bg-black/20 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center pointer-events-none">
                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" className="text-white">
                      <path d="M15 3h6v6M9 21H3v-6M21 3l-7 7M3 21l7-7" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                    </svg>
                  </div>
                  <div className="absolute bottom-2 left-2 flex gap-1">
                    <span className="px-2 py-0.5 rounded-full bg-black/50 backdrop-blur-sm text-white text-[10px] font-medium">
                      {side.label}
                    </span>
                    {side.validation?.garments_score !== undefined && (
                      <span className="px-2 py-0.5 rounded-full bg-black/50 backdrop-blur-sm text-white text-[10px] font-medium">
                        {side.validation.garments_score}%
                      </span>
                    )}
                  </div>
                  {side.status === 'discarded' && (
                    <div className="absolute top-2 right-2 px-2 py-0.5 rounded-full bg-yellow-500/80 backdrop-blur-sm text-white text-[10px] font-medium">
                      Best effort
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>

          {/* 3. Initial Loading or Generation Status */}
          {!videoResult && (
            <div className="mt-2 border-t border-black/[0.08] dark:border-white/10 pt-4">
              {!isVideoLoading && (
                <button
                  onClick={onAnimate}
                  className="btn-primary w-full max-w-[240px] mx-auto py-2.5 rounded-lg text-button font-medium flex items-center justify-center gap-2 shadow-md"
                >
                  <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
                    <path d="M5 3l14 9-14 9V3z" fill="currentColor"/>
                  </svg>
                  Animate Fitting
                </button>
              )}

              {isVideoLoading && (
                <div className="text-center py-4">
                  <div className="inline-block w-6 h-6 border-2 border-gm-accent/20 border-t-gm-accent rounded-full animate-spin mb-2"></div>
                  <p className="text-sm text-gm-text-secondary-light dark:text-gm-text-secondary">
                    Generating video (this may take a while)...
                  </p>
                </div>
              )}

              {videoError && (
                <p className="text-sm text-red-500 mt-2 text-center">{videoError}</p>
              )}
            </div>
          )}

          {/* 4. Lightbox Overlay for Preview */}
          {selectedImage && (
            <div 
              className="fixed inset-0 bg-black/80 backdrop-blur-md z-[9999] flex items-center justify-center p-4 animate-fade-in"
              onClick={() => setSelectedImage(null)}
            >
              <div 
                className="relative max-w-3xl max-h-[90vh] bg-black/40 border border-white/10 rounded-2xl p-2 overflow-hidden shadow-2xl"
                onClick={(e) => e.stopPropagation()}
              >
                <button
                  onClick={() => setSelectedImage(null)}
                  className="absolute top-4 right-4 w-8 h-8 bg-black/50 hover:bg-black/80 text-white rounded-full flex items-center justify-center backdrop-blur-sm transition-colors z-10"
                >
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M18 6L6 18M6 6L18 18" />
                  </svg>
                </button>
                <img 
                  src={selectedImage} 
                  alt="Detailed reference view" 
                  className="w-full h-full max-h-[85vh] object-contain rounded-xl"
                />
              </div>
            </div>
          )}
        </div>
      </div>
    )
  }

  // Default / empty state
  return (
    <div className={`${panelClass} flex items-center justify-center min-h-[600px] overflow-hidden relative sticky top-24 self-start ${className}`}>
      {showVideo && <BackgroundVideo />}
      <div className="absolute inset-0 bg-gradient-to-t from-gm-bg-light/50 via-gm-bg-light/20 to-transparent dark:from-gm-bg/50 dark:via-gm-bg/20"></div>

      <div className="text-center px-8 relative z-10 max-w-lg animate-fade-in-up">
        <div className="w-16 h-16 mx-auto mb-6 rounded-2xl bg-gm-accent/10 dark:bg-gm-accent/20 flex items-center justify-center glow-accent">
          <svg width="28" height="28" viewBox="0 0 24 24" fill="none" className="text-gm-accent">
            <path d="M20 21v-2a4 4 0 00-4-4H8a4 4 0 00-4 4v2" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
            <circle cx="12" cy="7" r="4" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
          </svg>
        </div>

        <h2 className="text-[40px] lg:text-[48px] font-display font-bold mb-4 leading-tight tracking-tight">
          Product{' '}
          <span className="text-gm-accent" style={{ textShadow: '0 0 40px rgba(138, 180, 248, 0.4), 0 0 80px rgba(138, 180, 248, 0.2)' }}>
            Fitting
          </span>
        </h2>
        <p className="text-body-lg text-gm-text-secondary-light dark:text-gm-text-secondary leading-relaxed">
          Upload garment images to generate AI-powered product fitting on models.
        </p>

        <div className="flex flex-wrap items-center justify-center gap-2 mt-6">
          <span className="px-3 py-1.5 rounded-full bg-black/[0.08] dark:bg-white/[0.12] text-caption text-gm-text-secondary-light dark:text-gm-text-secondary">
            Gemini
          </span>
          <span className="px-3 py-1.5 rounded-full bg-black/[0.08] dark:bg-white/[0.12] text-caption text-gm-text-secondary-light dark:text-gm-text-secondary">
            AI Fitting
          </span>
          <span className="px-3 py-1.5 rounded-full bg-black/[0.08] dark:bg-white/[0.12] text-caption text-gm-text-secondary-light dark:text-gm-text-secondary">
            Multi-View
          </span>
        </div>
      </div>
    </div>
  )
}

export default ProductFittingPreview
