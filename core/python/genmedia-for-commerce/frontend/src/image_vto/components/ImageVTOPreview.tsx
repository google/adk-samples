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
import type { VTOResult } from '../services/imageVtoApi'
import BackgroundVideo from '../../components/BackgroundVideo'

interface ImageVTOPreviewProps {
  className?: string
  showVideo?: boolean
  results: VTOResult[]
  isLoading: boolean
  numVariations?: number
  mode?: 'clothes' | 'glasses'
}

function ImageVTOPreview({
  className = '',
  showVideo = true,
  results,
  isLoading,
  numVariations = 4,
  mode = 'clothes',
}: ImageVTOPreviewProps) {
  const [selectedImageIndex, setSelectedImageIndex] = useState(0)
  const [enlargedImage, setEnlargedImage] = useState<string | null>(null)

  // Keyboard support for closing enlarged image modal
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && enlargedImage) {
        setEnlargedImage(null)
      }
    }

    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [enlargedImage])

  const panelClass = showVideo
    ? 'glass-panel rounded-lg'
    : 'border border-black/[0.08] dark:border-white/10 rounded-lg'

  // Show loading state with video background when loading and no results yet
  if (isLoading && results.length === 0) {
    return (
      <div className={`${panelClass} flex items-center justify-center min-h-[600px] overflow-hidden relative ${className}`}>
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
            Generating <span className="text-gm-accent">Virtual Try-On</span>
          </h2>
          <p className="text-sm text-gm-text-secondary-light dark:text-gm-text-secondary">
            Processing your images and creating variations...
          </p>
          <p className="text-xs text-gm-text-tertiary-light dark:text-gm-text-tertiary mt-2">
            This may take a few minutes
          </p>
        </div>
      </div>
    )
  }

  // Show empty state when no results and not loading
  if (!isLoading && results.length === 0) {
    return (
      <div className={`${panelClass} flex items-center justify-center min-h-[600px] overflow-hidden relative ${className}`}>
        {showVideo && <BackgroundVideo />}

        <div className="absolute inset-0 bg-gradient-to-t from-gm-bg-light/50 via-gm-bg-light/20 to-transparent dark:from-gm-bg/50 dark:via-gm-bg/20"></div>

        <div className="text-center px-8 relative z-10 max-w-lg animate-fade-in-up">
          <div className="w-16 h-16 mx-auto mb-6 rounded-2xl bg-gm-accent/10 dark:bg-gm-accent/20 flex items-center justify-center glow-accent">
            <svg width="28" height="28" viewBox="0 0 24 24" fill="none" className="text-gm-accent">
              <path d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" stroke="currentColor" strokeWidth="1.5" />
              <path d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" stroke="currentColor" strokeWidth="1.5" />
            </svg>
          </div>

          <h2 className="text-[40px] lg:text-[48px] font-display font-bold mb-4 leading-tight tracking-tight">
            {mode === 'glasses' ? 'Glasses' : 'Clothes'}{' '}
            <span className="text-gm-accent" style={{ textShadow: '0 0 40px rgba(138, 180, 248, 0.4), 0 0 80px rgba(138, 180, 248, 0.2)' }}>
              Image VTO
            </span>
          </h2>
          <p className="text-body-lg text-gm-text-secondary-light dark:text-gm-text-secondary leading-relaxed">
            {mode === 'glasses'
              ? 'Upload a face photo and eyewear to generate photorealistic virtual try-on images.'
              : 'Upload a model and garments to generate photorealistic virtual try-on images.'}
          </p>

          <div className="flex flex-wrap items-center justify-center gap-2 mt-6">
            <span className="px-3 py-1.5 rounded-full bg-black/[0.08] dark:bg-white/[0.12] text-caption text-gm-text-secondary-light dark:text-gm-text-secondary">
              {mode === 'glasses' ? '🍌 Nano Banana' : 'Dress Try-On'}
            </span>
            <span className="px-3 py-1.5 rounded-full bg-black/[0.08] dark:bg-white/[0.12] text-caption text-gm-text-secondary-light dark:text-gm-text-secondary">
              Face Consistency
            </span>
            <span className="px-3 py-1.5 rounded-full bg-black/[0.08] dark:bg-white/[0.12] text-caption text-gm-text-secondary-light dark:text-gm-text-secondary">
              {mode === 'glasses' ? 'Eyewear Fitting' : 'Automatic Evaluation'}
            </span>
          </div>
        </div>
      </div>
    )
  }

  // Show results - sort by similarity percentage (highest first)
  const sortedResults = results
    .filter(r => r.status === 'ready' && r.imageUrl)
    .sort((a, b) => {
      const aScore = a.final_score ?? a.evaluation?.similarity_percentage ?? 0
      const bScore = b.final_score ?? b.evaluation?.similarity_percentage ?? 0
      return bScore - aScore // Highest first
    })

  return (
    <div className={`${panelClass} min-h-[600px] overflow-hidden relative ${className}`}>
      {showVideo && results.length === 0 && <BackgroundVideo />}

      {results.length === 0 ? (
        <>
          <div className="absolute inset-0 bg-gradient-to-t from-gm-bg-light/50 via-gm-bg-light/20 to-transparent dark:from-gm-bg/50 dark:via-gm-bg/20"></div>
          <div className="absolute inset-0 flex items-center justify-center">
            <div className="text-center px-8 relative z-10 max-w-lg">
              {isLoading ? (
                /* Generating state */
                <>
                  {/* Animated rings */}
                  <div className="relative mb-6 h-16 flex items-center justify-center">
                    <div className="w-16 h-16 rounded-full border-2 border-gm-accent/30 animate-ping absolute" />
                    <div className="w-16 h-16 rounded-full border-2 border-gm-accent/20 animate-pulse" />
                    <div className="absolute inset-0 flex items-center justify-center">
                      <div className="w-3 h-3 rounded-full bg-gm-accent animate-pulse" />
                    </div>
                  </div>

                  <h2 className="text-[32px] lg:text-[40px] font-bold mb-3 leading-tight">
                    Generating your{' '}
                    <span
                      className="text-gm-accent"
                      style={{
                        textShadow: '0 0 40px rgba(138, 180, 248, 0.4), 0 0 80px rgba(138, 180, 248, 0.2)'
                      }}
                    >
                      try-on
                    </span>...
                  </h2>
                  <p className="text-lg text-gm-text-secondary-light dark:text-gm-text-secondary leading-relaxed">
                    Our AI is creating your virtual try-on. This may take a moment.
                  </p>
                </>
              ) : (
                /* Empty state */
                <>
                  <h2 className="text-[48px] font-bold mb-4 leading-tight">
                    Ready to <span className="text-gm-accent" style={{ textShadow: '0 0 30px rgba(138, 180, 248, 0.5), 0 0 60px rgba(138, 180, 248, 0.3)' }}>Create</span>
                  </h2>
                  <p className="text-lg text-gm-text-secondary-light dark:text-gm-text-secondary">
                    Configure your settings on the left to create your virtual try-on.
                  </p>
                </>
              )}
            </div>
          </div>
        </>
      ) : (
        <div className="p-6 h-full flex flex-col">
          {(() => {
            const currentImage = sortedResults[selectedImageIndex]

            return (
              <>
                {/* Main Image Display - Constrained but not cropped */}
                <div className="flex-1 flex items-center justify-center mb-4 bg-black/5 dark:bg-white/5 rounded-lg overflow-hidden max-h-[calc(100vh-300px)] min-h-[500px]">
                  {currentImage ? (
                    <div
                      className="relative w-full h-full flex items-center justify-center cursor-pointer group p-2"
                      onClick={() => setEnlargedImage(currentImage.imageUrl!)}
                    >
                      <img
                        src={currentImage.imageUrl}
                        alt={`Generated VTO ${selectedImageIndex + 1}`}
                        className="max-w-full max-h-full object-contain transition-transform duration-300 group-hover:scale-[1.02]"
                      />
                      <div className="absolute inset-0 bg-black/0 group-hover:bg-black/20 transition-all duration-300 flex items-center justify-center">
                        <div className="opacity-0 group-hover:opacity-100 transition-opacity duration-300 bg-white/90 dark:bg-black/90 rounded-full p-3">
                          <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
                            <path d="M15 3h6v6M9 21H3v-6M21 3l-7 7M3 21l7-7" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                          </svg>
                        </div>
                      </div>
                      <div className="absolute top-4 right-4 flex items-center gap-2">
                        <button
                          onClick={(e) => {
                            e.stopPropagation()
                            if (currentImage.imageUrl) {
                              const link = document.createElement('a')
                              link.href = currentImage.imageUrl
                              link.download = `vto-result-${selectedImageIndex + 1}.png`
                              document.body.appendChild(link)
                              link.click()
                              document.body.removeChild(link)
                            }
                          }}
                          className="w-10 h-10 bg-black/60 hover:bg-black/80 backdrop-blur-sm rounded-xl flex items-center justify-center transition-all"
                          title="Download image"
                        >
                          <svg width="18" height="18" viewBox="0 0 24 24" fill="none">
                            <path d="M12 15V3M12 15L8 11M12 15L16 11" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                            <path d="M2 17L2 19C2 20.1046 2.89543 21 4 21L20 21C21.1046 21 22 20.1046 22 19V17" stroke="white" strokeWidth="2" strokeLinecap="round" />
                          </svg>
                        </button>
                        {currentImage.face_score != null && currentImage.glasses_score != null ? (
                          <div className="flex items-center gap-1.5">
                            <div className="glass-panel px-3 py-1.5 rounded-lg">
                              <div className={`text-caption font-medium ${
                                currentImage.face_score >= 75
                                  ? 'text-green-600 dark:text-green-400'
                                  : 'text-gm-accent'
                              }`}>
                                Face: {currentImage.face_score.toFixed(1)}%
                              </div>
                            </div>
                            <div className="glass-panel px-3 py-1.5 rounded-lg">
                              <div className={`text-caption font-medium ${
                                currentImage.glasses_score >= 75
                                  ? 'text-green-600 dark:text-green-400'
                                  : 'text-gm-accent'
                              }`}>
                                Glasses: {currentImage.glasses_score.toFixed(1)}%
                              </div>
                            </div>
                          </div>
                        ) : currentImage.face_score != null && currentImage.garments_score != null ? (
                          <div className="flex items-center gap-1.5">
                            <div className="glass-panel px-3 py-1.5 rounded-lg">
                              <div className={`text-caption font-medium ${
                                currentImage.face_score >= 75
                                  ? 'text-green-600 dark:text-green-400'
                                  : 'text-gm-accent'
                              }`}>
                                Face: {currentImage.face_score.toFixed(1)}%
                              </div>
                            </div>
                            <div className="glass-panel px-3 py-1.5 rounded-lg">
                              <div className={`text-caption font-medium ${
                                currentImage.garments_score >= 75
                                  ? 'text-green-600 dark:text-green-400'
                                  : 'text-gm-accent'
                              }`}>
                                Garment: {currentImage.garments_score.toFixed(1)}%
                              </div>
                            </div>
                          </div>
                        ) : (currentImage.final_score != null || (currentImage.evaluation && currentImage.evaluation.similarity_percentage != null)) && (
                          <div className="glass-panel px-3 py-1.5 rounded-lg">
                            <div className={`text-caption font-medium ${
                              (currentImage.final_score ?? currentImage.evaluation?.similarity_percentage ?? 0) >= 75
                                ? 'text-green-600 dark:text-green-400'
                                : 'text-gm-accent'
                            }`}>
                              {currentImage.final_score != null
                                ? `${currentImage.final_score.toFixed(1)}% Score`
                                : `${currentImage.evaluation!.similarity_percentage!.toFixed(1)}% Match`}
                            </div>
                          </div>
                        )}
                      </div>
                      {/* Best Match Badge */}
                      {selectedImageIndex === 0 && sortedResults.length > 1 && (
                        <div className="absolute top-4 left-4 glass-panel px-3 py-1.5 rounded-lg">
                          <div className="text-caption font-medium text-green-600 dark:text-green-400 flex items-center gap-1.5">
                            <svg width="12" height="12" viewBox="0 0 24 24" fill="none">
                              <path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z" fill="currentColor"/>
                            </svg>
                            Best Match
                          </div>
                        </div>
                      )}
                    </div>
                  ) : (
                    <div className="flex items-center justify-center">
                      <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-gm-accent"></div>
                    </div>
                  )}
                </div>

                {/* Thumbnail Grid */}
                <div className="grid grid-cols-4 gap-3">
                  {Array.from({ length: numVariations }).map((_, idx) => {
                    const sortedItem = sortedResults[idx]
                    return (
                      <button
                        key={idx}
                        onClick={() => sortedItem && setSelectedImageIndex(idx)}
                        className={`aspect-square rounded-lg overflow-hidden border-2 transition-all relative ${
                          selectedImageIndex === idx
                            ? 'border-gm-accent shadow-lg'
                            : 'border-transparent hover:border-gm-accent/50'
                        } ${sortedItem ? 'bg-black/5 dark:bg-white/5' : 'bg-black/[0.02] dark:bg-white/[0.02]'}`}
                      >
                        {sortedItem ? (
                          <>
                            <img
                              src={sortedItem.imageUrl}
                              alt={`Thumbnail ${idx + 1}`}
                              className="w-full h-full object-contain p-1"
                            />
                            {(sortedItem.final_score != null || (sortedItem.evaluation && sortedItem.evaluation.similarity_percentage != null)) && (
                              <div className={`absolute bottom-1 right-1 px-1.5 py-0.5 rounded text-[10px] font-bold ${
                                (sortedItem.final_score ?? sortedItem.evaluation?.similarity_percentage ?? 0) >= 75
                                  ? 'bg-green-600 text-white'
                                  : 'bg-gm-accent/90 text-white'
                              }`}>
                                {(sortedItem.final_score ?? sortedItem.evaluation?.similarity_percentage ?? 0).toFixed(0)}%
                              </div>
                            )}
                          </>
                        ) : (
                          <div className="w-full h-full flex items-center justify-center">
                            {isLoading && (
                              <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-gm-accent"></div>
                            )}
                          </div>
                        )}
                      </button>
                    )
                  })}
                </div>
              </>
            )
          })()}
        </div>
      )}

      {/* Enlarged Image Modal */}
      {enlargedImage && (
        <div
          className="fixed inset-0 z-[9999] flex items-center justify-center bg-black/80 backdrop-blur-sm animate-fade-in p-4"
          onClick={() => setEnlargedImage(null)}
        >
          <div className="relative">
            <button
              onClick={() => setEnlargedImage(null)}
              className="absolute -top-4 -right-4 w-12 h-12 bg-white dark:bg-black rounded-full flex items-center justify-center shadow-lg hover:scale-110 transition-transform z-10"
            >
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
                <path d="M18 6L6 18M6 6L18 18" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
              </svg>
            </button>
            <img
              src={enlargedImage}
              alt="Enlarged view"
              className="max-w-[95vw] max-h-[95vh] object-contain rounded-lg shadow-2xl"
              onClick={(e) => e.stopPropagation()}
            />
          </div>
        </div>
      )}
    </div>
  )
}

export default ImageVTOPreview
