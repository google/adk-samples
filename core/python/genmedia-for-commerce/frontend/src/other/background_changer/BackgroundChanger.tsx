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

import { useState, useRef, useEffect } from 'react'
import PromptTextarea from '../../components/PromptTextarea'
import BackgroundVideo from '../../components/BackgroundVideo'
import { changeBackground, base64ToDataUrl, dataUrlToBlob } from '../services/backgroundChangerApi'
import type { BackgroundChangeResult } from '../services/backgroundChangerApi'

interface BackgroundChangerProps {
  showVideo?: boolean
  embedMode?: boolean
}

interface GeneratedImage {
  dataUrl: string
  evaluation: {
    similarity_percentage: number
    face_detected: boolean
  }
}

function BackgroundChanger({ showVideo = true, embedMode = false }: BackgroundChangerProps) {
  const [uploadedImage, setUploadedImage] = useState<string | null>(null)
  const [backgroundMode, setBackgroundMode] = useState<'prompt' | 'upload'>('prompt')
  const [backgroundDescription, setBackgroundDescription] = useState('')
  const [backgroundImage, setBackgroundImage] = useState<string | null>(null)
  const [isGenerating, setIsGenerating] = useState(false)
  const [generatedImages, setGeneratedImages] = useState<GeneratedImage[]>([])
  const [error, setError] = useState<string | null>(null)
  const [selectedImageIndex, setSelectedImageIndex] = useState(0)
  const [enlargedImage, setEnlargedImage] = useState<string | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const backgroundImageInputRef = useRef<HTMLInputElement>(null)

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      const reader = new FileReader()
      reader.onloadend = () => {
        setUploadedImage(reader.result as string)
      }
      reader.readAsDataURL(file)
    }
  }

  const handleRemoveFile = () => {
    setUploadedImage(null)
    if (fileInputRef.current) {
      fileInputRef.current.value = ''
    }
  }

  const handleBackgroundImageSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      const reader = new FileReader()
      reader.onloadend = () => {
        setBackgroundImage(reader.result as string)
      }
      reader.readAsDataURL(file)
    }
  }

  const handleRemoveBackgroundImage = () => {
    setBackgroundImage(null)
    if (backgroundImageInputRef.current) {
      backgroundImageInputRef.current.value = ''
    }
  }

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

  const handleGenerate = async () => {
    if (!uploadedImage) {
      setError('Please upload a person image first')
      return
    }

    if (backgroundMode === 'prompt' && !backgroundDescription.trim()) {
      setError('Please enter a background description')
      return
    }

    if (backgroundMode === 'upload' && !backgroundImage) {
      setError('Please upload a background image')
      return
    }

    setIsGenerating(true)
    setError(null)
    setGeneratedImages([])
    setSelectedImageIndex(0)

    try {
      const personImageBlob = dataUrlToBlob(uploadedImage)
      
      interface RequestData {
        personImage: Blob;
        numVariations: number;
        backgroundDescription?: string;
        backgroundImage?: Blob;
      }

      const requestData: RequestData = {
        personImage: personImageBlob,
        numVariations: 4,
      }

      if (backgroundMode === 'prompt') {
        requestData.backgroundDescription = backgroundDescription
      } else {
        requestData.backgroundImage = dataUrlToBlob(backgroundImage!)
      }

      await changeBackground(
        requestData,
        (result: BackgroundChangeResult) => {
          if (result.status === 'ready' && result.image_base64) {
            const dataUrl = base64ToDataUrl(result.image_base64)
            setGeneratedImages((prev) => {
              const newImages = [...prev]
              newImages[result.index] = {
                dataUrl,
                evaluation: result.evaluation || {
                  similarity_percentage: 0,
                  face_detected: false,
                },
              }

              // Auto-select the first image that arrives
              if (prev.length === 0 || !prev[selectedImageIndex]) {
                setSelectedImageIndex(result.index)
              }

              return newImages
            })
          } else if (result.status === 'failed') {
            console.error(`Variation ${result.index} failed:`, result.error)
          }
        },
        () => {
          setIsGenerating(false)
        },
        (errorMessage: string) => {
          setError(errorMessage)
          setIsGenerating(false)
        }
      )
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred')
      setIsGenerating(false)
    }
  }

  // Settings content (used in both standalone and embed modes)
  const settingsContent = (
    <>
        {/* Upload Image */}
        <div className="mb-8">
          <h3 className="text-h3 font-bold mb-2">Upload Image</h3>
          <p className="text-body text-gm-text-secondary-light dark:text-gm-text-secondary mb-4">
            Upload the image you want to change the background for.
          </p>

          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            onChange={handleFileSelect}
            className="hidden"
          />

          {uploadedImage ? (
            <div className="relative rounded-lg overflow-hidden bg-black/5 dark:bg-white/5 h-32 flex items-center justify-center">
              <img
                src={uploadedImage}
                alt="Person image"
                className="max-w-full max-h-full object-contain"
              />
              <button
                onClick={handleRemoveFile}
                className="absolute top-2 right-2 w-6 h-6 bg-black/70 hover:bg-black/90 rounded-full flex items-center justify-center transition-all"
              >
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none">
                  <path d="M18 6L6 18M6 6L18 18" stroke="white" strokeWidth="2" strokeLinecap="round"/>
                </svg>
              </button>
            </div>
          ) : (
            <div
              onClick={() => fileInputRef.current?.click()}
              className="upload-zone h-32 flex flex-col items-center justify-center cursor-pointer transition-all duration-300"
            >
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" className="mb-2">
                <path d="M12 15V3M12 3L8 7M12 3L16 7" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
                <path d="M2 17L2 19C2 20.1046 2.89543 21 4 21L20 21C21.1046 21 22 20.1046 22 19V17" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
              </svg>
              <p className="text-[10px] font-medium mb-1">CLICK TO UPLOAD OR DROP FILE HERE</p>
            </div>
          )}
        </div>

        {/* Background Input Method Toggle */}
        <div className="mb-8">
          <h3 className="text-h3 font-bold mb-2">Background</h3>
          <p className="text-body text-gm-text-secondary-light dark:text-gm-text-secondary mb-4">
            Choose how to define the new background for your product.
          </p>

          {/* Toggle Switch */}
          <div className="flex items-center gap-1 p-1 rounded-xl bg-black/[0.03] dark:bg-white/[0.04] glass-panel inline-flex mb-6">
            <button
              onClick={() => setBackgroundMode('prompt')}
              className={`px-4 py-2.5 rounded-lg text-button font-medium transition-all duration-200 flex items-center gap-2 ${
                backgroundMode === 'prompt'
                  ? 'bg-white dark:bg-white/10 shadow-sm text-gm-text-primary-light dark:text-gm-text-primary'
                  : 'text-gm-text-tertiary-light dark:text-gm-text-tertiary hover:text-gm-text-primary-light dark:hover:text-gm-text-primary'
              }`}
            >
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" className="opacity-70">
                <path d="M12 2L14.5 9.5L22 12L14.5 14.5L12 22L9.5 14.5L2 12L9.5 9.5L12 2Z" fill="currentColor" opacity="0.6"/>
                <path d="M19 4L20 7L23 8L20 9L19 12L18 9L15 8L18 7L19 4Z" fill="currentColor" opacity="0.8"/>
              </svg>
              Describe Background
            </button>
            <button
              onClick={() => setBackgroundMode('upload')}
              className={`px-4 py-2.5 rounded-lg text-button font-medium transition-all duration-200 flex items-center gap-2 ${
                backgroundMode === 'upload'
                  ? 'bg-white dark:bg-white/10 shadow-sm text-gm-text-primary-light dark:text-gm-text-primary'
                  : 'text-gm-text-tertiary-light dark:text-gm-text-tertiary hover:text-gm-text-primary-light dark:hover:text-gm-text-primary'
              }`}
            >
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" className="opacity-70">
                <rect x="2" y="2" width="20" height="20" rx="3" stroke="currentColor" strokeWidth="1.5" opacity="0.3"/>
                <path d="M2 14l4-4 4 4 6-6 6 6" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
                <circle cx="7" cy="7" r="1.5" fill="currentColor"/>
              </svg>
              Upload Background
            </button>
          </div>

          {/* Conditional Input - Prompt Mode */}
          {backgroundMode === 'prompt' && (
            <div className="animate-fade-in">
              <PromptTextarea
                value={backgroundDescription}
                onChange={setBackgroundDescription}
                placeholder="A clean white studio background with soft lighting, or a modern kitchen setting, or a natural outdoor environment..."
              />
            </div>
          )}

          {/* Conditional Input - Upload Mode */}
          {backgroundMode === 'upload' && (
            <div className="animate-fade-in">
              <input
                ref={backgroundImageInputRef}
                type="file"
                accept="image/*"
                onChange={handleBackgroundImageSelect}
                className="hidden"
              />

              {backgroundImage ? (
                <div className="relative rounded-lg overflow-hidden bg-black/5 dark:bg-white/5 h-32 flex items-center justify-center">
                  <img
                    src={backgroundImage}
                    alt="Background image"
                    className="max-w-full max-h-full object-contain"
                  />
                  <button
                    onClick={handleRemoveBackgroundImage}
                    className="absolute top-2 right-2 w-6 h-6 bg-black/70 hover:bg-black/90 rounded-full flex items-center justify-center transition-all"
                  >
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none">
                      <path d="M18 6L6 18M6 6L18 18" stroke="white" strokeWidth="2" strokeLinecap="round"/>
                    </svg>
                  </button>
                </div>
              ) : (
                <div
                  onClick={() => backgroundImageInputRef.current?.click()}
                  className="upload-zone h-32 flex flex-col items-center justify-center cursor-pointer transition-all duration-300"
                >
                  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" className="mb-2">
                    <path d="M12 15V3M12 3L8 7M12 3L16 7" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
                    <path d="M2 17L2 19C2 20.1046 2.89543 21 4 21L20 21C21.1046 21 22 20.1046 22 19V17" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
                  </svg>
                  <p className="text-[10px] font-medium mb-1">CLICK TO UPLOAD OR DROP FILE HERE</p>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Error Message */}
        {error && (
          <div className="mb-6 p-4 rounded-lg bg-red-500/10 border border-red-500/20 text-red-600 dark:text-red-400">
            {error}
          </div>
        )}

        {/* Generate Button */}
        <div className="flex justify-end">
          <button
            onClick={handleGenerate}
            disabled={
              isGenerating ||
              !uploadedImage ||
              (backgroundMode === 'prompt' && !backgroundDescription.trim()) ||
              (backgroundMode === 'upload' && !backgroundImage)
            }
            className="btn-primary px-8 py-3 rounded-pill text-button font-medium disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {isGenerating ? 'Generating...' : 'Generate Background'}
          </button>
        </div>
      </>
  )

  // In embed mode, just return the settings content
  if (embedMode) {
    return settingsContent
  }

  // In standalone mode, return full layout with preview panel
  return (
    <div className="grid grid-cols-[480px_1fr] gap-6">
      {/* Left Panel - Settings */}
      <div className="glass-panel rounded-lg p-6 h-fit">
        {settingsContent}
      </div>

      {/* Right Panel - Preview/Results */}
      <div className={`${showVideo && generatedImages.length === 0 ? 'glass-panel' : 'border border-black/[0.08] dark:border-white/10'} rounded-lg min-h-[600px] overflow-hidden relative`}>
        {showVideo && generatedImages.length === 0 && <BackgroundVideo />}

        {generatedImages.length === 0 ? (
          <>
            <div className="absolute inset-0 bg-gradient-to-t from-gm-bg-light/50 via-gm-bg-light/20 to-transparent dark:from-gm-bg/50 dark:via-gm-bg/20"></div>
            <div className="absolute inset-0 flex items-center justify-center">
              <div className="text-center px-8 relative z-10 max-w-lg">
                {isGenerating ? (
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
                        backgrounds
                      </span>...
                    </h2>
                    <p className="text-lg text-gm-text-secondary-light dark:text-gm-text-secondary leading-relaxed">
                      Our AI is transforming your image. This may take a moment.
                    </p>
                  </>
                ) : (
                  /* Empty state */
                  <div className="max-w-lg animate-fade-in-up">
                    <div className="w-16 h-16 mx-auto mb-6 rounded-2xl bg-gm-accent/10 dark:bg-gm-accent/20 flex items-center justify-center glow-accent">
                      <svg width="28" height="28" viewBox="0 0 24 24" fill="none" className="text-gm-accent">
                        <rect x="3" y="3" width="18" height="18" rx="2" stroke="currentColor" strokeWidth="1.5" />
                        <path d="M3 16l5-5 4 4 4-6 5 7" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
                      </svg>
                    </div>

                    <h2 className="text-[40px] lg:text-[48px] font-display font-bold mb-4 leading-tight tracking-tight">
                      Background{' '}
                      <span className="text-gm-accent" style={{ textShadow: '0 0 40px rgba(138, 180, 248, 0.4), 0 0 80px rgba(138, 180, 248, 0.2)' }}>
                        Changer
                      </span>
                    </h2>
                    <p className="text-body-lg text-gm-text-secondary-light dark:text-gm-text-secondary leading-relaxed">
                      Upload a product photo and describe a new scene to place it in a different background.
                    </p>

                    <div className="flex flex-wrap items-center justify-center gap-2 mt-6">
                      <span className="px-3 py-1.5 rounded-full bg-black/[0.08] dark:bg-white/[0.12] text-caption text-gm-text-secondary-light dark:text-gm-text-secondary">
                        🍌 Nano Banana
                      </span>
                      <span className="px-3 py-1.5 rounded-full bg-black/[0.08] dark:bg-white/[0.12] text-caption text-gm-text-secondary-light dark:text-gm-text-secondary">
                        Scene Generation
                      </span>
                      <span className="px-3 py-1.5 rounded-full bg-black/[0.08] dark:bg-white/[0.12] text-caption text-gm-text-secondary-light dark:text-gm-text-secondary">
                        Model Preservation
                      </span>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </>
        ) : (
          <div className="p-6 h-full flex flex-col">
            {/* Sort images by similarity */}
            {(() => {
              const sortedImages = generatedImages
                .map((img, idx) => ({ img, originalIndex: idx }))
                .filter(item => item.img !== undefined)
                .sort((a, b) => {
                  const simA = a.img.evaluation.similarity_percentage || 0
                  const simB = b.img.evaluation.similarity_percentage || 0
                  return simB - simA // Highest first
                })

              const currentImage = sortedImages[selectedImageIndex]

              return (
                <>
                  {/* Main Image Display - Fixed Height */}
                  <div className="h-[480px] flex items-center justify-center mb-4 bg-black/5 dark:bg-white/5 rounded-lg overflow-hidden">
                    {currentImage ? (
                      <div
                        className="relative w-full h-full flex items-center justify-center cursor-pointer group"
                        onClick={() => setEnlargedImage(currentImage.img.dataUrl)}
                      >
                        <img
                          src={currentImage.img.dataUrl}
                          alt={`Generated background ${selectedImageIndex + 1}`}
                          className="max-w-full max-h-full object-contain transition-transform duration-300 group-hover:scale-105"
                        />
                        <div className="absolute inset-0 bg-black/0 group-hover:bg-black/20 transition-all duration-300 flex items-center justify-center">
                          <div className="opacity-0 group-hover:opacity-100 transition-opacity duration-300 bg-white/90 dark:bg-black/90 rounded-full p-3">
                            <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
                              <path d="M15 3h6v6M9 21H3v-6M21 3l-7 7M3 21l7-7" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                            </svg>
                          </div>
                        </div>
                        {currentImage.img.evaluation.face_detected && (
                          <div className="absolute top-4 right-4 glass-panel px-3 py-1.5 rounded-lg">
                            <div className={`text-caption font-medium ${
                              currentImage.img.evaluation.similarity_percentage >= 75
                                ? 'text-green-600 dark:text-green-400'
                                : 'text-gm-accent'
                            }`}>
                              {currentImage.img.evaluation.similarity_percentage.toFixed(1)}% Match
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
                    {Array.from({ length: 4 }).map((_, idx) => {
                      const sortedItem = sortedImages[idx]
                      return (
                        <button
                          key={idx}
                          onClick={() => setSelectedImageIndex(idx)}
                          className={`aspect-square rounded-lg overflow-hidden border-2 transition-all relative ${
                            selectedImageIndex === idx
                              ? 'border-gm-accent shadow-lg'
                              : 'border-transparent hover:border-gm-accent/50'
                          } ${sortedItem ? 'bg-black/5 dark:bg-white/5' : 'bg-black/[0.02] dark:bg-white/[0.02]'}`}
                        >
                          {sortedItem ? (
                            <>
                              <img
                                src={sortedItem.img.dataUrl}
                                alt={`Thumbnail ${idx + 1}`}
                                className="w-full h-full object-contain p-1"
                              />
                              {sortedItem.img.evaluation.face_detected && (
                                <div className={`absolute bottom-1 right-1 px-1.5 py-0.5 rounded text-[10px] font-bold ${
                                  sortedItem.img.evaluation.similarity_percentage >= 75
                                    ? 'bg-green-600 text-white'
                                    : 'bg-gm-accent/90 text-white'
                                }`}>
                                  {sortedItem.img.evaluation.similarity_percentage.toFixed(0)}%
                                </div>
                              )}
                            </>
                          ) : (
                            <div className="w-full h-full flex items-center justify-center">
                              {isGenerating && (
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
      </div>

      {/* Enlarged Image Modal */}
      {enlargedImage && (
        <div
          className="fixed inset-0 z-[60] flex items-center justify-center bg-black/80 backdrop-blur-sm animate-fade-in"
          onClick={() => setEnlargedImage(null)}
        >
          <div className="relative max-w-7xl max-h-[90vh] p-4">
            <button
              onClick={() => setEnlargedImage(null)}
              className="absolute -top-2 -right-2 w-10 h-10 bg-white dark:bg-black rounded-full flex items-center justify-center shadow-lg hover:scale-110 transition-transform z-10"
            >
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
                <path d="M18 6L6 18M6 6L18 18" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
              </svg>
            </button>
            <img
              src={enlargedImage}
              alt="Enlarged view"
              className="max-w-full max-h-[90vh] object-contain rounded-lg shadow-2xl"
              onClick={(e) => e.stopPropagation()}
            />
          </div>
        </div>
      )}
    </div>
  )
}

export default BackgroundChanger
