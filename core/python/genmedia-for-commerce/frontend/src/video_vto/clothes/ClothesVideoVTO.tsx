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

import { useState, useEffect, useCallback, useRef, ChangeEvent } from 'react'
import type { Product } from '../../config/featureConstraints'
import ProductSelector from '../../components/ProductSelector'
import BackgroundVideo from '../../components/BackgroundVideo'
import ImageLightbox from '../../components/ImageLightbox'
import { openFeedbackForm } from '../../components/FeedbackButton'
import { dataUrlToBlob } from '../../image_vto/services/imageVtoApi'
import { MODEL_PRESETS } from '../../shared/modelGallery'
import { WOMEN_GARMENTS, MEN_GARMENTS } from '../../shared/garmentGallery'
import {
  generateVideoVTO,
  generateAnimateModel,
  base64ToVideoUrl,
  base64ToBlob,
  downloadVideoFromUrl,
} from '../services/videoVtoApi'
import type { VideoVTOEvent } from '../services/videoVtoApi'

const proxySrc = (url: string) =>
  url.startsWith('https://storage.cloud.google.com/')
    ? `/api/catalog/image?url=${encodeURIComponent(url)}`
    : url

interface CatalogResult {
  id: string
  description: string
  img_url: string
  gs_uri: string
  category: string
  color: string
  style: string
  audience: string
  score: number
}

interface ClothesVideoVTOProps {
  prefilledImage?: string | null
  prefilledPrompt?: string
  prefilledSubMode?: 'animate-frame' | 'custom-scene'
  prefilledGarmentImages?: string[]
  showVideo?: boolean
  currentProduct?: Product
  availableProducts?: Product[]
  onProductChange?: (product: Product) => void
}

type PipelineStep = 'idle' | 'image-vto' | 'video-generation'

function ClothesVideoVTO({
  prefilledImage,
  prefilledGarmentImages,
  showVideo = true,
  currentProduct,
  availableProducts,
  onProductChange,
}: ClothesVideoVTOProps) {
  // Form state
  const [fullBodyImage, setFullBodyImage] = useState<string | null>(null)
  const [garmentImages, setGarmentImages] = useState<string[]>([])
  const [showGarmentGallery, setShowGarmentGallery] = useState(false)
  const [showModelGallery, setShowModelGallery] = useState(false)
  const [selectedGarments, setSelectedGarments] = useState<string[]>([])
  const [garmentGcsMap, setGarmentGcsMap] = useState<Record<string, string>>({})

  // Catalog search state
  const [searchQuery, setSearchQuery] = useState('')
  const [searchResults, setSearchResults] = useState<CatalogResult[]>([])
  const [isSearching, setIsSearching] = useState(false)
  const debounceTimer = useRef<ReturnType<typeof setTimeout> | null>(null)

  // Pipeline state
  const [pipelineStep, setPipelineStep] = useState<PipelineStep>('idle')
  const [isGenerating, setIsGenerating] = useState(false)
  const [vtoResultImage, setVtoResultImage] = useState<string | null>(null)
  const [bestImageBase64, setBestImageBase64] = useState<string | null>(null)
  const [bestImageScore, setBestImageScore] = useState<number | null>(null)
  const [error, setError] = useState<string | null>(null)

  // Video results
  const [generatedVideos, setGeneratedVideos] = useState<string[]>([])
  // const [reversedVideos, setReversedVideos] = useState<string[]>([])
  const [videoFilenames, setVideoFilenames] = useState<string[]>([])
  const [videoScores, setVideoScores] = useState<number[]>([])
  const [selectedVideoIndex, setSelectedVideoIndex] = useState(0)
  const [showVtoPreview, setShowVtoPreview] = useState(false)
  const [lightboxImage, setLightboxImage] = useState<string | null>(null)

  // All video URLs for cleanup purposes
  const allVideoUrls = [...generatedVideos /*, ...reversedVideos*/]

  const womenModels = MODEL_PRESETS.filter(m => m.gender === 'woman')
  const menModels = MODEL_PRESETS.filter(m => m.gender === 'man')

  // Cleanup blob URLs on unmount
  useEffect(() => {
    return () => {
      allVideoUrls.forEach((url) => URL.revokeObjectURL(url))
    }
  }, [])

  // Prefill from GenMedia TV or other navigation
  useEffect(() => {
    if (prefilledImage) {
      setFullBodyImage(prefilledImage)
    }
    if (prefilledGarmentImages && prefilledGarmentImages.length > 0) {
      setGarmentImages(prefilledGarmentImages)
    }
  }, [prefilledImage, prefilledGarmentImages])

  const performSearch = useCallback(async (query: string) => {
    if (!query.trim()) {
      setSearchResults([])
      return
    }
    setIsSearching(true)
    try {
      const res = await fetch('/api/catalog/search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query }),
      }).then(r => r.json())
      setSearchResults(res.results ?? [])
    } catch (err) {
      console.error('Catalog search failed', err)
      setSearchResults([])
    } finally {
      setIsSearching(false)
    }
  }, [])

  const handleSearchChange = useCallback((e: ChangeEvent<HTMLInputElement>) => {
    const value = e.target.value
    setSearchQuery(value)
    if (debounceTimer.current) clearTimeout(debounceTimer.current)
    debounceTimer.current = setTimeout(() => performSearch(value), 1000)
  }, [performSearch])

  const convertImageToDataUrl = async (imagePath: string): Promise<string> => {
    if (imagePath.startsWith('data:')) return imagePath

    // Proxy external URLs (GCS) through backend to avoid CORS
    const fetchUrl = imagePath.startsWith('https://storage.cloud.google.com/')
      ? `/api/catalog/image?url=${encodeURIComponent(imagePath)}`
      : imagePath

    const response = await fetch(fetchUrl)
    let blob = await response.blob()

    // Fix missing/generic mime type based on file extension
    if (blob.type === '' || blob.type === 'application/octet-stream') {
      const ext = imagePath.split('.').pop()?.toLowerCase()
      const mimeMap: Record<string, string> = { jpg: 'image/jpeg', jpeg: 'image/jpeg', png: 'image/png', webp: 'image/webp', gif: 'image/gif' }
      const mime = mimeMap[ext || ''] || 'image/jpeg'
      blob = new Blob([blob], { type: mime })
    }

    return new Promise((resolve, reject) => {
      const reader = new FileReader()
      reader.onloadend = () => resolve(reader.result as string)
      reader.onerror = reject
      reader.readAsDataURL(blob)
    })
  }

  const handleGenerate = useCallback(async () => {
    if (!fullBodyImage || garmentImages.length === 0) return

    setIsGenerating(true)
    setError(null)
    setVtoResultImage(null)
    setBestImageBase64(null)
    setBestImageScore(null)
    // Clean up old blob URLs
    allVideoUrls.forEach((url) => URL.revokeObjectURL(url))
    setGeneratedVideos([])
    setVideoFilenames([])
    setVideoScores([])
    setSelectedVideoIndex(0)

    try {
      // Convert images to blobs for the unified endpoint
      const bodyDataUrl = await convertImageToDataUrl(fullBodyImage)
      const bodyBlob = dataUrlToBlob(bodyDataUrl)

      // Split garments: GCS catalog items pass as URIs, others convert to blobs
      const uploadGarments = garmentImages.filter(img => !garmentGcsMap[img])
      const gcsUris = garmentImages.filter(img => garmentGcsMap[img]).map(img => garmentGcsMap[img])

      const garmentBlobs = await Promise.all(
        uploadGarments.map(async (img) => {
          const dataUrl = await convertImageToDataUrl(img)
          return dataUrlToBlob(dataUrl)
        })
      )

      setPipelineStep('image-vto')

      await generateVideoVTO(
        {
          fullBodyImage: bodyBlob,
          garments: garmentBlobs,
          garmentUris: gcsUris,
          numVariations: 3,
          numberOfVideos: 4,
        },
        (event: VideoVTOEvent) => {
          switch (event.status) {
            case 'generating_image':
              setPipelineStep('image-vto')
              break
            case 'image_ready':
              if (event.image_base64) {
                setVtoResultImage(`data:image/png;base64,${event.image_base64}`)
                setBestImageBase64(event.image_base64)
                setBestImageScore(event.face_score ?? event.final_score ?? null)
                console.log(
                  `[VideoVTO] Best VTO image ready (face_score: ${event.face_score?.toFixed(1)}, final_score: ${event.final_score?.toFixed(1)})`
                )
              }
              setPipelineStep('video-generation')
              break
            case 'generating_videos':
              setPipelineStep('video-generation')
              break
            case 'videos':
              if (event.videos) {
                const urls = event.videos.map(base64ToVideoUrl)
                setGeneratedVideos(urls)
                setVideoFilenames(event.filenames ?? [])
                setVideoScores(event.scores ?? [])
                setSelectedVideoIndex(0)
              }
              break
          }
        },
        (errorMessage: string) => {
          setError(errorMessage)
        },
      )
    } catch (err) {
      console.error('Error in clothes video pipeline:', err)
      setError(err instanceof Error ? err.message : 'Failed to generate video')
    } finally {
      setIsGenerating(false)
      setPipelineStep('idle')
    }
  }, [fullBodyImage, garmentImages])

  const handleRegenerateVideos = useCallback(async () => {
    if (!bestImageBase64) return

    setIsGenerating(true)
    setError(null)
    allVideoUrls.forEach((url) => URL.revokeObjectURL(url))
    setGeneratedVideos([])
    setVideoFilenames([])
    setVideoScores([])
    setSelectedVideoIndex(0)

    try {
      const modelBlob = base64ToBlob(bestImageBase64, 'image/png')

      setPipelineStep('video-generation')

      await generateAnimateModel(
        {
          modelImage: modelBlob,
          numberOfVideos: 4,
        },
        (event) => {
          switch (event.status) {
            case 'generating_videos':
              setPipelineStep('video-generation')
              break
            case 'videos':
              if (event.videos) {
                const urls = event.videos.map(base64ToVideoUrl)
                setGeneratedVideos(urls)
                setVideoFilenames(event.filenames ?? [])
                setVideoScores(event.scores ?? [])
                setSelectedVideoIndex(0)
              }
              break
          }
        },
        (errorMessage: string) => {
          setError(errorMessage)
        },
      )
    } catch (err) {
      console.error('Error regenerating videos:', err)
      setError(err instanceof Error ? err.message : 'Failed to regenerate videos')
    } finally {
      setIsGenerating(false)
      setPipelineStep('idle')
    }
  }, [bestImageBase64])

  const handleDownload = (index: number) => {
    const url = generatedVideos[index]
    const filename = videoFilenames[index] || `clothes_vto_${index + 1}.mp4`
    downloadVideoFromUrl(url, filename)
  }

  // Image upload handlers
  const handleModelImageUpload = (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      const reader = new FileReader()
      reader.onloadend = () => setFullBodyImage(reader.result as string)
      reader.readAsDataURL(file)
    }
  }

  const handleGarmentUpload = (e: ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files
    if (!files || files.length === 0) return
    Array.from(files).forEach((file) => {
      const reader = new FileReader()
      reader.onloadend = () => setGarmentImages((prev) => [...prev, reader.result as string])
      reader.readAsDataURL(file)
    })
  }

  const handleToggleGarmentSelection = (garmentPath: string) => {
    setSelectedGarments((prev) =>
      prev.includes(garmentPath) ? prev.filter((p) => p !== garmentPath) : [...prev, garmentPath]
    )
  }

  const handleCloseGarmentGallery = () => {
    setSelectedGarments([])
    setShowGarmentGallery(false)
    setSearchQuery('')
    setSearchResults([])
    if (debounceTimer.current) clearTimeout(debounceTimer.current)
  }

  const handleAddSelectedGarments = () => {
    if (selectedGarments.length > 0) {
      // Record GCS URIs for catalog garments
      const allResults = searchResults
      const newMap: Record<string, string> = {}
      for (const url of selectedGarments) {
        const match = allResults.find(r => r.img_url === url)
        if (match?.gs_uri) {
          newMap[url] = match.gs_uri
        }
      }
      setGarmentGcsMap(prev => ({ ...prev, ...newMap }))
      setGarmentImages((prev) => [...prev, ...selectedGarments])
      setSelectedGarments([])
      setShowGarmentGallery(false)
    }
  }

  const handleSelectModelFromGallery = (modelPath: string) => {
    setFullBodyImage(modelPath)
    setShowModelGallery(false)
  }

  const isFormValid = fullBodyImage && garmentImages.length > 0

  // --- Render ---
  return (
    <>
      <div className="grid grid-cols-1 lg:grid-cols-[480px_1fr] gap-6 lg:gap-8">
        {/* Left Panel: Form */}
        <div className="glass-panel rounded-xl p-5 lg:p-6 h-fit animate-fade-in">
          {/* Product Selector */}
          {currentProduct && availableProducts && onProductChange && (
            <div className="mb-5 pb-5 border-b border-black/[0.06] dark:border-white/[0.08]">
              <ProductSelector
                currentProduct={currentProduct}
                availableProducts={availableProducts}
                onProductChange={onProductChange}
              />
            </div>
          )}

          <div className="space-y-6">
            {/* Section 1: Model Image */}
            <section>
              <div className="flex items-center justify-between mb-3">
                <div>
                  <h3 className="text-[15px] font-semibold">Model Image</h3>
                  <p className="text-caption text-gm-text-tertiary-light dark:text-gm-text-tertiary mt-0.5">
                    Full body shot with visible face
                  </p>
                </div>
                <button
                  onClick={() => setShowModelGallery(true)}
                  className="text-caption text-gm-accent hover:text-gm-accent-hover transition-colors flex items-center gap-1"
                >
                  <svg
                    width="14"
                    height="14"
                    viewBox="0 0 24 24"
                    fill="none"
                    className="opacity-70"
                  >
                    <rect
                      x="3"
                      y="3"
                      width="7"
                      height="7"
                      rx="1"
                      stroke="currentColor"
                      strokeWidth="2"
                    />
                    <rect
                      x="14"
                      y="3"
                      width="7"
                      height="7"
                      rx="1"
                      stroke="currentColor"
                      strokeWidth="2"
                    />
                    <rect
                      x="3"
                      y="14"
                      width="7"
                      height="7"
                      rx="1"
                      stroke="currentColor"
                      strokeWidth="2"
                    />
                    <rect
                      x="14"
                      y="14"
                      width="7"
                      height="7"
                      rx="1"
                      stroke="currentColor"
                      strokeWidth="2"
                    />
                  </svg>
                  From gallery
                </button>
              </div>

              {fullBodyImage ? (
                <div className="relative w-32 h-40 rounded-xl overflow-hidden bg-black/[0.02] dark:bg-white/[0.02] group cursor-pointer"
                  onClick={() => setLightboxImage(fullBodyImage)}
                >
                  <img
                    src={fullBodyImage}
                    alt="Model"
                    className="w-full h-full object-contain"
                  />
                  <div className="absolute inset-0 bg-gradient-to-t from-black/50 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-200 pointer-events-none" />
                  <button
                    onClick={(e) => { e.stopPropagation(); setFullBodyImage(null) }}
                    disabled={isGenerating}
                    className="absolute top-2 right-2 w-7 h-7 bg-black/50 hover:bg-black/70 backdrop-blur-sm rounded-full flex items-center justify-center transition-all opacity-0 group-hover:opacity-100 disabled:opacity-50"
                  >
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none">
                      <path
                        d="M18 6L6 18M6 6L18 18"
                        stroke="white"
                        strokeWidth="2"
                        strokeLinecap="round"
                      />
                    </svg>
                  </button>
                </div>
              ) : (
                <label
                  className={`upload-zone-compact cursor-pointer ${isGenerating ? 'opacity-50 pointer-events-none' : ''}`}
                >
                  <input
                    type="file"
                    accept="image/*"
                    className="hidden"
                    onChange={handleModelImageUpload}
                    disabled={isGenerating}
                  />
                  <div className="w-10 h-10 rounded-lg bg-gm-accent/10 dark:bg-gm-accent/20 flex items-center justify-center">
                    <svg
                      width="20"
                      height="20"
                      viewBox="0 0 24 24"
                      fill="none"
                      className="text-gm-accent"
                    >
                      <path
                        d="M12 15V3M12 3L8 7M12 3L16 7"
                        stroke="currentColor"
                        strokeWidth="1.5"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                      />
                      <path
                        d="M2 17v2a2 2 0 002 2h16a2 2 0 002-2v-2"
                        stroke="currentColor"
                        strokeWidth="1.5"
                        strokeLinecap="round"
                      />
                    </svg>
                  </div>
                  <div>
                    <p className="text-button font-medium">Drop file or click to upload</p>
                    <p className="text-[11px] text-gm-text-tertiary-light dark:text-gm-text-tertiary mt-0.5">
                      PNG, JPG up to 10MB
                    </p>
                  </div>
                </label>
              )}
            </section>

            <div className="divider" />

            {/* Section 2: Garment Images */}
            <section>
              <div className="flex items-center justify-between mb-3">
                <div>
                  <h3 className="text-[15px] font-semibold">Garment Images</h3>
                  <p className="text-caption text-gm-text-tertiary-light dark:text-gm-text-tertiary mt-0.5">
                    Upload one or more clothing items
                  </p>
                </div>
                <button
                  onClick={() => setShowGarmentGallery(true)}
                  className="text-caption text-gm-accent hover:text-gm-accent-hover transition-colors flex items-center gap-1"
                >
                  <svg
                    width="14"
                    height="14"
                    viewBox="0 0 24 24"
                    fill="none"
                    className="opacity-70"
                  >
                    <rect
                      x="3"
                      y="3"
                      width="7"
                      height="7"
                      rx="1"
                      stroke="currentColor"
                      strokeWidth="2"
                    />
                    <rect
                      x="14"
                      y="3"
                      width="7"
                      height="7"
                      rx="1"
                      stroke="currentColor"
                      strokeWidth="2"
                    />
                    <rect
                      x="3"
                      y="14"
                      width="7"
                      height="7"
                      rx="1"
                      stroke="currentColor"
                      strokeWidth="2"
                    />
                    <rect
                      x="14"
                      y="14"
                      width="7"
                      height="7"
                      rx="1"
                      stroke="currentColor"
                      strokeWidth="2"
                    />
                  </svg>
                  From gallery
                </button>
              </div>

              {garmentImages.length > 0 ? (
                <div className="rounded-xl overflow-hidden bg-gray-100 dark:bg-black">
                  <div className="grid grid-cols-4 gap-1 p-1">
                    {garmentImages.map((imgPath, index) => (
                      <div
                        key={index}
                        className="relative aspect-square bg-white dark:bg-white/5 group/item"
                      >
                        <img
                          src={proxySrc(imgPath)}
                          alt={`Garment ${index + 1}`}
                          className="w-full h-full object-contain"
                        />
                        <div className="absolute top-1 left-1 w-5 h-5 rounded-full bg-black/60 dark:bg-black/50 backdrop-blur-sm flex items-center justify-center text-white text-[10px] font-medium">
                          {index + 1}
                        </div>
                        <button
                          onClick={() => setGarmentImages(prev => prev.filter((_, i) => i !== index))}
                          disabled={isGenerating}
                          className="absolute top-1 right-1 w-5 h-5 bg-black/50 hover:bg-black/70 backdrop-blur-sm rounded-full flex items-center justify-center transition-all opacity-0 group-hover/item:opacity-100 disabled:opacity-50"
                        >
                          <svg width="8" height="8" viewBox="0 0 24 24" fill="none">
                            <path d="M18 6L6 18M6 6L18 18" stroke="white" strokeWidth="3" strokeLinecap="round"/>
                          </svg>
                        </button>
                      </div>
                    ))}
                  </div>
                </div>
              ) : (
                <label
                  className={`upload-zone-compact cursor-pointer ${isGenerating ? 'opacity-50 pointer-events-none' : ''}`}
                >
                  <input
                    type="file"
                    accept="image/*"
                    multiple
                    className="hidden"
                    onChange={handleGarmentUpload}
                    disabled={isGenerating}
                  />
                  <div className="w-10 h-10 rounded-lg bg-gm-accent/10 dark:bg-gm-accent/20 flex items-center justify-center flex-shrink-0">
                    <svg
                      width="20"
                      height="20"
                      viewBox="0 0 24 24"
                      fill="none"
                      className="text-gm-accent"
                    >
                      <path
                        d="M12 15V3M12 3L8 7M12 3L16 7"
                        stroke="currentColor"
                        strokeWidth="1.5"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                      />
                      <path
                        d="M2 17v2a2 2 0 002 2h16a2 2 0 002-2v-2"
                        stroke="currentColor"
                        strokeWidth="1.5"
                        strokeLinecap="round"
                      />
                    </svg>
                  </div>
                  <div>
                    <p className="text-button font-medium">Drop files or click to upload</p>
                    <p className="text-[11px] text-gm-text-tertiary-light dark:text-gm-text-tertiary mt-0.5">
                      PNG, JPG up to 10MB each
                    </p>
                  </div>
                </label>
              )}
            </section>

            {/* Generate Button */}
            <button
              type="button"
              onClick={handleGenerate}
              disabled={isGenerating || !isFormValid}
              className="btn-primary w-full py-3.5 rounded-xl text-button font-medium flex items-center justify-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {isGenerating ? (
                <>
                  <svg className="animate-spin h-5 w-5" viewBox="0 0 24 24">
                    <circle
                      className="opacity-25"
                      cx="12"
                      cy="12"
                      r="10"
                      stroke="currentColor"
                      strokeWidth="4"
                      fill="none"
                    />
                    <path
                      className="opacity-75"
                      fill="currentColor"
                      d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                    />
                  </svg>
                  {pipelineStep === 'image-vto'
                    ? 'Step 1: Generating VTO image...'
                    : 'Step 2: Generating videos...'}
                </>
              ) : (
                <>
                  <svg width="18" height="18" viewBox="0 0 24 24" fill="none">
                    <path
                      d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14v-4z"
                      stroke="currentColor"
                      strokeWidth="2"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    />
                    <rect
                      x="3"
                      y="6"
                      width="12"
                      height="12"
                      rx="2"
                      stroke="currentColor"
                      strokeWidth="2"
                    />
                  </svg>
                  Generate Video
                </>
              )}
            </button>

            {!isFormValid && (
              <p className="text-caption text-center text-gm-text-tertiary-light dark:text-gm-text-tertiary">
                Please upload a model image and at least one garment
              </p>
            )}
          </div>
        </div>

        {/* Right Panel: Preview */}
        <div
          className={`${showVideo ? 'glass-panel' : 'border border-black/[0.06] dark:border-white/[0.08]'} rounded-xl flex items-center justify-center min-h-[600px] overflow-hidden relative animate-fade-in delay-100`}
        >
          {error ? (
            <div className="text-center px-8 animate-fade-in">
              <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-red-500/10 flex items-center justify-center">
                <svg
                  width="32"
                  height="32"
                  viewBox="0 0 24 24"
                  fill="none"
                  className="text-red-500"
                >
                  <path
                    d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
                    stroke="currentColor"
                    strokeWidth="2"
                    strokeLinecap="round"
                  />
                </svg>
              </div>
              <h3 className="text-xl font-semibold mb-2">Generation Failed</h3>
              <p className="text-gm-text-secondary mb-3">{error}</p>
              <button
                onClick={() => openFeedbackForm({ feedbackType: 'Bug / Error', capability: 'Video VTO', errorMessage: error || undefined })}
                className="text-sm text-gm-accent hover:text-gm-accent-hover transition-colors underline underline-offset-2 mb-4 block"
              >
                Report this issue
              </button>
              {bestImageBase64 ? (
                <button
                  onClick={() => {
                    setError(null)
                    handleRegenerateVideos()
                  }}
                  className="btn-primary px-4 py-2 rounded-lg"
                >
                  Regenerate Videos
                </button>
              ) : (
                <button
                  onClick={() => setError(null)}
                  className="btn-secondary px-4 py-2 rounded-lg"
                >
                  Try Again
                </button>
              )}
            </div>
          ) : generatedVideos.length > 0 ? (
            <div className="w-full h-full p-6 flex flex-col animate-scale-in">
              {/* Primary Video Player */}
              <div className="flex-1 relative rounded-xl overflow-hidden bg-black group/player">
                <video
                  key={generatedVideos[selectedVideoIndex]}
                  controls
                  autoPlay
                  className="w-full h-full object-contain"
                >
                  <source src={generatedVideos[selectedVideoIndex]} type="video/mp4" />
                </video>

                <div className="absolute top-4 right-4 flex gap-2 opacity-0 group-hover/player:opacity-100 transition-opacity duration-200">
                  <button
                    onClick={() => handleDownload(selectedVideoIndex)}
                    className="w-10 h-10 bg-black/60 hover:bg-black/80 backdrop-blur-sm rounded-xl flex items-center justify-center transition-all"
                    title="Download video"
                  >
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none">
                      <path
                        d="M12 15V3M12 15L8 11M12 15L16 11"
                        stroke="white"
                        strokeWidth="2"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                      />
                      <path
                        d="M2 17L2 19C2 20.1046 2.89543 21 4 21L20 21C21.1046 21 22 20.1046 22 19V17"
                        stroke="white"
                        strokeWidth="2"
                        strokeLinecap="round"
                      />
                    </svg>
                  </button>
                </div>
              </div>

              {/* Video Thumbnails Grid */}
              <div className="grid grid-cols-4 gap-3 mt-4">
                {generatedVideos.map((videoUrl, index) => (
                    <div key={index} className="relative group/thumb">
                      <button
                        onClick={() => setSelectedVideoIndex(index)}
                        className={`w-full aspect-video bg-black/30 rounded-lg overflow-hidden mb-1.5 transition-all ${
                          selectedVideoIndex === index
                            ? 'ring-2 ring-gm-accent ring-offset-2 ring-offset-black/20'
                            : 'hover:ring-1 hover:ring-white/30'
                        }`}
                      >
                        <video
                          className="w-full h-full object-cover"
                          muted
                          playsInline
                          onMouseEnter={(e) => e.currentTarget.play()}
                          onMouseLeave={(e) => {
                            e.currentTarget.pause()
                            e.currentTarget.currentTime = 0
                          }}
                        >
                          <source src={videoUrl} type="video/mp4" />
                        </video>
                        {selectedVideoIndex === index && (
                          <div className="absolute inset-0 flex items-center justify-center bg-black/30">
                            <div className="w-6 h-6 rounded-full bg-gm-accent flex items-center justify-center">
                              <svg width="12" height="12" viewBox="0 0 24 24" fill="none">
                                <path
                                  d="M20 6L9 17l-5-5"
                                  stroke="white"
                                  strokeWidth="3"
                                  strokeLinecap="round"
                                  strokeLinejoin="round"
                                />
                              </svg>
                            </div>
                          </div>
                        )}
                        {videoScores[index] != null && (
                          <div className="absolute bottom-1 left-1 px-1.5 py-0.5 rounded bg-black/70 backdrop-blur-sm text-white text-[10px] font-medium leading-none">
                            {videoScores[index].toFixed(1)}%
                          </div>
                        )}
                      </button>
                      <div className="flex items-center justify-between">
                        <button
                          onClick={() => handleDownload(index)}
                          className="w-6 h-6 rounded-md flex items-center justify-center opacity-0 group-hover/thumb:opacity-100 transition-opacity text-gm-text-tertiary hover:text-gm-text-primary"
                          title="Download"
                        >
                          <svg width="12" height="12" viewBox="0 0 24 24" fill="none">
                            <path
                              d="M12 15V3M12 15L8 11M12 15L16 11"
                              stroke="currentColor"
                              strokeWidth="2"
                              strokeLinecap="round"
                              strokeLinejoin="round"
                            />
                            <path
                              d="M2 17L2 19C2 20.1046 2.89543 21 4 21L20 21C21.1046 21 22 20.1046 22 19V17"
                              stroke="currentColor"
                              strokeWidth="2"
                              strokeLinecap="round"
                            />
                          </svg>
                        </button>
                      </div>
                    </div>
                ))}
              </div>

              {/* VTO result preview + reset */}
              <div className="flex items-center justify-between mt-4">
                {vtoResultImage && (
                  <button
                    onClick={() => setShowVtoPreview(true)}
                    className="flex items-center gap-3 hover:opacity-80 transition-opacity"
                  >
                    <img
                      src={vtoResultImage}
                      alt="VTO Result"
                      className="w-12 h-12 rounded-lg object-cover border border-white/10"
                    />
                    <span className="text-caption text-gm-text-tertiary">
                      VTO starting image{bestImageScore != null && ` (${bestImageScore.toFixed(1)})`}
                    </span>
                  </button>
                )}
                <button
                  onClick={handleRegenerateVideos}
                  disabled={isGenerating}
                  className="btn-secondary px-4 py-2 rounded-lg text-caption disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  Regenerate Video
                </button>
              </div>
            </div>
          ) : (
            <>
              {showVideo && <BackgroundVideo />}
              <div className="absolute inset-0 bg-gradient-to-t from-gm-bg-light/70 via-gm-bg-light/30 to-transparent dark:from-gm-bg/70 dark:via-gm-bg/30" />
              <div className="absolute inset-0 bg-radial-glow opacity-50" />

              {isGenerating ? (
                <div className="text-center px-8 relative z-10 max-w-lg">
                  {/* Step indicator */}
                  <div className="flex items-center justify-center gap-4 mb-6">
                    <div
                      className={`flex items-center gap-2 px-3 py-1.5 rounded-full text-caption ${
                        pipelineStep === 'image-vto'
                          ? 'bg-gm-accent/20 text-gm-accent'
                          : 'bg-white/[0.06] text-gm-text-tertiary'
                      }`}
                    >
                      {pipelineStep === 'image-vto' ? (
                        <svg className="animate-spin h-3.5 w-3.5" viewBox="0 0 24 24">
                          <circle
                            className="opacity-25"
                            cx="12"
                            cy="12"
                            r="10"
                            stroke="currentColor"
                            strokeWidth="4"
                            fill="none"
                          />
                          <path
                            className="opacity-75"
                            fill="currentColor"
                            d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                          />
                        </svg>
                      ) : (
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none">
                          <path
                            d="M20 6L9 17l-5-5"
                            stroke="currentColor"
                            strokeWidth="2"
                            strokeLinecap="round"
                            strokeLinejoin="round"
                          />
                        </svg>
                      )}
                      Step 1: Image VTO
                    </div>
                    <div className="w-6 h-px bg-white/20" />
                    <div
                      className={`flex items-center gap-2 px-3 py-1.5 rounded-full text-caption ${
                        pipelineStep === 'video-generation'
                          ? 'bg-gm-accent/20 text-gm-accent'
                          : 'bg-white/[0.06] text-gm-text-tertiary'
                      }`}
                    >
                      {pipelineStep === 'video-generation' && (
                        <svg className="animate-spin h-3.5 w-3.5" viewBox="0 0 24 24">
                          <circle
                            className="opacity-25"
                            cx="12"
                            cy="12"
                            r="10"
                            stroke="currentColor"
                            strokeWidth="4"
                            fill="none"
                          />
                          <path
                            className="opacity-75"
                            fill="currentColor"
                            d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                          />
                        </svg>
                      )}
                      Step 2: Video
                    </div>
                  </div>

                  {/* Animated rings */}
                  <div className="relative mb-6 h-16 flex items-center justify-center">
                    <div className="w-16 h-16 rounded-full border-2 border-gm-accent/30 animate-ping absolute" />
                    <div className="w-16 h-16 rounded-full border-2 border-gm-accent/20 animate-pulse" />
                    <div className="absolute inset-0 flex items-center justify-center">
                      <div className="w-3 h-3 rounded-full bg-gm-accent animate-pulse" />
                    </div>
                  </div>

                  <h2 className="text-[32px] lg:text-[40px] font-display font-bold mb-3 leading-tight tracking-tight">
                    {pipelineStep === 'image-vto'
                      ? 'Dressing the '
                      : 'Creating your '}
                    <span
                      className="text-gm-accent"
                      style={{
                        textShadow:
                          '0 0 40px rgba(138, 180, 248, 0.4), 0 0 80px rgba(138, 180, 248, 0.2)',
                      }}
                    >
                      {pipelineStep === 'image-vto' ? 'model' : 'video'}
                    </span>
                    ...
                  </h2>
                  <p className="text-body-lg text-gm-text-secondary-light dark:text-gm-text-secondary leading-relaxed">
                    {pipelineStep === 'image-vto'
                      ? 'Applying garments to the model with AI virtual try-on.'
                      : 'Generating a cinematic reveal video with Veo.'}
                  </p>

                  {/* Show VTO result preview during step 2 */}
                  {pipelineStep === 'video-generation' && vtoResultImage && (
                    <div className="mt-6 flex justify-center">
                      <img
                        src={vtoResultImage}
                        alt="VTO Result"
                        className="w-32 h-auto rounded-xl border border-white/10 shadow-lg"
                      />
                    </div>
                  )}
                </div>
              ) : (
                <div className="text-center px-8 relative z-10 max-w-lg animate-fade-in-up">
                  <div className="w-16 h-16 mx-auto mb-6 rounded-2xl bg-gm-accent/10 dark:bg-gm-accent/20 flex items-center justify-center glow-accent">
                    <svg
                      width="28"
                      height="28"
                      viewBox="0 0 24 24"
                      fill="none"
                      className="text-gm-accent"
                    >
                      <path
                        d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14v-4z"
                        stroke="currentColor"
                        strokeWidth="1.5"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                      />
                      <rect
                        x="3"
                        y="6"
                        width="12"
                        height="12"
                        rx="2"
                        stroke="currentColor"
                        strokeWidth="1.5"
                      />
                    </svg>
                  </div>

                  <h2 className="text-[40px] lg:text-[48px] font-display font-bold mb-4 leading-tight tracking-tight">
                    Clothes{' '}
                    <span
                      className="text-gm-accent"
                      style={{
                        textShadow:
                          '0 0 40px rgba(138, 180, 248, 0.4), 0 0 80px rgba(138, 180, 248, 0.2)',
                      }}
                    >
                      Video VTO
                    </span>
                  </h2>
                  <p className="text-body-lg text-gm-text-secondary-light dark:text-gm-text-secondary leading-relaxed">
                    Upload a model and garments to generate a cinematic reveal video with face-locked identity.
                  </p>

                  <div className="flex flex-wrap items-center justify-center gap-2 mt-6">
                    <span className="px-3 py-1.5 rounded-full bg-black/[0.08] dark:bg-white/[0.12] text-caption text-gm-text-secondary-light dark:text-gm-text-secondary">
                      Video Try On
                    </span>
                    <span className="px-3 py-1.5 rounded-full bg-black/[0.08] dark:bg-white/[0.12] text-caption text-gm-text-secondary-light dark:text-gm-text-secondary">
                      Face Consistency
                    </span>
                    <span className="px-3 py-1.5 rounded-full bg-black/[0.08] dark:bg-white/[0.12] text-caption text-gm-text-secondary-light dark:text-gm-text-secondary">
                      Automatic Evaluation
                    </span>
                  </div>
                </div>
              )}
            </>
          )}
        </div>
      </div>

      {/* Garment Gallery Modal */}
      {showGarmentGallery && (
        <div
          className="fixed inset-0 bg-black flex items-center justify-center z-[9999] p-6"
          onClick={handleCloseGarmentGallery}
        >
          <div
            className="rounded-2xl p-6 max-w-2xl w-full max-h-[75vh] overflow-hidden flex flex-col bg-black border border-gray-800"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-center justify-between mb-5">
              <div>
                <h2 className="text-xl font-semibold">Garment Gallery</h2>
                <p className="text-caption text-gm-text-tertiary-light dark:text-gm-text-tertiary mt-1">
                  {selectedGarments.length > 0
                    ? `${selectedGarments.length} item${selectedGarments.length > 1 ? 's' : ''} selected`
                    : 'Select clothing items to try on'}
                </p>
              </div>
              <button
                onClick={handleCloseGarmentGallery}
                className="w-9 h-9 rounded-lg bg-black/[0.06] hover:bg-black/[0.1] dark:bg-white/10 dark:hover:bg-white/20 flex items-center justify-center transition-all"
              >
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none">
                  <path
                    d="M18 6L6 18M6 6L18 18"
                    stroke="currentColor"
                    strokeWidth="2"
                    strokeLinecap="round"
                  />
                </svg>
              </button>
            </div>

            {/* Search Bar */}
            <div className="relative mb-4">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" className="absolute left-3 top-1/2 -translate-y-1/2 text-gm-text-tertiary-light dark:text-gm-text-tertiary">
                <circle cx="11" cy="11" r="7" stroke="currentColor" strokeWidth="2"/>
                <path d="M20 20l-4-4" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
              </svg>
              <input
                type="text"
                value={searchQuery}
                onChange={handleSearchChange}
                placeholder="Search catalogue..."
                className="input-glass w-full pl-9 pr-4 py-2.5 rounded-lg text-sm"
              />
            </div>

            <div className="overflow-y-auto flex-1 -mx-2 px-2">
              {/* Catalogue Search Results */}
              {isSearching && (
                <div className="flex items-center justify-center py-8">
                  <svg className="animate-spin h-6 w-6 text-gm-accent" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none"/>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"/>
                  </svg>
                </div>
              )}

              {!isSearching && searchQuery.trim() && (
                <>
                  {searchResults.length > 0 ? (
                    <div className="mb-6">
                      <div className="grid grid-cols-3 lg:grid-cols-4 gap-4">
                        {searchResults.map((item) => {
                          const isSelected = selectedGarments.includes(item.img_url)
                          return (
                            <div
                              key={`cat-${item.id}`}
                              onClick={() => handleToggleGarmentSelection(item.img_url)}
                              className={`group cursor-pointer rounded-xl overflow-hidden bg-white dark:bg-white/10 border-2 transition-all ${
                                isSelected
                                  ? 'border-gm-accent shadow-lg shadow-gm-accent/20'
                                  : 'border-transparent hover:border-gm-accent/50'
                              }`}
                            >
                              <div className="relative aspect-square">
                                <img
                                  src={`/api/catalog/image?url=${encodeURIComponent(item.img_url)}`}
                                  alt={item.description}
                                  className="w-full h-full object-contain p-3"
                                />
                                <div className={`absolute inset-0 transition-all flex items-center justify-center pointer-events-none ${
                                  isSelected ? 'bg-gm-accent/10' : 'bg-transparent'
                                }`}>
                                  <div className={`w-10 h-10 rounded-full flex items-center justify-center transition-all ${
                                    isSelected
                                      ? 'bg-gm-accent opacity-100 scale-100'
                                      : 'bg-white/90 opacity-0 group-hover:opacity-100 scale-75 group-hover:scale-100'
                                  }`}>
                                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" className={isSelected ? 'text-white' : 'text-gm-bg'}>
                                      <path d="M5 12l5 5L20 7" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"/>
                                    </svg>
                                  </div>
                                </div>
                              </div>
                            </div>
                          )
                        })}
                      </div>
                    </div>
                  ) : (
                    <p className="text-center text-gm-text-tertiary-light dark:text-gm-text-tertiary py-8">No results found</p>
                  )}
                </>
              )}

              {/* Preset garments (shown when not searching) */}
              {!searchQuery.trim() && (
              <>
              <div className="mb-6">
                <h3 className="text-sm font-semibold text-gm-text-secondary-light dark:text-gm-text-secondary mb-3 px-2">
                  Women
                </h3>
                <div className="grid grid-cols-3 lg:grid-cols-4 gap-4">
                  {WOMEN_GARMENTS.map((garmentPath, index) => {
                    const isSelected = selectedGarments.includes(garmentPath)
                    return (
                      <div
                        key={`women-${index}`}
                        onClick={() => handleToggleGarmentSelection(garmentPath)}
                        className={`group cursor-pointer rounded-xl overflow-hidden bg-white dark:bg-white/10 border-2 transition-all ${
                          isSelected
                            ? 'border-gm-accent shadow-lg shadow-gm-accent/20'
                            : 'border-transparent hover:border-gm-accent/50'
                        }`}
                      >
                        <div className="relative aspect-square">
                          <img
                            src={garmentPath}
                            alt={`Women garment ${index + 1}`}
                            className="w-full h-full object-contain p-3"
                          />
                          <div
                            className={`absolute inset-0 transition-all flex items-center justify-center pointer-events-none ${
                              isSelected ? 'bg-gm-accent/10' : 'bg-transparent'
                            }`}
                          >
                            <div
                              className={`w-10 h-10 rounded-full flex items-center justify-center transition-all ${
                                isSelected
                                  ? 'bg-gm-accent opacity-100 scale-100'
                                  : 'bg-white/90 opacity-0 group-hover:opacity-100 scale-75 group-hover:scale-100'
                              }`}
                            >
                              <svg
                                width="16"
                                height="16"
                                viewBox="0 0 24 24"
                                fill="none"
                                className={isSelected ? 'text-white' : 'text-gm-bg'}
                              >
                                <path
                                  d="M5 12l5 5L20 7"
                                  stroke="currentColor"
                                  strokeWidth="2.5"
                                  strokeLinecap="round"
                                  strokeLinejoin="round"
                                />
                              </svg>
                            </div>
                          </div>
                        </div>
                      </div>
                    )
                  })}
                </div>
              </div>

              <div>
                <h3 className="text-sm font-semibold text-gm-text-secondary-light dark:text-gm-text-secondary mb-3 px-2">
                  Men
                </h3>
                <div className="grid grid-cols-3 lg:grid-cols-4 gap-4">
                  {MEN_GARMENTS.map((garmentPath, index) => {
                    const isSelected = selectedGarments.includes(garmentPath)
                    return (
                      <div
                        key={`men-${index}`}
                        onClick={() => handleToggleGarmentSelection(garmentPath)}
                        className={`group cursor-pointer rounded-xl overflow-hidden bg-white dark:bg-white/10 border-2 transition-all ${
                          isSelected
                            ? 'border-gm-accent shadow-lg shadow-gm-accent/20'
                            : 'border-transparent hover:border-gm-accent/50'
                        }`}
                      >
                        <div className="relative aspect-square">
                          <img
                            src={garmentPath}
                            alt={`Men garment ${index + 1}`}
                            className="w-full h-full object-contain p-3"
                          />
                          <div
                            className={`absolute inset-0 transition-all flex items-center justify-center pointer-events-none ${
                              isSelected ? 'bg-gm-accent/10' : 'bg-transparent'
                            }`}
                          >
                            <div
                              className={`w-10 h-10 rounded-full flex items-center justify-center transition-all ${
                                isSelected
                                  ? 'bg-gm-accent opacity-100 scale-100'
                                  : 'bg-white/90 opacity-0 group-hover:opacity-100 scale-75 group-hover:scale-100'
                              }`}
                            >
                              <svg
                                width="16"
                                height="16"
                                viewBox="0 0 24 24"
                                fill="none"
                                className={isSelected ? 'text-white' : 'text-gm-bg'}
                              >
                                <path
                                  d="M5 12l5 5L20 7"
                                  stroke="currentColor"
                                  strokeWidth="2.5"
                                  strokeLinecap="round"
                                  strokeLinejoin="round"
                                />
                              </svg>
                            </div>
                          </div>
                        </div>
                      </div>
                    )
                  })}
                </div>
              </div>
              </>
              )}
            </div>

            <div className="flex items-center gap-3 mt-5 pt-5 border-t border-black/[0.06] dark:border-white/10">
              {selectedGarments.length > 0 && (
                <button
                  onClick={() => setSelectedGarments([])}
                  className="px-4 py-2 rounded-lg text-button text-gm-text-secondary-light dark:text-gm-text-secondary hover:bg-black/[0.04] dark:hover:bg-white/[0.08] transition-all"
                >
                  Clear selection
                </button>
              )}
              <button
                onClick={handleAddSelectedGarments}
                disabled={selectedGarments.length === 0}
                className="ml-auto btn-primary px-6 py-2.5 rounded-lg text-button font-medium flex items-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                Add {selectedGarments.length > 0 && `(${selectedGarments.length})`}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Model Gallery Modal */}
      {showModelGallery && (
        <div
          className="fixed inset-0 bg-black flex items-center justify-center z-[9999] p-6"
          onClick={() => setShowModelGallery(false)}
        >
          <div
            className="rounded-2xl p-6 max-w-2xl w-full max-h-[75vh] overflow-hidden flex flex-col bg-black border border-gray-800"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-center justify-between mb-5">
              <div>
                <h2 className="text-xl font-semibold">Model Gallery</h2>
                <p className="text-caption text-gm-text-tertiary-light dark:text-gm-text-tertiary mt-1">
                  Select a model for virtual try-on
                </p>
              </div>
              <button
                onClick={() => setShowModelGallery(false)}
                className="w-9 h-9 rounded-lg bg-black/[0.06] hover:bg-black/[0.1] dark:bg-white/10 dark:hover:bg-white/20 flex items-center justify-center transition-all"
              >
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none">
                  <path
                    d="M18 6L6 18M6 6L18 18"
                    stroke="currentColor"
                    strokeWidth="2"
                    strokeLinecap="round"
                  />
                </svg>
              </button>
            </div>

            <div className="overflow-y-auto flex-1 -mx-2 px-2">
              <div className="mb-6">
                <h3 className="text-sm font-semibold text-gm-text-secondary-light dark:text-gm-text-secondary mb-3 px-2">
                  Women
                </h3>
                <div className="grid grid-cols-3 lg:grid-cols-4 gap-4">
                  {womenModels.map((preset) => (
                    <div
                      key={preset.id}
                      onClick={() => handleSelectModelFromGallery(preset.thumbnail)}
                      className="group cursor-pointer rounded-xl overflow-hidden bg-white dark:bg-white/10 border border-transparent hover:border-gm-accent/50 transition-all"
                    >
                      <div className="relative aspect-[3/4]">
                        <img
                          src={preset.thumbnail}
                          alt={`${preset.label} woman`}
                          className="w-full h-full object-cover"
                        />
                        <div className="absolute inset-0 bg-transparent transition-all flex items-center justify-center pointer-events-none">
                          <div className="w-10 h-10 rounded-full bg-white/90 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-all scale-75 group-hover:scale-100">
                            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" className="text-gm-bg">
                              <path d="M5 12l5 5L20 7" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"/>
                            </svg>
                          </div>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              <div>
                <h3 className="text-sm font-semibold text-gm-text-secondary-light dark:text-gm-text-secondary mb-3 px-2">
                  Men
                </h3>
                <div className="grid grid-cols-3 lg:grid-cols-4 gap-4">
                  {menModels.map((preset) => (
                    <div
                      key={preset.id}
                      onClick={() => handleSelectModelFromGallery(preset.thumbnail)}
                      className="group cursor-pointer rounded-xl overflow-hidden bg-white dark:bg-white/10 border border-transparent hover:border-gm-accent/50 transition-all"
                    >
                      <div className="relative aspect-[3/4]">
                        <img
                          src={preset.thumbnail}
                          alt={`${preset.label} man`}
                          className="w-full h-full object-cover"
                        />
                        <div className="absolute inset-0 bg-transparent transition-all flex items-center justify-center pointer-events-none">
                          <div className="w-10 h-10 rounded-full bg-white/90 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-all scale-75 group-hover:scale-100">
                            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" className="text-gm-bg">
                              <path d="M5 12l5 5L20 7" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"/>
                            </svg>
                          </div>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* VTO Result Lightbox */}
      {showVtoPreview && vtoResultImage && (
        <ImageLightbox src={vtoResultImage} alt="VTO Result" onClose={() => setShowVtoPreview(false)} />
      )}

      {/* Model Image Lightbox */}
      {lightboxImage && (
        <ImageLightbox src={lightboxImage} alt="Model" onClose={() => setLightboxImage(null)} />
      )}
    </>
  )
}

export default ClothesVideoVTO
