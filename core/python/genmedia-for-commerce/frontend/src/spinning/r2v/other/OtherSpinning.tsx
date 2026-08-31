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

import { useState, ChangeEvent, useEffect } from 'react'
import type { JSX } from 'react'
import Product360Preview from '../../components/Product360Preview'
import Product360Form from '../../components/Product360Form'
import ProductSelector from '../../../components/ProductSelector'
import {
  preprocessImages,
  generateAllVideos,
  mergeVideos,
  base64ToBlob,
  dataUrlToFile,
} from '../../services/interpolationApi'
import {
  dataUrlToBase64,
  base64ToVideoUrl as r2vBase64ToVideoUrl,
} from '../../services/shoesSpinningApi'
import { runOtherSpinningPipeline } from '../../services/otherSpinningApi'
import type { Product } from '../../../config/featureConstraints'

interface OtherSpinningProps {
  prefilledImages?: string[]
  prefilledSpinMode?: 'r2v-standard' | 'interpolation'
  showVideo?: boolean
  currentProduct?: Product
  availableProducts?: Product[]
  onProductChange?: (product: Product) => void
}

interface ViewImages {
  front: string | null
  right: string | null
  back: string | null
  left: string | null
}

interface Template {
  product_name?: string
  product_type?: string
  product_img?: string
  product_images?: string[]
  video_path?: string
  video_prompt?: object
}

type ProcessingStep = 'idle' | 'preprocessing' | 'generating' | 'merging' | 'complete';
type SpinMode = 'r2v-standard' | 'interpolation';

function OtherSpinning({
  prefilledImages = [],
  prefilledSpinMode,
  showVideo = true,
  currentProduct,
  availableProducts,
  onProductChange
}: OtherSpinningProps) {
  const [spinMode, setSpinMode] = useState<SpinMode>(prefilledSpinMode || 'r2v-standard')
  const [showGallery, setShowGallery] = useState(false)
  const [templates, setTemplates] = useState<Template[]>([])

  // R2V Standard mode state
  const [selectedProductImages, setSelectedProductImages] = useState<string[]>(prefilledImages)
  const [uploadedImage, setUploadedImage] = useState<string | null>(
    prefilledImages.length > 0 ? prefilledImages[0] : null
  )

  // Interpolation mode state
  const [viewImages, setViewImages] = useState<ViewImages>({
    right: prefilledImages[0] || null,
    front: prefilledImages[1] || null,
    left: prefilledImages[2] || null,
    back: prefilledImages[3] || null,
  })

  // Shared state
  const [isLoading, setIsLoading] = useState(false)
  const [processingStep, setProcessingStep] = useState<ProcessingStep>('idle')
  const [generatedVideo, setGeneratedVideo] = useState<string | null>(null)
  const [videoBase64, setVideoBase64] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  // Load templates on mount
  useEffect(() => {
    fetch('/templates/360-templates.json')
      .then(response => response.json())
      .then(data => setTemplates(data))
      .catch(error => console.error('Error loading templates:', error))
  }, [])

  // R2V Standard handlers
  const handleR2VFileUpload = (files: File[]) => {
    const sorted = [...files].sort((a, b) => a.name.localeCompare(b.name))
    const newImages: string[] = new Array(sorted.length)
    let loadedCount = 0

    sorted.forEach((file, idx) => {
      const reader = new FileReader()
      reader.onloadend = () => {
        newImages[idx] = reader.result as string
        loadedCount++
        if (loadedCount === sorted.length) {
          setSelectedProductImages(newImages)
          setUploadedImage(newImages[0])
          setGeneratedVideo(null)
          setVideoBase64(null)
          setError(null)
        }
      }
      reader.readAsDataURL(file)
    })
  }

  const handleR2VRemoveFile = (index?: number) => {
    if (index !== undefined) {
      const updated = selectedProductImages.filter((_, i) => i !== index)
      setSelectedProductImages(updated)
      setUploadedImage(updated.length > 0 ? updated[0] : null)
    } else {
      setUploadedImage(null)
      setSelectedProductImages([])
    }
    setGeneratedVideo(null)
    setVideoBase64(null)
    setError(null)
  }

  const handleR2VGenerate = async () => {
    if (selectedProductImages.length === 0 && !uploadedImage) {
      setError('Please upload product images')
      return
    }

    const imagesToProcess = selectedProductImages.length > 0
      ? selectedProductImages
      : (uploadedImage ? [uploadedImage] : [])

    if (imagesToProcess.length === 0) {
      setError('No images to process')
      return
    }

    setIsLoading(true)
    setError(null)

    try {
      // Convert images to base64 (handle both data URLs and gallery URLs)
      const imagesBase64 = await Promise.all(
        imagesToProcess.map(async (img) => {
          if (img.startsWith('data:')) {
            return dataUrlToBase64(img)
          }
          // Fetch from gallery URL
          const response = await fetch(img)
          const blob = await response.blob()
          return new Promise<string>((resolve, reject) => {
            const reader = new FileReader()
            reader.onloadend = () => {
              const result = reader.result as string
              resolve(dataUrlToBase64(result))
            }
            reader.onerror = reject
            reader.readAsDataURL(blob)
          })
        })
      )

      const response = await runOtherSpinningPipeline(imagesBase64)

      if (response.video_base64) {
        setVideoBase64(response.video_base64)
        const videoUrl = r2vBase64ToVideoUrl(response.video_base64)
        setGeneratedVideo(videoUrl)
      }
    } catch (err) {
      console.error('Error generating 360 spin:', err)
      setError(err instanceof Error ? err.message : 'Failed to generate 360° spin')
    } finally {
      setIsLoading(false)
    }
  }

  const handleSelectFromGallery = () => {
    setShowGallery(true)
  }

  const handleSelectProduct = (template: Template) => {
    if (template.product_images) {
      // For R2V standard mode
      setSelectedProductImages(template.product_images)
      setUploadedImage(template.product_images[0])

      // For Interpolation mode - populate the 4 views
      if (template.product_images.length >= 4) {
        setViewImages({
          right: template.product_images[0],
          front: template.product_images[1],
          left: template.product_images[2],
          back: template.product_images[3],
        })
      } else {
        // If less than 4 images, fill what we can
        setViewImages({
          right: template.product_images[0] || null,
          front: template.product_images[1] || null,
          left: template.product_images[2] || null,
          back: template.product_images[3] || null,
        })
      }
    } else if (template.product_img) {
      setUploadedImage(template.product_img)
      setSelectedProductImages([])
    }
    setShowGallery(false)
    setGeneratedVideo(null)
    setVideoBase64(null)
    setError(null)
  }

  // Interpolation handlers
  const handleInterpolationFileUpload = (view: keyof ViewImages, file: File) => {
    const reader = new FileReader()
    reader.onloadend = () => {
      setViewImages(prev => ({
        ...prev,
        [view]: reader.result as string
      }))
      setGeneratedVideo(null)
      setVideoBase64(null)
      setError(null)
      setProcessingStep('idle')
    }
    reader.readAsDataURL(file)
  }

  const handleInterpolationRemoveView = (view: keyof ViewImages) => {
    setViewImages(prev => ({
      ...prev,
      [view]: null
    }))
  }

  const handleInterpolationGenerate = async () => {
    const images = [viewImages.right, viewImages.front, viewImages.left, viewImages.back].filter(Boolean) as string[]

    if (images.length < 2) {
      setError('Please upload at least 2 views to generate 360° spin')
      return
    }

    setIsLoading(true)
    setError(null)

    try {
      // Convert images to data URLs (handle both data URLs and gallery URLs)
      const imagesDataUrls = await Promise.all(
        images.map(async (img) => {
          if (img.startsWith('data:')) {
            return img
          }
          // Fetch from gallery URL
          const response = await fetch(img)
          const blob = await response.blob()
          return new Promise<string>((resolve, reject) => {
            const reader = new FileReader()
            reader.onloadend = () => {
              resolve(reader.result as string)
            }
            reader.onerror = reject
            reader.readAsDataURL(blob)
          })
        })
      )

      const imageFiles = imagesDataUrls.map((dataUrl, index) =>
        dataUrlToFile(dataUrl, `view_${index}.png`)
      )

      setProcessingStep('preprocessing')
      const preprocessResponse = await preprocessImages(imageFiles)
      const processedImageFiles = preprocessResponse.images.map((img) =>
        dataUrlToFile(`data:image/png;base64,${img.data}`, `processed_${img.index}.png`)
      )

      setProcessingStep('generating')
      const generateResponse = await generateAllVideos(
        processedImageFiles,
        '',
        '#FFFFFF'
      )

      if (generateResponse.num_failed > 0) {
        console.warn(`${generateResponse.num_failed} video segments failed validation`)
      }

      const videoBlobs = generateResponse.videos.map((video) =>
        base64ToBlob(video.data)
      )

      setProcessingStep('merging')
      const weights = new Array(videoBlobs.length).fill(1)
      const mergedVideoBlob = await mergeVideos(
        videoBlobs,
        weights
      )

      const reader = new FileReader()
      reader.onloadend = () => {
        const base64Data = (reader.result as string).split(',')[1]
        setVideoBase64(base64Data)
        const videoUrl = URL.createObjectURL(mergedVideoBlob)
        setGeneratedVideo(videoUrl)
        setProcessingStep('complete')
      }
      reader.readAsDataURL(mergedVideoBlob)

    } catch (err) {
      console.error('Error generating 360 spin:', err)
      setError(err instanceof Error ? err.message : 'Failed to generate 360° spin')
      setProcessingStep('idle')
    } finally {
      setIsLoading(false)
    }
  }

  const uploadedViewsCount = Object.values(viewImages).filter(Boolean).length
  const hasEnoughViews = uploadedViewsCount >= 2

  const getLoadingMessage = () => {
    switch (processingStep) {
      case 'preprocessing':
        return 'Preprocessing images...'
      case 'generating':
        return 'Generating video segments...'
      case 'merging':
        return 'Merging final video...'
      default:
        return 'Generating...'
    }
  }

  const views: Array<{ key: keyof ViewImages; label: string; icon: JSX.Element }> = [
    {
      key: 'right',
      label: 'Right View',
      icon: (
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
          <path d="M8 4h8a2 2 0 0 1 2 2v12a2 2 0 0 1-2 2H8" stroke="currentColor" strokeWidth="1.5"/>
          <path d="M14 12l4-2v4l-4-2z" fill="currentColor"/>
        </svg>
      )
    },
    {
      key: 'front',
      label: 'Front View',
      icon: (
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
          <rect x="8" y="4" width="8" height="16" rx="1" stroke="currentColor" strokeWidth="1.5"/>
          <circle cx="12" cy="10" r="2" stroke="currentColor" strokeWidth="1.5"/>
          <path d="M10 14h4M10 17h4" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
        </svg>
      )
    },
    {
      key: 'left',
      label: 'Left View',
      icon: (
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
          <path d="M16 4H8a2 2 0 0 0-2 2v12a2 2 0 0 0 2 2h8" stroke="currentColor" strokeWidth="1.5"/>
          <path d="M10 12l-4-2v4l4-2z" fill="currentColor"/>
        </svg>
      )
    },
    {
      key: 'back',
      label: 'Back View',
      icon: (
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
          <rect x="8" y="4" width="8" height="16" rx="1" stroke="currentColor" strokeWidth="1.5"/>
          <path d="M10 6h4M10 9h4M10 12h4" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
        </svg>
      )
    }
  ]

  return (
    <div className="grid grid-cols-[480px_1fr] gap-6">
      {/* Form Panel */}
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

        {/* Mode Toggle */}
        <div className="flex items-center gap-1 p-1 rounded-xl bg-black/[0.03] dark:bg-white/[0.04] mb-6">
          <button
            onClick={() => setSpinMode('r2v-standard')}
            className={`flex-1 px-4 py-2.5 rounded-lg text-button font-medium transition-all duration-200 flex items-center justify-center gap-2 ${
              spinMode === 'r2v-standard'
                ? 'bg-white dark:bg-white/10 shadow-sm text-gm-text-primary-light dark:text-gm-text-primary'
                : 'text-gm-text-tertiary-light dark:text-gm-text-tertiary hover:text-gm-text-primary-light dark:hover:text-gm-text-primary'
            }`}
          >
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" className="opacity-70">
              <rect x="3" y="3" width="7" height="7" rx="1" stroke="currentColor" strokeWidth="2"/>
              <rect x="14" y="3" width="7" height="7" rx="1" stroke="currentColor" strokeWidth="2"/>
              <rect x="3" y="14" width="7" height="7" rx="1" stroke="currentColor" strokeWidth="2"/>
              <rect x="14" y="14" width="7" height="7" rx="1" stroke="currentColor" strokeWidth="2"/>
            </svg>
            Reference to Video
          </button>
          <button
            onClick={() => setSpinMode('interpolation')}
            className={`flex-1 px-4 py-2.5 rounded-lg text-button font-medium transition-all duration-200 flex items-center justify-center gap-2 ${
              spinMode === 'interpolation'
                ? 'bg-white dark:bg-white/10 shadow-sm text-gm-text-primary-light dark:text-gm-text-primary'
                : 'text-gm-text-tertiary-light dark:text-gm-text-tertiary hover:text-gm-text-primary-light dark:hover:text-gm-text-primary'
            }`}
          >
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" className="opacity-70">
              <path d="M23 4v6h-6" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
              <path d="M1 20v-6h6" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
              <path d="M3.51 9a9 9 0 0114.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0020.49 15" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
            Interpolation
          </button>
        </div>

        {spinMode === 'r2v-standard' ? (
          /* R2V Standard Mode */
          <Product360Form
            selectedProductImages={selectedProductImages}
            uploadedImage={uploadedImage}
            onFileUpload={handleR2VFileUpload}
            onRemoveFile={handleR2VRemoveFile}
            onSelectFromGallery={handleSelectFromGallery}
            onGenerate={handleR2VGenerate}
            isLoading={isLoading}
            showPanel={false}
            showProductSelector={false}
          />
        ) : (
          /* Interpolation Mode */
          <div className="space-y-6">
            <section>
              <div className="flex items-center justify-between mb-3">
                <div>
                  <h3 className="text-[15px] font-semibold">Product Views</h3>
                  <p className="text-caption text-gm-text-tertiary-light dark:text-gm-text-tertiary mt-0.5">
                    Upload 2-4 views (right, front, left, back)
                  </p>
                </div>
                <button
                  onClick={handleSelectFromGallery}
                  className="text-caption text-gm-accent hover:text-gm-accent-hover transition-colors flex items-center gap-1"
                >
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" className="opacity-70">
                    <rect x="3" y="3" width="7" height="7" rx="1" stroke="currentColor" strokeWidth="2"/>
                    <rect x="14" y="3" width="7" height="7" rx="1" stroke="currentColor" strokeWidth="2"/>
                    <rect x="3" y="14" width="7" height="7" rx="1" stroke="currentColor" strokeWidth="2"/>
                    <rect x="14" y="14" width="7" height="7" rx="1" stroke="currentColor" strokeWidth="2"/>
                  </svg>
                  From gallery
                </button>
              </div>
            </section>

            <section className="grid grid-cols-4 gap-2">
              {views.map(({ key, label }) => {
                const image = viewImages[key]

                return (
                  <div key={key}>
                    {image ? (
                      <div className="relative aspect-square rounded-lg overflow-hidden bg-white dark:bg-white/10 group border border-black/[0.08] dark:border-white/10">
                        <img
                          src={image}
                          alt={label}
                          className="w-full h-full object-contain p-1"
                        />
                        <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/60 to-transparent py-1">
                          <p className="text-[9px] text-white text-center font-medium">{label}</p>
                        </div>
                        <button
                          onClick={() => handleInterpolationRemoveView(key)}
                          className="absolute top-1 right-1 w-4 h-4 bg-black/60 hover:bg-black/80 backdrop-blur-sm rounded-full flex items-center justify-center transition-all opacity-0 group-hover:opacity-100"
                        >
                          <svg width="7" height="7" viewBox="0 0 24 24" fill="none">
                            <path d="M18 6L6 18M6 6L18 18" stroke="white" strokeWidth="2.5" strokeLinecap="round"/>
                          </svg>
                        </button>
                      </div>
                    ) : (
                      <label className="block cursor-pointer">
                        <input
                          type="file"
                          accept="image/*"
                          className="hidden"
                          onChange={(e: ChangeEvent<HTMLInputElement>) => {
                            const file = e.target.files?.[0]
                            if (file) handleInterpolationFileUpload(key, file)
                          }}
                        />
                        <div className="aspect-square rounded-lg border border-dashed border-black/[0.15] dark:border-white/[0.15] hover:border-gm-accent/50 dark:hover:border-gm-accent/50 transition-all flex flex-col items-center justify-center gap-1 bg-white dark:bg-white/[0.02] hover:bg-gm-accent/5 p-2">
                          <div className="w-5 h-5 rounded bg-gm-accent/10 dark:bg-gm-accent/20 flex items-center justify-center">
                            <svg width="10" height="10" viewBox="0 0 24 24" fill="none" className="text-gm-accent">
                              <path d="M12 15V3M12 3L8 7M12 3L16 7" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                              <path d="M2 17v2a2 2 0 002 2h16a2 2 0 002-2v-2" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
                            </svg>
                          </div>
                          <p className="text-[9px] text-gm-text-tertiary-light dark:text-gm-text-tertiary text-center leading-tight font-medium">
                            {label.replace(' View', '')}
                          </p>
                        </div>
                      </label>
                    )}
                  </div>
                )
              })}
            </section>

            <div className="flex items-center gap-2 px-3 py-2 rounded-lg bg-black/[0.03] dark:bg-white/[0.04]">
              <div className="flex gap-1">
                {views.map(({ key }) => (
                  <div
                    key={key}
                    className={`w-2 h-2 rounded-full transition-all ${
                      viewImages[key]
                        ? 'bg-gm-accent'
                        : 'bg-black/[0.15] dark:bg-white/[0.15]'
                    }`}
                  />
                ))}
              </div>
              <span className="text-caption text-gm-text-tertiary-light dark:text-gm-text-tertiary">
                {uploadedViewsCount} {uploadedViewsCount === 1 ? 'view' : 'views'} uploaded {uploadedViewsCount >= 2 ? '(ready)' : '(min. 2)'}
              </span>
            </div>

            {error && (
              <div className="px-3 py-2 rounded-lg bg-red-500/10 border border-red-500/20 text-red-500 text-caption">
                {error}
              </div>
            )}

            <button
              onClick={handleInterpolationGenerate}
              disabled={!hasEnoughViews || isLoading}
              className="btn-primary w-full py-3.5 rounded-xl text-button font-medium flex flex-col items-center justify-center gap-1 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {isLoading ? (
                <>
                  <div className="flex items-center gap-2">
                    <svg className="animate-spin" width="18" height="18" viewBox="0 0 24 24" fill="none">
                      <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="2" opacity="0.25"/>
                      <path d="M12 2a10 10 0 0 1 10 10" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
                    </svg>
                    {getLoadingMessage()}
                  </div>
                  {processingStep !== 'idle' && (
                    <div className="text-[10px] opacity-70">
                      {processingStep === 'preprocessing' && 'Removing backgrounds & enhancing...'}
                      {processingStep === 'generating' && `Creating ${uploadedViewsCount} video segments...`}
                      {processingStep === 'merging' && 'Combining into 360° video...'}
                    </div>
                  )}
                </>
              ) : (
                <>
                  <div className="flex items-center gap-2">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none">
                      <path d="M23 4v6h-6" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                      <path d="M1 20v-6h6" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                      <path d="M3.51 9a9 9 0 0114.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0020.49 15" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                    </svg>
                    Generate 360° Spin
                  </div>
                </>
              )}
            </button>
          </div>
        )}
      </div>

      {/* Preview Panel */}
      <Product360Preview
        showVideo={showVideo}
        isLoading={isLoading}
        videoUrl={generatedVideo}
        videoBase64={videoBase64}
        error={error}
        flow={spinMode === 'interpolation' ? 'interpolation-other' : 'r2v-other'}
      />

      {/* Gallery Modal */}
      {showGallery && (
        <div
          className="fixed inset-0 bg-black/70 backdrop-blur-sm flex items-center justify-center z-[9999] p-6"
          onClick={() => setShowGallery(false)}
        >
          <div
            className="rounded-2xl p-6 max-w-4xl w-full max-h-[85vh] overflow-hidden flex flex-col"
            style={{
              background: 'var(--dropdown-bg)',
              border: '1px solid var(--dropdown-border)'
            }}
            onClick={e => e.stopPropagation()}
          >
            {/* Header */}
            <div className="flex items-center justify-between mb-5">
              <div>
                <h2 className="text-xl font-semibold">Product 360 Gallery</h2>
                <p className="text-caption text-gm-text-tertiary-light dark:text-gm-text-tertiary mt-1">
                  Select a product template
                </p>
              </div>
              <button
                onClick={() => setShowGallery(false)}
                className="w-9 h-9 rounded-lg bg-black/[0.06] hover:bg-black/[0.1] dark:bg-white/10 dark:hover:bg-white/20 flex items-center justify-center transition-all"
              >
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none">
                  <path d="M18 6L6 18M6 6L18 18" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
                </svg>
              </button>
            </div>

            {/* Grid */}
            <div className="overflow-y-auto flex-1 -mx-2 px-2">
              <div className="grid grid-cols-2 lg:grid-cols-3 gap-4">
                {templates.filter(template => template.product_type && template.product_type !== 'shoes' && (spinMode === 'r2v-standard' ? template.product_type === 'clothes' : ['cars', 'smartphones'].includes(template.product_type))).map((template, index) => {
                  const hasVideo = template.video_path
                  const hasMultipleImages = template.product_images && template.product_images.length > 0
                  const displayImages = hasMultipleImages ? template.product_images : (template.product_img ? [template.product_img] : [])

                  return (
                    <div
                      key={index}
                      onClick={() => handleSelectProduct(template)}
                      className="group cursor-pointer rounded-xl overflow-hidden bg-black/[0.04] dark:bg-white/[0.04] border border-transparent hover:border-gm-accent/50 transition-all"
                    >
                      {/* Video/Image Preview */}
                      <div className="relative aspect-video bg-black/5 dark:bg-black/20">
                        {hasVideo ? (
                          <video
                            src={template.video_path}
                            autoPlay
                            loop
                            muted
                            playsInline
                            className="w-full h-full object-cover"
                          />
                        ) : displayImages && displayImages[0] ? (
                          <img
                            src={displayImages[0]}
                            alt={template.product_name || 'Product'}
                            className="w-full h-full object-contain bg-white/5 dark:bg-white/10"
                          />
                        ) : (
                          <div className="w-full h-full flex items-center justify-center text-gm-text-tertiary">
                            No preview
                          </div>
                        )}

                        {/* Hover overlay */}
                        <div className="absolute inset-0 bg-gradient-to-t from-black/60 to-transparent opacity-0 group-hover:opacity-100 transition-all pointer-events-none" />

                        {/* Select indicator */}
                        <div className="absolute top-3 right-3 w-8 h-8 rounded-full bg-white/90 dark:bg-white/20 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-all scale-75 group-hover:scale-100">
                          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" className="text-gm-bg dark:text-white">
                            <path d="M5 12l5 5L20 7" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"/>
                          </svg>
                        </div>

                        {/* Product info overlay */}
                        <div className="absolute bottom-0 left-0 right-0 p-3 bg-gradient-to-t from-black/80 to-transparent">
                          <div className="flex items-center gap-2">
                            {displayImages && displayImages[0] && (
                              <img
                                src={displayImages[0]}
                                alt="Product"
                                className="w-6 h-6 rounded object-contain bg-white"
                              />
                            )}
                            <span className="text-[13px] font-medium text-white">
                              {template.product_name || `Product ${index + 1}`}
                            </span>
                          </div>
                        </div>
                      </div>
                    </div>
                  )
                })}
              </div>

              {templates.filter(template => template.product_type && template.product_type !== 'shoes' && (spinMode === 'r2v-standard' ? template.product_type === 'clothes' : ['cars', 'smartphones'].includes(template.product_type))).length === 0 && (
                <div className="text-center py-12 text-gm-text-tertiary-light dark:text-gm-text-tertiary">
                  <p>No templates available for this product</p>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default OtherSpinning
