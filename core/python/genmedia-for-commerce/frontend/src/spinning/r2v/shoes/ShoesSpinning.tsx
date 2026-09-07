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
import type { Product } from '../../../config/featureConstraints'
import Product360Form from '../../components/Product360Form'
import Product360Preview from '../../components/Product360Preview'
import {
  runSpinningPipeline,
  dataUrlToBase64,
  base64ToVideoUrl,
  PipelineResponse,
} from '../../services/shoesSpinningApi'

interface Template {
  product_name?: string
  product_type?: string
  product_img?: string
  product_images?: string[]
  video_path?: string
  video_prompt?: object
}

interface ShoesSpinningProps {
  prefilledImages?: string[]
  showVideo?: boolean
  currentProduct?: Product
  availableProducts?: Product[]
  onProductChange?: (product: Product) => void
}

function ShoesSpinning({
  prefilledImages = [],
  showVideo = true,
  currentProduct,
  availableProducts,
  onProductChange
}: ShoesSpinningProps) {
  const [selectedProductImages, setSelectedProductImages] = useState<string[]>(prefilledImages)
  const [uploadedImage, setUploadedImage] = useState<string | null>(
    prefilledImages.length > 0 ? prefilledImages[0] : null
  )
  const [showGallery, setShowGallery] = useState(false)
  const [templates, setTemplates] = useState<Template[]>([])

  // Generation state
  const [isLoading, setIsLoading] = useState(false)
  const [generatedVideo, setGeneratedVideo] = useState<string | null>(null)
  const [videoBase64, setVideoBase64] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [showClassifierWarning, setShowClassifierWarning] = useState(false)

  useEffect(() => {
    fetch('/templates/360-templates.json')
      .then(response => response.json())
      .then(data => setTemplates(data))
      .catch(error => console.error('Error loading templates:', error))

    fetch('/api/shoes/spinning/config')
      .then(response => response.json())
      .then(data => {
        if (data.using_local_classifier) {
          setShowClassifierWarning(true)
        }
      })
      .catch(error => console.error('Error fetching shoes config:', error))
  }, [])

  const handleFileUpload = (files: File[]) => {
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
          // Clear previous generation when new images are uploaded
          setGeneratedVideo(null)
          setVideoBase64(null)
          setError(null)
        }
      }
      reader.readAsDataURL(file)
    })
  }

  const handleRemoveFile = (index?: number) => {
    if (index !== undefined) {
      const updated = selectedProductImages.filter((_, i) => i !== index)
      setSelectedProductImages(updated)
      setUploadedImage(updated.length > 0 ? updated[0] : null)
    } else {
      setUploadedImage(null)
      setSelectedProductImages([])
    }
    // Clear generation state
    setGeneratedVideo(null)
    setVideoBase64(null)
    setError(null)
  }

  const handleSelectFromGallery = () => {
    setShowGallery(true)
  }

  const handleSelectProduct = (template: Template) => {
    if (template.product_images) {
      setSelectedProductImages(template.product_images)
      setUploadedImage(template.product_images[0])
    } else if (template.product_img) {
      setUploadedImage(template.product_img)
      setSelectedProductImages([])
    }
    setShowGallery(false)
    // Clear previous generation when selecting new product
    setGeneratedVideo(null)
    setVideoBase64(null)
    setError(null)
  }

  const handleGenerate = async () => {
    // Validate that we have images to process
    if (selectedProductImages.length === 0 && !uploadedImage) {
      setError('Please upload product images first')
      return
    }

    // Get images to process
    const imagesToProcess = selectedProductImages.length > 0
      ? selectedProductImages
      : (uploadedImage ? [uploadedImage] : [])

    if (imagesToProcess.length === 0) {
      setError('No images to process')
      return
    }

    setIsLoading(true)
    setError(null)
    setGeneratedVideo(null)
    setVideoBase64(null)

    try {
      // Convert images to base64 (strip data URL prefix if present)
      const imagesBase64 = await Promise.all(
        imagesToProcess.map(async (img) => {
          // If it's a data URL (from FileReader), extract base64
          if (img.startsWith('data:')) {
            return dataUrlToBase64(img)
          }
          // If it's a URL (from gallery), fetch and convert to base64
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

      // Call the API
      const result: PipelineResponse = await runSpinningPipeline({
        images_base64: imagesBase64,
        max_retries: 4,
        upscale_images: true,
      })

      // Store the base64 for download and create blob URL for playback
      setVideoBase64(result.video_base64)
      const videoUrl = base64ToVideoUrl(result.video_base64)
      setGeneratedVideo(videoUrl)
    } catch (err) {
      console.error('Error generating 360 spin:', err)
      setError(err instanceof Error ? err.message : 'Failed to generate video')
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <>
      {/* Local classifier warning popup */}
      {showClassifierWarning && (
        <div className="fixed inset-0 bg-black/70 backdrop-blur-sm flex items-center justify-center z-[9999] p-6">
          <div
            className="rounded-2xl p-6 max-w-md w-full"
            style={{
              background: 'var(--dropdown-bg)',
              border: '1px solid var(--dropdown-border)'
            }}
          >
            <div className="flex items-start gap-3 mb-4">
              <svg width="24" height="24" viewBox="0 0 24 24" fill="none" className="text-yellow-500 flex-shrink-0 mt-0.5">
                <path d="M12 9v4m0 4h.01M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
              </svg>
              <div>
                <h3 className="text-lg font-semibold mb-2">Local Classifier Active</h3>
                <p className="text-sm text-gm-text-secondary-light dark:text-gm-text-secondary leading-relaxed">
                  For optimal results, please fine-tune the shoe classifier model. See documentation for instructions.
                </p>
              </div>
            </div>
            <div className="flex justify-end">
              <button
                onClick={() => setShowClassifierWarning(false)}
                className="px-4 py-2 rounded-lg text-sm font-medium bg-gm-accent text-white hover:opacity-90 transition-opacity"
              >
                Got it
              </button>
            </div>
          </div>
        </div>
      )}

      <div className="grid grid-cols-[480px_1fr] gap-6">
        <Product360Form
          selectedProductImages={selectedProductImages}
          uploadedImage={uploadedImage}
          onFileUpload={handleFileUpload}
          onRemoveFile={handleRemoveFile}
          onSelectFromGallery={handleSelectFromGallery}
          onGenerate={handleGenerate}
          isLoading={isLoading}
          currentProduct={currentProduct}
          availableProducts={availableProducts}
          onProductChange={onProductChange}
        />
        <Product360Preview
          showVideo={showVideo}
          videoUrl={generatedVideo}
          videoBase64={videoBase64}
          isLoading={isLoading}
          error={error}
          flow="r2v-shoes"
        />
      </div>

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
                  Select a product template from GenMedia TV
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
                {templates.filter(template => template.product_type === 'shoes').map((template, index) => {
                  const hasVideo = template.video_path
                  const hasMultipleImages = template.product_images && template.product_images.length > 0
                  const displayImages = hasMultipleImages ? template.product_images : (template.product_img ? [template.product_img] : [])
                  const imageCount = displayImages?.length || 0

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
                        {/* Play overlay on hover */}
                        <div className="absolute inset-0 bg-black/0 group-hover:bg-black/30 transition-all flex items-center justify-center">
                          <div className="w-12 h-12 rounded-full bg-white/90 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-all scale-75 group-hover:scale-100">
                            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" className="text-gm-bg ml-0.5">
                              <path d="M8 5v14l11-7L8 5z" fill="currentColor"/>
                            </svg>
                          </div>
                        </div>
                      </div>

                      {/* Info */}
                      <div className="p-3">
                        <div className="flex items-center gap-2 mb-1.5">
                          {displayImages && displayImages[0] && (
                            <img
                              src={displayImages[0]}
                              alt="Product"
                              className="w-7 h-7 rounded object-contain bg-white"
                            />
                          )}
                          <span className="text-[13px] font-medium text-gm-text-primary-light dark:text-gm-text-primary">
                            {template.product_name || `Product ${index + 1}`}
                          </span>
                        </div>
                        <p className="text-[11px] text-gm-text-tertiary-light dark:text-gm-text-tertiary line-clamp-2 leading-relaxed">
                          {imageCount} {imageCount === 1 ? 'image' : 'images'} • 360° rotation
                        </p>
                      </div>
                    </div>
                  )
                })}
              </div>

              {templates.filter(template => template.product_type === 'shoes').length === 0 && (
                <div className="text-center py-12 text-gm-text-tertiary-light dark:text-gm-text-tertiary">
                  <p>{templates.length === 0 ? 'Loading templates...' : 'No templates available for shoes'}</p>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </>
  )
}

export default ShoesSpinning
