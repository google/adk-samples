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
import CameraCapture from '../../components/CameraCapture'
import ProductSelector from '../../components/ProductSelector'
import type { Product } from '../../config/featureConstraints'
import { MODEL_PRESETS } from '../../shared/modelGallery'

export interface GlassesVTOFormData {
  glassesImage: string | null
  modelImage: string | null
}

interface GlassesTemplate {
  video_path: string
  product_img?: string
  product_images?: string[]
}

interface GlassesTemplatesData {
  men: GlassesTemplate[]
  women: GlassesTemplate[]
  gallery?: GlassesTemplate[]
}

interface GlassesImageVTOFormProps {
  onSubmit: (data: GlassesVTOFormData) => void
  isGenerating?: boolean
  uploadedImage?: string | null
  onRemoveImage?: () => void
  currentProduct?: Product
  availableProducts?: Product[]
  onProductChange?: (product: Product) => void
  prefilledModelImage?: string
  prefilledGlassesImage?: string
}

function GlassesImageVTOForm({
  onSubmit,
  isGenerating = false,
  uploadedImage,
  onRemoveImage,
  currentProduct,
  availableProducts,
  onProductChange,
  prefilledModelImage,
  prefilledGlassesImage
}: GlassesImageVTOFormProps) {
  const [glassesImage, setGlassesImage] = useState<string | null>(null)
  const [modelImage, setModelImage] = useState<string | null>(null)
  const [isCameraOpen, setIsCameraOpen] = useState(false)
  const [showGallery, setShowGallery] = useState(false)
  const [showModelGallery, setShowModelGallery] = useState(false)
  const [glassesTemplates, setGlassesTemplates] = useState<GlassesTemplatesData>({ men: [], women: [] })
  const [selectedFromGallery, setSelectedFromGallery] = useState(false)

  // Load glasses templates on mount
  useEffect(() => {
    fetch('/products/glasses/templates.json')
      .then(response => response.json())
      .then(data => {
        if (data && data.men && data.women) {
          setGlassesTemplates(data)
        }
      })
      .catch(error => console.error('Error loading glasses templates:', error))
  }, [])

  // Pre-fill glasses image from uploaded image or prefilled prop
  useEffect(() => {
    if (prefilledGlassesImage && !glassesImage) {
      setGlassesImage(prefilledGlassesImage)
    } else if (uploadedImage && !glassesImage) {
      setGlassesImage(uploadedImage)
    }
  }, [uploadedImage, prefilledGlassesImage, glassesImage])

  // Pre-fill model image from prefilled prop
  useEffect(() => {
    if (prefilledModelImage && !modelImage) {
      setModelImage(prefilledModelImage)
    }
  }, [prefilledModelImage, modelImage])

  const handleImageUpload = (setter: (img: string | null) => void) => (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      const reader = new FileReader()
      reader.onloadend = () => {
        setter(reader.result as string)
      }
      reader.readAsDataURL(file)
    }
  }

  const handleRemoveImage = (setter: (img: string | null) => void, isPrefilledImage?: boolean) => () => {
    setter(null)
    setSelectedFromGallery(false)
    if (isPrefilledImage && onRemoveImage) {
      onRemoveImage()
    }
  }

  const handleSelectFromGallery = (template: GlassesTemplate) => {
    if (template.product_img) {
      setGlassesImage(template.product_img)
      setSelectedFromGallery(true)
      setShowGallery(false)
    }
  }

  // Get unique glasses by product_img to avoid duplicates
  const allTemplates = [...glassesTemplates.women, ...glassesTemplates.men, ...(glassesTemplates.gallery ?? [])]
  const uniqueGlasses = allTemplates.reduce((acc, template) => {
    const imgPath = template.product_img
    if (imgPath && !acc.some(t => t.product_img === imgPath)) {
      acc.push(template)
    }
    return acc
  }, [] as GlassesTemplate[])

  const handleCameraCapture = (imageDataUrl: string) => {
    setModelImage(imageDataUrl)
    setIsCameraOpen(false)
  }

  const convertToDataUrl = async (imagePath: string): Promise<string> => {
    if (imagePath.startsWith('data:')) return imagePath
    const response = await fetch(imagePath)
    let blob = await response.blob()
    if (blob.type === '' || blob.type === 'application/octet-stream') {
      const ext = imagePath.split('.').pop()?.toLowerCase()
      const mimeMap: Record<string, string> = { jpg: 'image/jpeg', jpeg: 'image/jpeg', png: 'image/png', webp: 'image/webp' }
      blob = new Blob([blob], { type: mimeMap[ext || ''] || 'image/jpeg' })
    }
    return new Promise((resolve, reject) => {
      const reader = new FileReader()
      reader.onloadend = () => resolve(reader.result as string)
      reader.onerror = reject
      reader.readAsDataURL(blob)
    })
  }

  const handleGenerate = async () => {
    try {
      const glassesDataUrl = glassesImage ? await convertToDataUrl(glassesImage) : null
      const modelDataUrl = modelImage ? await convertToDataUrl(modelImage) : null
      onSubmit({
        glassesImage: glassesDataUrl,
        modelImage: modelDataUrl
      })
    } catch (err) {
      console.error('Failed to convert images:', err)
    }
  }

  const isFormValid = glassesImage && modelImage
  const buttonDisabled = isGenerating || !isFormValid

  const renderImageUpload = (
    image: string | null,
    setter: (img: string | null) => void,
    label: string,
    description: string,
    isPrefilledImage?: boolean
  ) => {
    if (image) {
      return (
        <div className="relative w-32 h-32 rounded-xl overflow-hidden bg-black/[0.02] dark:bg-white/[0.02] group">
          <img
            src={image}
            alt={label}
            className="w-full h-full object-cover"
          />
          <div className="absolute inset-0 bg-gradient-to-t from-black/50 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-200" />
          <button
            onClick={handleRemoveImage(setter, isPrefilledImage)}
            disabled={isGenerating}
            className="absolute top-2 right-2 w-7 h-7 bg-black/50 hover:bg-black/70 backdrop-blur-sm rounded-full flex items-center justify-center transition-all opacity-0 group-hover:opacity-100 disabled:opacity-50"
          >
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none">
              <path d="M18 6L6 18M6 6L18 18" stroke="white" strokeWidth="2" strokeLinecap="round"/>
            </svg>
          </button>
        </div>
      )
    }

    return (
      <label className={`upload-zone-compact cursor-pointer ${isGenerating ? 'opacity-50 pointer-events-none' : ''}`}>
        <input
          type="file"
          accept="image/*"
          className="hidden"
          onChange={handleImageUpload(setter)}
          disabled={isGenerating}
        />
        <div className="w-10 h-10 rounded-lg bg-gm-accent/10 dark:bg-gm-accent/20 flex items-center justify-center">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" className="text-gm-accent">
            <path d="M12 15V3M12 3L8 7M12 3L16 7" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
            <path d="M2 17v2a2 2 0 002 2h16a2 2 0 002-2v-2" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
          </svg>
        </div>
        <div>
          <p className="text-button font-medium">Drop file or click to upload</p>
          <p className="text-[11px] text-gm-text-tertiary-light dark:text-gm-text-tertiary mt-0.5">{description}</p>
        </div>
      </label>
    )
  }

  const renderModelImageSection = () => {
    if (modelImage) {
      return (
        <div className="relative w-32 h-32 rounded-xl overflow-hidden bg-black/[0.02] dark:bg-white/[0.02] group">
          <img
            src={modelImage}
            alt="Model photo"
            className="w-full h-full object-cover"
          />
          <div className="absolute inset-0 bg-gradient-to-t from-black/50 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-200" />
          <button
            onClick={() => setModelImage(null)}
            disabled={isGenerating}
            className="absolute top-2 right-2 w-7 h-7 bg-black/50 hover:bg-black/70 backdrop-blur-sm rounded-full flex items-center justify-center transition-all opacity-0 group-hover:opacity-100 disabled:opacity-50"
          >
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none">
              <path d="M18 6L6 18M6 6L18 18" stroke="white" strokeWidth="2" strokeLinecap="round"/>
            </svg>
          </button>
        </div>
      )
    }

    return (
      <div className="flex gap-3">
        {/* Upload option */}
        <label className={`upload-zone-compact cursor-pointer flex-1 ${isGenerating ? 'opacity-50 pointer-events-none' : ''}`}>
          <input
            type="file"
            accept="image/*"
            className="hidden"
            onChange={handleImageUpload(setModelImage)}
            disabled={isGenerating}
          />
          <div className="w-10 h-10 rounded-lg bg-gm-accent/10 dark:bg-gm-accent/20 flex items-center justify-center">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" className="text-gm-accent">
              <path d="M12 15V3M12 3L8 7M12 3L16 7" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
              <path d="M2 17v2a2 2 0 002 2h16a2 2 0 002-2v-2" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
            </svg>
          </div>
          <div>
            <p className="text-button font-medium">Upload photo</p>
            <p className="text-[11px] text-gm-text-tertiary-light dark:text-gm-text-tertiary mt-0.5">PNG, JPG</p>
          </div>
        </label>

        {/* Camera option */}
        <button
          onClick={() => setIsCameraOpen(true)}
          disabled={isGenerating}
          className={`upload-zone-compact cursor-pointer flex-1 ${isGenerating ? 'opacity-50 pointer-events-none' : ''}`}
        >
          <div className="w-10 h-10 rounded-lg bg-gm-accent/10 dark:bg-gm-accent/20 flex items-center justify-center">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" className="text-gm-accent">
              <circle cx="12" cy="12" r="3" stroke="currentColor" strokeWidth="1.5"/>
              <path d="M3 9a2 2 0 012-2h1.5a1 1 0 00.8-.4l1.4-1.8a1 1 0 01.8-.4h5a1 1 0 01.8.4l1.4 1.8a1 1 0 00.8.4H19a2 2 0 012 2v9a2 2 0 01-2 2H5a2 2 0 01-2-2V9z" stroke="currentColor" strokeWidth="1.5"/>
            </svg>
          </div>
          <div>
            <p className="text-button font-medium">Take photo</p>
            <p className="text-[11px] text-gm-text-tertiary-light dark:text-gm-text-tertiary mt-0.5">Use camera</p>
          </div>
        </button>
      </div>
    )
  }

  return (
    <>
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
          {/* Section 1: Glasses Image */}
          <section>
            <div className="flex items-center justify-between mb-3">
              <div>
                <h3 className="text-[15px] font-semibold">Glasses Image</h3>
                <p className="text-caption text-gm-text-tertiary-light dark:text-gm-text-tertiary mt-0.5">
                  Product image of the glasses
                </p>
              </div>
              <button
                onClick={() => setShowGallery(true)}
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

            {selectedFromGallery && glassesImage ? (
              <div className="relative w-32 h-32 rounded-xl overflow-hidden bg-white dark:bg-white/10 group">
                <img
                  src={glassesImage}
                  alt="Glasses"
                  className="w-full h-full object-contain p-2"
                />
                <button
                  onClick={() => {
                    setGlassesImage(null)
                    setSelectedFromGallery(false)
                  }}
                  disabled={isGenerating}
                  className="absolute top-2 right-2 w-7 h-7 bg-black/50 hover:bg-black/70 backdrop-blur-sm rounded-full flex items-center justify-center transition-all opacity-0 group-hover:opacity-100 disabled:opacity-50"
                >
                  <svg width="12" height="12" viewBox="0 0 24 24" fill="none">
                    <path d="M18 6L6 18M6 6L18 18" stroke="white" strokeWidth="2" strokeLinecap="round"/>
                  </svg>
                </button>
              </div>
            ) : (
              renderImageUpload(glassesImage, setGlassesImage, 'Glasses', 'PNG, JPG up to 10MB', glassesImage === uploadedImage)
            )}
          </section>

          {/* Divider */}
          <div className="divider" />

          {/* Section 2: Model Image with Camera option */}
          <section>
            <div className="flex items-center justify-between mb-3">
              <div>
                <h3 className="text-[15px] font-semibold">Model Image</h3>
                <p className="text-caption text-gm-text-tertiary-light dark:text-gm-text-tertiary mt-0.5">
                  Front face photo
                </p>
              </div>
              <button
                onClick={() => setShowModelGallery(true)}
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

            {renderModelImageSection()}
          </section>

          {/* Generate Button */}
          <button
            type="button"
            onClick={handleGenerate}
            disabled={buttonDisabled}
            className="btn-primary w-full py-3.5 rounded-xl text-button font-medium flex items-center justify-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {isGenerating ? (
              <>
                <svg className="animate-spin h-5 w-5" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none"/>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"/>
                </svg>
                Generating...
              </>
            ) : (
              <>
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none">
                  <path d="M12 2L2 7l10 5 10-5-10-5z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                  <path d="M2 17l10 5 10-5M2 12l10 5 10-5" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                </svg>
                Generate VTO
              </>
            )}
          </button>

          {/* Validation hint */}
          {!isFormValid && (
            <p className="text-caption text-center text-gm-text-tertiary-light dark:text-gm-text-tertiary">
              Please upload glasses and model images to continue
            </p>
          )}
        </div>
      </div>

      {/* Camera Modal */}
      <CameraCapture
        isOpen={isCameraOpen}
        onCapture={handleCameraCapture}
        onClose={() => setIsCameraOpen(false)}
      />

      {/* Gallery Modal */}
      {showGallery && (
        <div
          className="fixed inset-0 bg-black/70 backdrop-blur-sm flex items-center justify-center z-[9999] p-6"
          onClick={() => setShowGallery(false)}
        >
          <div
            className="rounded-2xl p-6 max-w-2xl w-full max-h-[75vh] overflow-hidden flex flex-col"
            style={{
              background: 'var(--dropdown-bg)',
              border: '1px solid var(--dropdown-border)'
            }}
            onClick={e => e.stopPropagation()}
          >
            {/* Header */}
            <div className="flex items-center justify-between mb-5">
              <div>
                <h2 className="text-xl font-semibold">Eyewear Gallery</h2>
                <p className="text-caption text-gm-text-tertiary-light dark:text-gm-text-tertiary mt-1">
                  Select a product from the collection
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
              <div className="grid grid-cols-3 gap-4">
                {uniqueGlasses.map((template, index) => {
                  const displayImage = template.product_img

                  return (
                    <div
                      key={index}
                      onClick={() => handleSelectFromGallery(template)}
                      className="group cursor-pointer rounded-xl overflow-hidden bg-white dark:bg-white/10 border border-transparent hover:border-gm-accent/50 transition-all"
                    >
                      {/* Image Only */}
                      <div className="relative aspect-square">
                        {displayImage ? (
                          <img
                            src={displayImage}
                            alt="Product"
                            className="w-full h-full object-contain p-3"
                          />
                        ) : (
                          <div className="w-full h-full flex items-center justify-center text-gm-text-tertiary">
                            No preview
                          </div>
                        )}
                        {/* Hover overlay */}
                        <div className="absolute inset-0 bg-black/0 group-hover:bg-black/20 transition-all flex items-center justify-center">
                          <div className="w-10 h-10 rounded-full bg-white/90 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-all scale-75 group-hover:scale-100">
                            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" className="text-gm-bg">
                              <path d="M5 12l5 5L20 7" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"/>
                            </svg>
                          </div>
                        </div>
                      </div>
                    </div>
                  )
                })}
              </div>

              {uniqueGlasses.length === 0 && (
                <div className="text-center py-12 text-gm-text-tertiary-light dark:text-gm-text-tertiary">
                  <p>Loading...</p>
                </div>
              )}
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
            onClick={e => e.stopPropagation()}
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
                  <path d="M18 6L6 18M6 6L18 18" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
                </svg>
              </button>
            </div>

            <div className="overflow-y-auto flex-1 -mx-2 px-2">
              <div className="mb-6">
                <h3 className="text-sm font-semibold text-gm-text-secondary-light dark:text-gm-text-secondary mb-3 px-2">Women</h3>
                <div className="grid grid-cols-3 lg:grid-cols-4 gap-4">
                  {MODEL_PRESETS.filter(m => m.gender === 'woman').map((preset) => (
                    <div
                      key={preset.id}
                      onClick={() => { setModelImage(preset.thumbnail); setShowModelGallery(false) }}
                      className="group cursor-pointer rounded-xl overflow-hidden bg-white dark:bg-white/10 border border-transparent hover:border-gm-accent/50 transition-all"
                    >
                      <div className="relative aspect-[3/4]">
                        <img src={preset.thumbnail} alt={`${preset.label} woman`} className="w-full h-full object-cover" />
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
                <h3 className="text-sm font-semibold text-gm-text-secondary-light dark:text-gm-text-secondary mb-3 px-2">Men</h3>
                <div className="grid grid-cols-3 lg:grid-cols-4 gap-4">
                  {MODEL_PRESETS.filter(m => m.gender === 'man').map((preset) => (
                    <div
                      key={preset.id}
                      onClick={() => { setModelImage(preset.thumbnail); setShowModelGallery(false) }}
                      className="group cursor-pointer rounded-xl overflow-hidden bg-white dark:bg-white/10 border border-transparent hover:border-gm-accent/50 transition-all"
                    >
                      <div className="relative aspect-[3/4]">
                        <img src={preset.thumbnail} alt={`${preset.label} man`} className="w-full h-full object-cover" />
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
    </>
  )
}

export default GlassesImageVTOForm
