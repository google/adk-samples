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

import { useState, useEffect, ChangeEvent } from 'react'
import { createPortal } from 'react-dom'
import PromptTextarea from '../../components/PromptTextarea'
import CameraCapture from '../../components/CameraCapture'
import type { Product, Capability } from '../../config/featureConstraints'

interface VideoPrompt {
  input_subject?: string
  subject?: string
  action?: string
  scene?: string
  camera_angles_and_movements?: string
  lighting?: string
  negative_prompt?: string
}

interface GlassesTemplate {
  video_path: string
  product_img?: string
  product_images?: string[]
  video_prompt?: VideoPrompt
}

interface GlassesTemplatesData {
  men: GlassesTemplate[]
  women: GlassesTemplate[]
}

interface CustomSceneModeProps {
  product: Product
  capability: Capability
  uploadedImage?: string | null
  onFileUpload: (file: File) => void
  onRemoveImage: () => void
  animationDescription: string
  onAnimationDescriptionChange: (value: string) => void
  onGenerate: () => void
  modelImage?: string | null
  onModelImageChange?: (image: string | null) => void
  onEnhancePrompt?: () => Promise<void>
  isEnhancingPrompt?: boolean
  isGenerating?: boolean
}

function CustomSceneMode({
  product,
  uploadedImage,
  onFileUpload,
  onRemoveImage,
  animationDescription,
  onAnimationDescriptionChange,
  onGenerate,
  modelImage,
  onModelImageChange,
  onEnhancePrompt,
  isEnhancingPrompt = false,
  isGenerating = false
}: CustomSceneModeProps) {
  const [showGallery, setShowGallery] = useState(false)
  const [glassesTemplates, setGlassesTemplates] = useState<GlassesTemplatesData>({ men: [], women: [] })
  const [selectedTemplate, setSelectedTemplate] = useState<GlassesTemplate | null>(null)
  const [isCameraOpen, setIsCameraOpen] = useState(false)
  const [localModelImage, setLocalModelImage] = useState<string | null>(modelImage || null)

  // Sync localModelImage with parent's modelImage prop
  useEffect(() => {
    if (modelImage !== undefined) {
      setLocalModelImage(modelImage)
    }
  }, [modelImage])

  // Load glasses templates on mount
  useEffect(() => {
    if (product === 'glasses') {
      fetch('/products/glasses/templates.json')
        .then(response => response.json())
        .then(data => {
          if (data && data.men && data.women) {
            setGlassesTemplates(data)
          }
        })
        .catch(error => console.error('Error loading glasses templates:', error))
    }
  }, [product])

  const handleSelectTemplate = async (template: GlassesTemplate) => {
    console.log('Template selected:', template.product_img)
    setSelectedTemplate(template)
    setShowGallery(false)

    // Fetch the template image and convert to data URL for the parent component
    if (template.product_img) {
      try {
        const response = await fetch(template.product_img)
        const blob = await response.blob()
        const reader = new FileReader()
        reader.onloadend = () => {
          // Create a fake File object from the blob
          const file = new File([blob], 'template_product.png', { type: blob.type })
          console.log('Template image fetched, calling onFileUpload')
          onFileUpload(file)
        }
        reader.readAsDataURL(blob)
      } catch (error) {
        console.error('Error loading template image:', error)
      }
    }
  }

  const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      onFileUpload(file)
      setSelectedTemplate(null) // Clear template selection when uploading custom image
    }
  }

  const allTemplates = [...glassesTemplates.women, ...glassesTemplates.men]

  // Get unique glasses by product_img to avoid duplicates
  const uniqueGlasses = allTemplates.reduce((acc, template) => {
    const imgPath = template.product_img
    if (imgPath && !acc.some(t => t.product_img === imgPath)) {
      acc.push(template)
    }
    return acc
  }, [] as GlassesTemplate[])

  const handleCameraCapture = (imageDataUrl: string) => {
    console.log('Camera capture - setting model image')
    setLocalModelImage(imageDataUrl)
    onModelImageChange?.(imageDataUrl)
    setIsCameraOpen(false)
  }

  const handleModelImageUpload = (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      const reader = new FileReader()
      reader.onloadend = () => {
        const result = reader.result as string
        console.log('File upload - setting model image')
        setLocalModelImage(result)
        onModelImageChange?.(result)
      }
      reader.readAsDataURL(file)
    }
  }

  const handleRemoveModelImage = () => {
    setLocalModelImage(null)
    onModelImageChange?.(null)
  }

  return (
    <>
      <div className="space-y-6">
        {/* Product Image Section */}
        <section>
          <div className="flex items-center justify-between mb-3">
            <div>
              <h3 className="text-[15px] font-semibold">Product Image</h3>
              <p className="text-caption text-gm-text-tertiary-light dark:text-gm-text-tertiary mt-0.5">
                Upload your product with glasses
              </p>
            </div>
            {product === 'glasses' && (
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
            )}
          </div>

          {selectedTemplate ? (
            <div className="relative w-32 h-32 rounded-xl overflow-hidden bg-white dark:bg-white/10 group">
              <img
                src={selectedTemplate.product_img}
                alt="Product"
                className="w-full h-full object-contain p-2"
              />
              <button
                onClick={() => {
                  setSelectedTemplate(null)
                  onAnimationDescriptionChange('')
                  onRemoveImage()
                }}
                className="absolute top-2 right-2 w-7 h-7 bg-black/50 hover:bg-black/70 backdrop-blur-sm rounded-full flex items-center justify-center transition-all opacity-0 group-hover:opacity-100"
              >
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none">
                  <path d="M18 6L6 18M6 6L18 18" stroke="white" strokeWidth="2" strokeLinecap="round"/>
                </svg>
              </button>
            </div>
          ) : uploadedImage ? (
            <div className="relative w-32 h-32 rounded-xl overflow-hidden bg-white dark:bg-white/10 group">
              <img
                src={uploadedImage}
                alt="Product image"
                className="w-full h-full object-contain p-2"
              />
              <button
                onClick={onRemoveImage}
                className="absolute top-2 right-2 w-7 h-7 bg-black/50 hover:bg-black/70 backdrop-blur-sm rounded-full flex items-center justify-center transition-all opacity-0 group-hover:opacity-100"
              >
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none">
                  <path d="M18 6L6 18M6 6L18 18" stroke="white" strokeWidth="2" strokeLinecap="round"/>
                </svg>
              </button>
            </div>
          ) : (
            <label className="upload-zone-compact cursor-pointer">
              <input type="file" accept="image/*" className="hidden" onChange={handleFileChange} />
              <div className="w-10 h-10 rounded-lg bg-gm-accent/10 dark:bg-gm-accent/20 flex items-center justify-center flex-shrink-0">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" className="text-gm-accent">
                  <path d="M12 15V3M12 3L8 7M12 3L16 7" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
                  <path d="M2 17v2a2 2 0 002 2h16a2 2 0 002-2v-2" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
                </svg>
              </div>
              <div>
                <p className="text-button font-medium">Drop file or click to upload</p>
                <p className="text-[11px] text-gm-text-tertiary-light dark:text-gm-text-tertiary mt-0.5">PNG, JPG up to 10MB</p>
              </div>
            </label>
          )}
        </section>

        {/* Divider */}
        <div className="divider" />

        {/* Model Image Section */}
        <section>
          <div className="flex items-center justify-between mb-3">
            <div>
              <h3 className="text-[15px] font-semibold">
                Model Image <span className="text-caption text-gm-text-tertiary-light dark:text-gm-text-tertiary font-normal">(Optional)</span>
              </h3>
              <p className="text-caption text-gm-text-tertiary-light dark:text-gm-text-tertiary mt-0.5">
                {product === 'glasses' ? 'Front face photo' : 'Full body shot'}
              </p>
            </div>
          </div>

          {localModelImage ? (
            <div className="relative w-32 h-32 rounded-xl overflow-hidden bg-black/[0.02] dark:bg-white/[0.02] group">
              <img
                src={localModelImage}
                alt="Model photo"
                className="w-full h-full object-cover"
              />
              <div className="absolute inset-0 bg-gradient-to-t from-black/50 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-200" />
              <button
                onClick={handleRemoveModelImage}
                className="absolute top-2 right-2 w-7 h-7 bg-black/50 hover:bg-black/70 backdrop-blur-sm rounded-full flex items-center justify-center transition-all opacity-0 group-hover:opacity-100"
              >
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none">
                  <path d="M18 6L6 18M6 6L18 18" stroke="white" strokeWidth="2" strokeLinecap="round"/>
                </svg>
              </button>
            </div>
          ) : product === 'glasses' ? (
            <div className="flex gap-3">
              {/* Upload option */}
              <label className="upload-zone-compact cursor-pointer flex-1">
                <input
                  type="file"
                  accept="image/*"
                  className="hidden"
                  onChange={handleModelImageUpload}
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
                className="upload-zone-compact cursor-pointer flex-1"
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
          ) : (
            <label className="upload-zone-compact cursor-pointer">
              <input
                type="file"
                accept="image/*"
                className="hidden"
                onChange={handleModelImageUpload}
              />
              <div className="w-10 h-10 rounded-lg bg-gm-accent/10 dark:bg-gm-accent/20 flex items-center justify-center flex-shrink-0">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" className="text-gm-accent">
                  <path d="M12 15V3M12 3L8 7M12 3L16 7" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
                  <path d="M2 17v2a2 2 0 002 2h16a2 2 0 002-2v-2" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
                </svg>
              </div>
              <div>
                <p className="text-button font-medium">Drop file or click to upload</p>
                <p className="text-[11px] text-gm-text-tertiary-light dark:text-gm-text-tertiary mt-0.5">PNG, JPG up to 10MB</p>
              </div>
            </label>
          )}
        </section>

        {/* Divider */}
        <div className="divider" />

        {/* Scene Description */}
        <section>
          <div className="mb-3">
            <h3 className="text-[15px] font-semibold">Scene Description</h3>
            <p className="text-caption text-gm-text-tertiary-light dark:text-gm-text-tertiary mt-0.5">
              Describe environment, mood, and camera movements
            </p>
          </div>
          <PromptTextarea
            value={animationDescription}
            onChange={onAnimationDescriptionChange}
            placeholder="Professional video, model moves naturally, confident expression, modern environment..."
            className="h-28"
            showGeminiButton={!!onEnhancePrompt && !isGenerating}
            onGeminiClick={onEnhancePrompt}
            isEnhancing={isEnhancingPrompt}
          />
        </section>

        {/* Generate Button */}
        <button
          onClick={onGenerate}
          disabled={isGenerating || !uploadedImage}
          className="btn-primary w-full py-3.5 rounded-xl text-button font-medium flex items-center justify-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {isGenerating ? (
            <>
              <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none"/>
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"/>
              </svg>
              Generating...
            </>
          ) : (
            <>
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none">
                <path d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14v-4z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                <rect x="3" y="6" width="12" height="12" rx="2" stroke="currentColor" strokeWidth="2"/>
              </svg>
              Generate Video
            </>
          )}
        </button>
      </div>

      {/* Gallery Modal - Using Portal to escape glass-panel container */}
      {showGallery && createPortal(
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
              <div className="grid grid-cols-3 lg:grid-cols-4 gap-4">
                {uniqueGlasses.map((template, index) => {
                  const displayImage = template.product_img

                  return (
                    <div
                      key={index}
                      onClick={() => handleSelectTemplate(template)}
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
        </div>,
        document.body
      )}

      {/* Camera Modal */}
      <CameraCapture
        isOpen={isCameraOpen}
        onCapture={handleCameraCapture}
        onClose={() => setIsCameraOpen(false)}
      />
    </>
  )
}

export default CustomSceneMode
