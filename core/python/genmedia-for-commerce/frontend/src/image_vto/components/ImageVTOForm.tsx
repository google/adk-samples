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

import { useState, useRef, useCallback, ChangeEvent, useEffect } from 'react'
import type { Product, Capability } from '../../config/featureConstraints'
import ProductSelector from '../../components/ProductSelector'
import ImageLightbox from '../../components/ImageLightbox'
import { MODEL_PRESETS } from '../../shared/modelGallery'
import { WOMEN_GARMENTS, MEN_GARMENTS } from '../../shared/garmentGallery'

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

export interface VTOFormData {
  garmentImages: string[]
  garmentUris: string[]
  faceImage: string | null
  fullBodyImage: string | null
  scenario: string
}

interface ImageVTOFormProps {
  product: Product
  capability: Capability
  onSubmit: (data: VTOFormData) => void
  isGenerating?: boolean
  uploadedImage?: string | null
  onRemoveImage?: () => void
  currentProduct?: Product
  availableProducts?: Product[]
  onProductChange?: (product: Product) => void
  prefilledModelImage?: string
  prefilledGarmentImages?: string[]
}

function ImageVTOForm({
  onSubmit,
  isGenerating = false,
  uploadedImage,
  onRemoveImage,
  currentProduct,
  availableProducts,
  onProductChange,
  prefilledModelImage,
  prefilledGarmentImages
}: ImageVTOFormProps) {
  const [garmentImages, setGarmentImages] = useState<string[]>([])
  const [garmentGcsMap, setGarmentGcsMap] = useState<Record<string, string>>({})
  const [fullBodyImage, setFullBodyImage] = useState<string | null>(null)
  const [showGarmentGallery, setShowGarmentGallery] = useState(false)
  const [showModelGallery, setShowModelGallery] = useState(false)
  const [selectedGarments, setSelectedGarments] = useState<string[]>([])
  const [lightboxImage, setLightboxImage] = useState<string | null>(null)

  // Catalog search state
  const [searchQuery, setSearchQuery] = useState('')
  const [searchResults, setSearchResults] = useState<CatalogResult[]>([])
  const [isSearching, setIsSearching] = useState(false)
  const debounceTimer = useRef<ReturnType<typeof setTimeout> | null>(null)

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

  const womenModels = MODEL_PRESETS.filter(m => m.gender === 'woman')
  const menModels = MODEL_PRESETS.filter(m => m.gender === 'man')

  // Pre-fill full body image from uploaded image prop
  useEffect(() => {
    if (uploadedImage && !fullBodyImage) {
      setFullBodyImage(uploadedImage)
    }
  }, [uploadedImage, fullBodyImage])

  // Pre-fill model image from GenMedia TV
  useEffect(() => {
    if (prefilledModelImage && !fullBodyImage) {
      setFullBodyImage(prefilledModelImage)
    }
  }, [prefilledModelImage])

  // Pre-fill garment images from GenMedia TV
  useEffect(() => {
    if (prefilledGarmentImages && prefilledGarmentImages.length > 0 && garmentImages.length === 0) {
      setGarmentImages(prefilledGarmentImages)
      setSelectedGarments(prefilledGarmentImages)
    }
  }, [prefilledGarmentImages])

  const handleMultipleImageUpload = (e: ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files
    if (!files || files.length === 0) return

    const fileArray = Array.from(files)
    fileArray.forEach(file => {
      const reader = new FileReader()
      reader.onloadend = () => {
        setGarmentImages(prev => [...prev, reader.result as string])
      }
      reader.readAsDataURL(file)
    })
  }

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
    if (isPrefilledImage && onRemoveImage) {
      onRemoveImage()
    }
  }

  const handleToggleGarmentSelection = (garmentPath: string) => {
    setSelectedGarments(prev =>
      prev.includes(garmentPath)
        ? prev.filter(p => p !== garmentPath)
        : [...prev, garmentPath]
    )
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
      setGarmentImages(prev => [...prev, ...selectedGarments])
      setSelectedGarments([])
      setShowGarmentGallery(false)
    }
  }

  const handleCloseGarmentGallery = () => {
    setSelectedGarments([])
    setShowGarmentGallery(false)
    setSearchQuery('')
    setSearchResults([])
    if (debounceTimer.current) clearTimeout(debounceTimer.current)
  }

  const handleSelectModelFromGallery = (modelPath: string) => {
    setFullBodyImage(modelPath)
    setShowModelGallery(false)
  }

  const convertImageToDataUrl = async (imagePath: string): Promise<string> => {
    // If it's already a data URL, return as-is
    if (imagePath.startsWith('data:')) {
      return imagePath
    }

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

  const handleGenerate = async () => {
    console.log('Generate VTO clicked', {
      garmentImages,
      fullBodyImage,
      isFormValid
    })

    try {
      // Split garments: GCS catalog items pass as URIs, others convert to data URLs
      const uploadGarments = garmentImages.filter(img => !garmentGcsMap[img])
      const gcsUris = garmentImages.filter(img => garmentGcsMap[img]).map(img => garmentGcsMap[img])

      const garmentDataUrls = await Promise.all(
        uploadGarments.map(img => convertImageToDataUrl(img))
      )
      const bodyDataUrl = fullBodyImage ? await convertImageToDataUrl(fullBodyImage) : null

      onSubmit({
        garmentImages: garmentDataUrls,
        garmentUris: gcsUris,
        faceImage: null,
        fullBodyImage: bodyDataUrl,
        scenario: 'a plain white studio background'
      })
    } catch (error) {
      console.error('Failed to convert images:', error)
    }
  }

  const isFormValid = garmentImages.length > 0 && fullBodyImage
  const buttonDisabled = isGenerating || !isFormValid

  const renderSingleImageUpload = (
    image: string | null,
    setter: (img: string | null) => void,
    label: string,
    description: string,
    isPrefilledImage?: boolean
  ) => {
    if (image) {
      return (
        <div className="relative w-32 h-40 rounded-xl overflow-hidden bg-black/[0.02] dark:bg-white/[0.02] group cursor-pointer"
          onClick={() => setLightboxImage(image)}
        >
          <img
            src={image}
            alt={label}
            className="w-full h-full object-cover"
          />
          <div className="absolute inset-0 bg-gradient-to-t from-black/50 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-200 pointer-events-none" />
          <button
            onClick={(e) => { e.stopPropagation(); handleRemoveImage(setter, isPrefilledImage)() }}
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
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" className="opacity-70">
                <rect x="3" y="3" width="7" height="7" rx="1" stroke="currentColor" strokeWidth="2"/>
                <rect x="14" y="3" width="7" height="7" rx="1" stroke="currentColor" strokeWidth="2"/>
                <rect x="3" y="14" width="7" height="7" rx="1" stroke="currentColor" strokeWidth="2"/>
                <rect x="14" y="14" width="7" height="7" rx="1" stroke="currentColor" strokeWidth="2"/>
              </svg>
              From gallery
            </button>
          </div>

          {renderSingleImageUpload(fullBodyImage, setFullBodyImage, 'Model photo', 'PNG, JPG up to 10MB', fullBodyImage === uploadedImage)}
        </section>

        {/* Divider */}
        <div className="divider" />

        {/* Section 2: Garment Images (Multiple) */}
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
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" className="opacity-70">
                <rect x="3" y="3" width="7" height="7" rx="1" stroke="currentColor" strokeWidth="2"/>
                <rect x="14" y="3" width="7" height="7" rx="1" stroke="currentColor" strokeWidth="2"/>
                <rect x="3" y="14" width="7" height="7" rx="1" stroke="currentColor" strokeWidth="2"/>
                <rect x="14" y="14" width="7" height="7" rx="1" stroke="currentColor" strokeWidth="2"/>
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
            <label className={`upload-zone-compact cursor-pointer ${isGenerating ? 'opacity-50 pointer-events-none' : ''}`}>
              <input
                type="file"
                accept="image/*"
                multiple
                className="hidden"
                onChange={handleMultipleImageUpload}
                disabled={isGenerating}
              />
              <div className="w-10 h-10 rounded-lg bg-gm-accent/10 dark:bg-gm-accent/20 flex items-center justify-center flex-shrink-0">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" className="text-gm-accent">
                  <path d="M12 15V3M12 3L8 7M12 3L16 7" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
                  <path d="M2 17v2a2 2 0 002 2h16a2 2 0 002-2v-2" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
                </svg>
              </div>
              <div>
                <p className="text-button font-medium">Drop files or click to upload</p>
                <p className="text-[11px] text-gm-text-tertiary-light dark:text-gm-text-tertiary mt-0.5">PNG, JPG up to 10MB each</p>
              </div>
            </label>
          )}
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
            Please upload all required images to continue
          </p>
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
            onClick={e => e.stopPropagation()}
          >
            {/* Header */}
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
                  <path d="M18 6L6 18M6 6L18 18" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
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

            {/* Grid */}
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

              {/* Default Garments - only shown when search bar is empty */}
              {!searchQuery.trim() && !isSearching && (
                <>
              {/* Women Garments Section */}
              <div className="mb-6">
                <h3 className="text-sm font-semibold text-gm-text-secondary-light dark:text-gm-text-secondary mb-3 px-2">Women</h3>
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

              {/* Men Garments Section */}
              <div>
                <h3 className="text-sm font-semibold text-gm-text-secondary-light dark:text-gm-text-secondary mb-3 px-2">Men</h3>
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
                </>
              )}
            </div>

            {/* Footer Actions */}
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
            onClick={e => e.stopPropagation()}
          >
            {/* Header */}
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

            {/* Grid */}
            <div className="overflow-y-auto flex-1 -mx-2 px-2">
              {/* Women Models Section */}
              <div className="mb-6">
                <h3 className="text-sm font-semibold text-gm-text-secondary-light dark:text-gm-text-secondary mb-3 px-2">Women</h3>
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

              {/* Men Models Section */}
              <div>
                <h3 className="text-sm font-semibold text-gm-text-secondary-light dark:text-gm-text-secondary mb-3 px-2">Men</h3>
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

      {/* Image Lightbox */}
      {lightboxImage && (
        <ImageLightbox src={lightboxImage} alt="Preview" onClose={() => setLightboxImage(null)} />
      )}
    </>
  )
}

export default ImageVTOForm
