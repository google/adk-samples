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

import { useState, useCallback } from 'react'
import { PRODUCTS, CAPABILITIES } from '../../config/featureConstraints'
import type { Product } from '../../config/featureConstraints'
import ImageVTOForm from '../components/ImageVTOForm'
import type { VTOFormData } from '../components/ImageVTOForm'
import ImageVTOPreview from '../components/ImageVTOPreview'
import { generateVTO } from '../services/imageVtoApi'
import type { VTOResult } from '../services/imageVtoApi'

interface ClothesImageVTOProps {
  uploadedImage?: string | null
  onRemoveImage?: () => void
  showVideo?: boolean
  currentProduct?: Product
  availableProducts?: Product[]
  onProductChange?: (product: Product) => void
  prefilledModelImage?: string
  prefilledGarmentImages?: string[]
}

function ClothesImageVTO({
  uploadedImage,
  onRemoveImage,
  showVideo = true,
  currentProduct,
  availableProducts,
  onProductChange,
  prefilledModelImage,
  prefilledGarmentImages
}: ClothesImageVTOProps) {
  const [results, setResults] = useState<VTOResult[]>([])
  const [isGenerating, setIsGenerating] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const handleSubmit = useCallback(async (formData: VTOFormData) => {
    // Validate required fields (face image is optional)
    if ((!formData.garmentImages.length && !formData.garmentUris?.length) || !formData.fullBodyImage) {
      setError('Please upload garment and full body images')
      return
    }

    // Reset state for new generation
    setResults([])
    setError(null)
    setIsGenerating(true)

    try {
      await generateVTO(
        {
          faceImage: formData.faceImage,
          fullBodyImage: formData.fullBodyImage,
          garments: formData.garmentImages,
          garmentUris: formData.garmentUris,
          scenario: formData.scenario || 'a plain light grey studio environment',
          numVariations: 3
        },
        // onResult callback - add each result as it streams in
        (result) => {
          setResults(prev => {
            // Replace if already exists (shouldn't happen), otherwise add
            const existing = prev.findIndex(r => r.index === result.index)
            if (existing >= 0) {
              const updated = [...prev]
              updated[existing] = result
              return updated
            }
            return [...prev, result]
          })
        },
        // onComplete callback
        (total) => {
          console.log(`VTO generation complete. ${total} variations processed.`)
          setIsGenerating(false)
        },
        // onError callback
        (errorMessage) => {
          console.error('VTO generation error:', errorMessage)
          setError(errorMessage)
          setIsGenerating(false)
        }
      )
    } catch (err) {
      const message = err instanceof Error ? err.message : 'An unexpected error occurred'
      setError(message)
      setIsGenerating(false)
    }
  }, [])

  return (
    <div className="grid grid-cols-[480px_1fr] gap-6">
      <ImageVTOForm
        product={PRODUCTS.CLOTHES}
        capability={CAPABILITIES.IMAGE_VTO}
        onSubmit={handleSubmit}
        isGenerating={isGenerating}
        uploadedImage={uploadedImage}
        onRemoveImage={onRemoveImage}
        currentProduct={currentProduct}
        availableProducts={availableProducts}
        onProductChange={onProductChange}
        prefilledModelImage={prefilledModelImage}
        prefilledGarmentImages={prefilledGarmentImages}
      />

      {error && (
        <div className="mb-4 p-3 bg-red-50 border border-red-200 rounded-lg text-red-700 text-sm">
          {error}
        </div>
      )}
      <ImageVTOPreview
        showVideo={showVideo}
        results={results}
        isLoading={isGenerating}
        numVariations={3}
      />
    </div>
  )
}

export default ClothesImageVTO
