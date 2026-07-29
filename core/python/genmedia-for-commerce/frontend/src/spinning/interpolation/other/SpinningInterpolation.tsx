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

import { useState } from 'react'
import Product360Form from '../../components/Product360Form'
import Product360Preview from '../../components/Product360Preview'
import type { Product } from '../../../config/featureConstraints'

interface SpinningInterpolationProps {
  product: Product
  prefilledImages?: string[]
  prefilledPrompt?: string
}

function SpinningInterpolation({
  product,
  prefilledImages = []
}: SpinningInterpolationProps) {
  const [selectedProductImages, setSelectedProductImages] = useState<string[]>(prefilledImages)
  const [uploadedImage, setUploadedImage] = useState<string | null>(
    prefilledImages.length > 0 ? prefilledImages[0] : null
  )

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
  }

  const handleSelectFromGallery = () => {
    console.log('Opening gallery for interpolation mode...')
  }

  const handleGenerate = () => {
    console.log(`Generating ${product} 360 spin with interpolation...`)
  }

  return (
    <div className="grid grid-cols-[480px_1fr] gap-6">
      <Product360Form
        selectedProductImages={selectedProductImages}
        uploadedImage={uploadedImage}
        onFileUpload={handleFileUpload}
        onRemoveFile={handleRemoveFile}
        onSelectFromGallery={handleSelectFromGallery}
        onGenerate={handleGenerate}
      />
      <Product360Preview flow="interpolation-other" />
    </div>
  )
}

export default SpinningInterpolation
