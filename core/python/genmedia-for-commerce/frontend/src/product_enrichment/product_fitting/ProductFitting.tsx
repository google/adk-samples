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
import ProductFittingForm from './ProductFittingForm'
import ProductFittingPreview from './ProductFittingPreview'
import { generateFitting, FittingPipelineResult, generateVideo, VideoResult } from './productFittingApi'
import type { ModelPreset } from '../../shared/modelGallery'

interface ProductFittingProps {
  showVideo?: boolean
}

async function convertImageToDataUrl(imagePath: string): Promise<string> {
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

function ProductFitting({ showVideo = true }: ProductFittingProps) {
  const [garmentImages, setGarmentImages] = useState<string[]>([])
  const [selectedModel, setSelectedModel] = useState<ModelPreset | null>(null)
  const [customModelImage, setCustomModelImage] = useState<string | null>(null)

  const [isLoading, setIsLoading] = useState(false)
  const [result, setResult] = useState<FittingPipelineResult | null>(null)
  const [error, setError] = useState<string | null>(null)

  const [isVideoLoading, setIsVideoLoading] = useState(false)
  const [videoResult, setVideoResult] = useState<VideoResult | null>(null)
  const [videoError, setVideoError] = useState<string | null>(null)

  const handleFileUpload = (files: File[]) => {
    const newImages: string[] = []
    let loadedCount = 0

    files.forEach((file) => {
      const reader = new FileReader()
      reader.onloadend = () => {
        newImages.push(reader.result as string)
        loadedCount++
        if (loadedCount === files.length) {
          setGarmentImages(prev => [...prev, ...newImages])
          setResult(null)
          setError(null)
        }
      }
      reader.readAsDataURL(file)
    })
  }

  const handleAddGarments = (paths: string[]) => {
    setGarmentImages(prev => [...prev, ...paths])
    setResult(null)
    setError(null)
  }

  const handleRemoveImages = () => {
    setGarmentImages([])
    setResult(null)
    setError(null)
  }

  const handleModelSelect = (preset: ModelPreset) => {
    setSelectedModel(preset)
    setCustomModelImage(null)
    setResult(null)
    setError(null)
  }

  const handleModelFileUpload = (file: File) => {
    const reader = new FileReader()
    reader.onloadend = () => {
      setCustomModelImage(reader.result as string)
      setSelectedModel(null)
      setResult(null)
      setError(null)
    }
    reader.readAsDataURL(file)
  }

  const handleModelRemove = () => {
    setSelectedModel(null)
    setCustomModelImage(null)
    setResult(null)
    setError(null)
  }

  // Derive the model image to display (gallery preset thumbnail or custom upload)
  const modelImage = selectedModel ? selectedModel.thumbnail : customModelImage

  const handleGenerate = async () => {
    if (garmentImages.length === 0) {
      setError('Please upload garment images first')
      return
    }
    if (!selectedModel && !customModelImage) {
      setError('Please select a model first')
      return
    }

    setIsLoading(true)
    setError(null)
    setResult(null)
    setIsVideoLoading(false)
    setVideoResult(null)
    setVideoError(null)

    try {
      const garmentDataUrls = await Promise.all(
        garmentImages.map(img => convertImageToDataUrl(img))
      )

      // Extract ethnicity+gender from preset id (e.g. "african_woman" -> "african", "woman")
      const ethnicity = selectedModel ? selectedModel.id.split('_')[0] : 'european'
      const gender = selectedModel ? selectedModel.gender : 'woman'

      const pipelineResult = await generateFitting({
        garmentImages: garmentDataUrls,
        scenario: 'product_fitting',
        ethnicity,
        gender,
        ...(customModelImage ? { modelPhotos: [customModelImage] } : {}),
      })

      setResult(pipelineResult)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to generate fitting')
    } finally {
      setIsLoading(false)
    }
  }

  const handleGenerateVideo = async () => {
    if (!result || (!result.front && !result.back)) return
    
    setIsVideoLoading(true)
    setVideoError(null)
    
    try {
      const frontImage = result.front?.imageBase64
      const backImage = result.back?.imageBase64
      
      if (!frontImage) {
         setVideoError('No front image available for animation')
         return
      }
      
      const videoRes = await generateVideo({
        frontImage,
        backImage,
        framing: result.framing ?? 'full_body',
      })
      
      setVideoResult(videoRes)
    } catch (err) {
      setVideoError(err instanceof Error ? err.message : 'Failed to generate video')
    } finally {
      setIsVideoLoading(false)
    }
  }

  return (
    <div className="grid grid-cols-[480px_1fr] gap-6">
      <ProductFittingForm
        garmentImages={garmentImages}
        selectedModelId={selectedModel?.id ?? null}
        modelImage={modelImage}
        onModelSelect={handleModelSelect}
        onModelFileUpload={handleModelFileUpload}
        onModelRemove={handleModelRemove}
        onFileUpload={handleFileUpload}
        onAddGarments={handleAddGarments}
        onRemoveImages={handleRemoveImages}
        onGenerate={handleGenerate}
        isLoading={isLoading}
      />
      <ProductFittingPreview
        key={result?.front?.imageUrl || 'empty'}
        showVideo={showVideo}
        isLoading={isLoading}
        error={error}
        result={result}
        onAnimate={handleGenerateVideo}
        isVideoLoading={isVideoLoading}
        videoResult={videoResult}
        videoError={videoError}
      />
    </div>
  )
}

export default ProductFitting
