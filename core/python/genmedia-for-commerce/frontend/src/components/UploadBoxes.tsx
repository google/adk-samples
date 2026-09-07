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

import { getUploadBoxes } from '../config/featureConstraints'
import type { Product, Capability } from '../config/featureConstraints'
import { Fragment, ChangeEvent, useState } from 'react'

interface UploadBoxesProps {
  product: Product
  capability: Capability
  prefilledImage?: string | null
  onRemoveImage?: () => void
  onFileUpload?: (file: File) => void
}

// Helper to get friendly label for upload box type
const getBoxLabel = (boxType: string): string => {
  const labels: Record<string, string> = {
    'model_image': 'Model Image',
    'face_photo': 'Face Photo',
    'product_image': 'Product Image',
    'garment_image': 'Garment Image'
  }
  return labels[boxType] || boxType.replace('_', ' ')
}

function UploadBoxes({ product, capability, prefilledImage, onRemoveImage, onFileUpload }: UploadBoxesProps) {
  const uploadBoxes = getUploadBoxes(product, capability)
  const [uploadedImages, setUploadedImages] = useState<Record<string, string>>({})

  const handleFileChange = (boxType: string) => (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      if (onFileUpload) {
        onFileUpload(file)
      }
      const reader = new FileReader()
      reader.onloadend = () => {
        setUploadedImages(prev => ({ ...prev, [boxType]: reader.result as string }))
      }
      reader.readAsDataURL(file)
    }
  }

  const handleRemoveImage = (boxType: string) => () => {
    setUploadedImages(prev => {
      const updated = { ...prev }
      delete updated[boxType]
      return updated
    })
    if (onRemoveImage) {
      onRemoveImage()
    }
  }

  // Single upload box
  if (uploadBoxes.length === 1) {
    const boxType = uploadBoxes[0]
    const currentImage = uploadedImages[boxType] || prefilledImage

    // Show image if available
    if (currentImage) {
      return (
        <div className="relative rounded-xl overflow-hidden bg-black/[0.02] dark:bg-white/[0.02] group">
          <img
            src={currentImage}
            alt="Uploaded asset"
            className="w-full aspect-[4/3] object-cover"
          />
          <div className="absolute inset-0 bg-gradient-to-t from-black/50 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-200" />
          <button
            onClick={handleRemoveImage(boxType)}
            className="absolute top-2 right-2 w-7 h-7 bg-black/50 hover:bg-black/70 backdrop-blur-sm rounded-full flex items-center justify-center transition-all opacity-0 group-hover:opacity-100"
          >
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none">
              <path d="M18 6L6 18M6 6L18 18" stroke="white" strokeWidth="2" strokeLinecap="round"/>
            </svg>
          </button>
        </div>
      )
    }

    return (
      <label className="upload-zone-compact cursor-pointer">
        <input type="file" accept="image/*" className="hidden" onChange={handleFileChange(boxType)} />
        <div className="w-10 h-10 rounded-lg bg-gm-accent/10 dark:bg-gm-accent/20 flex items-center justify-center flex-shrink-0">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" className="text-gm-accent">
            <path d="M12 15V3M12 3L8 7M12 3L16 7" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
            <path d="M2 17v2a2 2 0 002 2h16a2 2 0 002-2v-2" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
          </svg>
        </div>
        <div>
          <p className="text-button font-medium">Drop file or click to upload</p>
          <p className="text-[11px] text-gm-text-tertiary-light dark:text-gm-text-tertiary mt-0.5">{getBoxLabel(boxType)}</p>
        </div>
      </label>
    )
  }

  // Multiple upload boxes - more compact layout
  if (uploadBoxes.length > 1) {
    return (
      <div className="space-y-2">
        {uploadBoxes.map((boxType) => {
          const currentImage = uploadedImages[boxType] || (prefilledImage && (boxType === 'model_image' || boxType === 'face_photo') ? prefilledImage : null)

          return (
            <Fragment key={boxType}>
              {currentImage ? (
                <div className="relative rounded-xl overflow-hidden bg-black/[0.02] dark:bg-white/[0.02] group">
                  <img
                    src={currentImage}
                    alt="Uploaded asset"
                    className="w-full aspect-[4/3] object-cover"
                  />
                  <div className="absolute inset-0 bg-gradient-to-t from-black/50 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-200" />
                  <button
                    onClick={handleRemoveImage(boxType)}
                    className="absolute top-2 right-2 w-7 h-7 bg-black/50 hover:bg-black/70 backdrop-blur-sm rounded-full flex items-center justify-center transition-all opacity-0 group-hover:opacity-100"
                  >
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none">
                      <path d="M18 6L6 18M6 6L18 18" stroke="white" strokeWidth="2" strokeLinecap="round"/>
                    </svg>
                  </button>
                  <div className="absolute bottom-2 left-2 px-2 py-1 bg-black/50 backdrop-blur-sm rounded text-white text-[11px] font-medium">
                    {getBoxLabel(boxType)}
                  </div>
                </div>
              ) : (
                <label className="upload-zone-compact cursor-pointer">
                  <input type="file" accept="image/*" className="hidden" onChange={handleFileChange(boxType)} />
                  <div className="w-10 h-10 rounded-lg bg-gm-accent/10 dark:bg-gm-accent/20 flex items-center justify-center flex-shrink-0">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" className="text-gm-accent">
                      <path d="M12 15V3M12 3L8 7M12 3L16 7" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
                      <path d="M2 17v2a2 2 0 002 2h16a2 2 0 002-2v-2" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
                    </svg>
                  </div>
                  <div>
                    <p className="text-button font-medium">Drop file or click to upload</p>
                    <p className="text-[11px] text-gm-text-tertiary-light dark:text-gm-text-tertiary mt-0.5">{getBoxLabel(boxType)}</p>
                  </div>
                </label>
              )}
            </Fragment>
          )
        })}
      </div>
    )
  }

  return null
}

export default UploadBoxes
