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

import { ChangeEvent } from 'react'
import ProductSelector from '../../components/ProductSelector'
import type { Product } from '../../config/featureConstraints'

interface Product360FormProps {
  selectedProductImages: string[]
  uploadedImage?: string | null
  onFileUpload: (files: File[]) => void
  onRemoveFile: (index?: number) => void
  onSelectFromGallery: () => void
  onGenerate: () => void
  isLoading?: boolean
  currentProduct?: Product
  availableProducts?: Product[]
  onProductChange?: (product: Product) => void
  showPanel?: boolean
  showProductSelector?: boolean
}

function Product360Form({
  selectedProductImages,
  uploadedImage,
  onFileUpload,
  onRemoveFile,
  onSelectFromGallery,
  onGenerate,
  isLoading = false,
  currentProduct,
  availableProducts,
  onProductChange,
  showPanel = true,
  showProductSelector = true
}: Product360FormProps) {
  const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files
    if (files && files.length > 0) {
      onFileUpload(Array.from(files))
    }
  }

  const content = (
    <>
      {/* Product Selector */}
      {showProductSelector && currentProduct && availableProducts && onProductChange && (
        <div className="mb-5 pb-5 border-b border-black/[0.06] dark:border-white/[0.08]">
          <ProductSelector
            currentProduct={currentProduct}
            availableProducts={availableProducts}
            onProductChange={onProductChange}
          />
        </div>
      )}

      <div className="space-y-6">
        {/* Upload Product Images */}
        <section>
          <div className="flex items-center justify-between mb-3">
            <div>
              <h3 className="text-[15px] font-semibold">Product Images</h3>
              <p className="text-caption text-gm-text-tertiary-light dark:text-gm-text-tertiary mt-0.5">
                Upload multiple views for 360° spin
              </p>
            </div>
            <button
              onClick={onSelectFromGallery}
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

          {selectedProductImages.length > 0 ? (
            <div className="relative rounded-xl overflow-hidden bg-black/5 dark:bg-black/20">
              <div className="grid grid-cols-4 gap-1 p-1">
                {selectedProductImages.map((imgPath, index) => (
                  <div
                    key={index}
                    className="relative aspect-square bg-white/5 dark:bg-white/10 group/img"
                  >
                    <img
                      src={imgPath}
                      alt={`Product view ${index + 1}`}
                      className="w-full h-full object-contain"
                    />
                    <div className="absolute top-1 left-1 w-5 h-5 rounded-full bg-black/50 backdrop-blur-sm flex items-center justify-center text-white text-[10px] font-medium">
                      {index + 1}
                    </div>
                    <button
                      onClick={() => onRemoveFile(index)}
                      className="absolute top-1 right-1 w-5 h-5 bg-black/50 hover:bg-red-600 backdrop-blur-sm rounded-full flex items-center justify-center transition-all opacity-0 group-hover/img:opacity-100"
                    >
                      <svg width="10" height="10" viewBox="0 0 24 24" fill="none">
                        <path d="M18 6L6 18M6 6L18 18" stroke="white" strokeWidth="2.5" strokeLinecap="round"/>
                      </svg>
                    </button>
                  </div>
                ))}
              </div>
            </div>
          ) : uploadedImage ? (
            <div className="relative rounded-xl overflow-hidden bg-black/[0.02] dark:bg-white/[0.02] group">
              <img
                src={uploadedImage}
                alt="Product image"
                className="w-full aspect-[4/3] object-cover"
              />
              <div className="absolute inset-0 bg-gradient-to-t from-black/50 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-200" />
              <button
                onClick={() => onRemoveFile()}
                className="absolute top-2 right-2 w-7 h-7 bg-black/50 hover:bg-black/70 backdrop-blur-sm rounded-full flex items-center justify-center transition-all opacity-0 group-hover:opacity-100"
              >
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none">
                  <path d="M18 6L6 18M6 6L18 18" stroke="white" strokeWidth="2" strokeLinecap="round"/>
                </svg>
              </button>
            </div>
          ) : (
            <label className="upload-zone-compact cursor-pointer">
              <input type="file" accept="image/*" multiple className="hidden" onChange={handleFileChange} />
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
          onClick={onGenerate}
          disabled={isLoading}
          className="btn-primary w-full py-3.5 rounded-xl text-button font-medium flex items-center justify-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {isLoading ? (
            <>
              <svg className="animate-spin" width="18" height="18" viewBox="0 0 24 24" fill="none">
                <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="2" opacity="0.25"/>
                <path d="M12 2a10 10 0 0 1 10 10" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
              </svg>
              Generating...
            </>
          ) : (
            <>
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none">
                <path d="M23 4v6h-6" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                <path d="M1 20v-6h6" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                <path d="M3.51 9a9 9 0 0114.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0020.49 15" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
              </svg>
              Generate 360° Spin
            </>
          )}
        </button>
      </div>
    </>
  )

  if (showPanel) {
    return (
      <div className="glass-panel rounded-xl p-5 lg:p-6 h-fit animate-fade-in">
        {content}
      </div>
    )
  }

  return content
}

export default Product360Form
