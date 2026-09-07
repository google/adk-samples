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

import AnimateFrameMode from './AnimateFrameMode'
import CustomSceneMode from './CustomSceneMode'
import ProductSelector from '../../components/ProductSelector'
import type { Product, Capability } from '../../config/featureConstraints'

interface VideoVTOFormProps {
  product: Product
  capability: Capability
  subMode: 'animate-frame' | 'custom-scene'
  onSubModeChange: (mode: 'animate-frame' | 'custom-scene') => void
  uploadedImage?: string | null
  animationDescription: string
  onAnimationDescriptionChange: (value: string) => void
  onFileUpload: (file: File) => void
  onRemoveFile: () => void
  onGenerate: () => void
  isGenerating: boolean
  generatedVideo: string | null
  onRegenerate: () => void
  onEnhancePrompt?: () => Promise<void>
  isEnhancingPrompt?: boolean
  modelImage?: string | null
  onModelImageChange?: (image: string | null) => void
  currentProduct?: Product
  availableProducts?: Product[]
  onProductChange?: (product: Product) => void
}

function VideoVTOForm({
  product,
  capability,
  subMode,
  onSubModeChange,
  uploadedImage,
  animationDescription,
  onAnimationDescriptionChange,
  onFileUpload,
  onRemoveFile,
  onGenerate,
  isGenerating,
  generatedVideo,
  onRegenerate,
  onEnhancePrompt,
  isEnhancingPrompt,
  modelImage,
  onModelImageChange,
  currentProduct,
  availableProducts,
  onProductChange
}: VideoVTOFormProps) {
  return (
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
          onClick={() => onSubModeChange('animate-frame')}
          className={`flex-1 px-4 py-2.5 rounded-lg text-button font-medium transition-all duration-200 flex items-center justify-center gap-2 ${
            subMode === 'animate-frame'
              ? 'bg-white dark:bg-white/10 shadow-sm text-gm-text-primary-light dark:text-gm-text-primary'
              : 'text-gm-text-tertiary-light dark:text-gm-text-tertiary hover:text-gm-text-primary-light dark:hover:text-gm-text-primary'
          }`}
        >
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" className="opacity-70">
            <rect x="3" y="3" width="18" height="18" rx="2" stroke="currentColor" strokeWidth="2"/>
            <circle cx="12" cy="12" r="3" stroke="currentColor" strokeWidth="2"/>
          </svg>
          Animate Frame
        </button>
        <button
          onClick={() => onSubModeChange('custom-scene')}
          className={`flex-1 px-4 py-2.5 rounded-lg text-button font-medium transition-all duration-200 flex items-center justify-center gap-2 ${
            subMode === 'custom-scene'
              ? 'bg-white dark:bg-white/10 shadow-sm text-gm-text-primary-light dark:text-gm-text-primary'
              : 'text-gm-text-tertiary-light dark:text-gm-text-tertiary hover:text-gm-text-primary-light dark:hover:text-gm-text-primary'
          }`}
        >
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" className="opacity-70">
            <path d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14v-4z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
            <rect x="3" y="6" width="12" height="12" rx="2" stroke="currentColor" strokeWidth="2"/>
          </svg>
          Custom Scene
        </button>
      </div>

      {subMode === 'custom-scene' ? (
        <CustomSceneMode
          product={product}
          capability={capability}
          uploadedImage={uploadedImage}
          onFileUpload={onFileUpload}
          onRemoveImage={onRemoveFile}
          animationDescription={animationDescription}
          onAnimationDescriptionChange={onAnimationDescriptionChange}
          onGenerate={onGenerate}
          modelImage={modelImage}
          onModelImageChange={onModelImageChange}
          onEnhancePrompt={onEnhancePrompt}
          isEnhancingPrompt={isEnhancingPrompt}
          isGenerating={isGenerating}
        />
      ) : (
        <AnimateFrameMode
          uploadedImage={uploadedImage}
          animationDescription={animationDescription}
          onAnimationDescriptionChange={onAnimationDescriptionChange}
          onFileUpload={onFileUpload}
          onRemoveFile={onRemoveFile}
          onGenerate={onGenerate}
          isGenerating={isGenerating}
          generatedVideo={generatedVideo}
          onRegenerate={onRegenerate}
          onEnhancePrompt={onEnhancePrompt}
          isEnhancingPrompt={isEnhancingPrompt}
        />
      )}
    </div>
  )
}

export default VideoVTOForm
