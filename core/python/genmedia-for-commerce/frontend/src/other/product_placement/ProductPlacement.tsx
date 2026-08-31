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
import PromptTextarea from '../../components/PromptTextarea'
import BackgroundVideo from '../../components/BackgroundVideo'

interface ProductPlacementProps {
  showVideo?: boolean
  embedMode?: boolean
}

function ProductPlacement({ showVideo = true, embedMode = false }: ProductPlacementProps) {
  const [uploadedImage, setUploadedImage] = useState<string | null>(null)
  const [sceneDescription, setSceneDescription] = useState('')

  const handleRemoveFile = () => {
    setUploadedImage(null)
  }

  const handleGenerate = () => {
    console.log('Generating product placement...')
  }

  // Settings content
  const settingsContent = (
    <>
        {/* Coming Soon Badge (only in standalone mode) */}
        {!embedMode && (
          <div className="absolute -top-3 -right-3 z-10">
            <div className="bg-gradient-to-r from-gm-accent to-blue-500 text-white px-4 py-1.5 rounded-full text-caption font-bold tracking-wide shadow-lg">
              COMING SOON
            </div>
          </div>
        )}
        {/* Upload Product Image */}
        <div className="mb-8">
          <h3 className="text-h3 font-bold mb-2">Upload Product Image</h3>
          <p className="text-body text-gm-text-secondary-light dark:text-gm-text-secondary mb-4">
            Upload the product you want to place into a custom scene.
          </p>

          {uploadedImage ? (
            <div className="relative rounded-lg overflow-hidden bg-black/5 dark:bg-white/5 opacity-50">
              <img
                src={uploadedImage}
                alt="Product image"
                className="w-full aspect-square object-contain"
              />
              <button
                onClick={handleRemoveFile}
                className="absolute top-2 right-2 w-6 h-6 bg-black/70 hover:bg-black/90 rounded-full flex items-center justify-center transition-all"
                disabled
              >
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none">
                  <path d="M18 6L6 18M6 6L18 18" stroke="white" strokeWidth="2" strokeLinecap="round"/>
                </svg>
              </button>
            </div>
          ) : (
            <div className="upload-zone h-32 flex flex-col items-center justify-center cursor-not-allowed transition-all duration-300 opacity-50">
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" className="mb-2">
                <path d="M12 15V3M12 3L8 7M12 3L16 7" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
                <path d="M2 17L2 19C2 20.1046 2.89543 21 4 21L20 21C21.1046 21 22 20.1046 22 19V17" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
              </svg>
              <p className="text-[10px] font-medium mb-1">DROP YOUR FILE HERE</p>
            </div>
          )}
        </div>

        {/* Scene Description */}
        <div className="mb-8 opacity-50">
          <h3 className="text-h3 font-bold mb-2">Scene Description</h3>
          <p className="text-body text-gm-text-secondary-light dark:text-gm-text-secondary mb-4">
            Describe the scene where you want to place your product.
          </p>
          <div className="pointer-events-none">
            <PromptTextarea
              value={sceneDescription}
              onChange={setSceneDescription}
              placeholder="Place the product on a wooden desk in a modern home office with natural lighting, or on a marble countertop in a luxury boutique, or outdoors on a park bench..."
            />
          </div>
        </div>

        {/* Generate Button */}
        <div className="flex justify-end">
          <button
            onClick={handleGenerate}
            disabled
            className="btn-primary px-8 py-3 rounded-pill text-button font-medium opacity-50 cursor-not-allowed"
          >
            Place Product
          </button>
        </div>
      </>
  )

  // In embed mode, just return the settings content
  if (embedMode) {
    return settingsContent
  }

  // In standalone mode, return full layout with preview panel
  return (
    <div className="grid grid-cols-[480px_1fr] gap-6">
      {/* Left Panel - Settings */}
      <div className="glass-panel rounded-lg p-6 h-fit relative">
        {settingsContent}
      </div>

      {/* Right Panel - Preview */}
      <div className={`${showVideo ? 'glass-panel' : 'border border-black/[0.08] dark:border-white/10'} rounded-lg flex items-center justify-center min-h-[600px] overflow-hidden relative`}>
        {showVideo && <BackgroundVideo />}

        <div className="absolute inset-0 bg-gradient-to-t from-gm-bg-light/50 via-gm-bg-light/20 to-transparent dark:from-gm-bg/50 dark:via-gm-bg/20"></div>

        <div className="text-center px-8 relative z-10">
          <div className="mb-6 inline-block">
            <div className="bg-gradient-to-r from-gm-accent to-blue-500 text-white px-6 py-2 rounded-full text-button font-bold tracking-wide shadow-xl">
              COMING SOON
            </div>
          </div>
          <h2 className="text-[48px] font-bold mb-4 leading-tight">
            Product <span className="text-gm-accent" style={{ textShadow: '0 0 30px rgba(138, 180, 248, 0.5), 0 0 60px rgba(138, 180, 248, 0.3)' }}>Placement</span>
          </h2>
          <p className="text-lg text-gm-text-secondary-light dark:text-gm-text-secondary max-w-md mx-auto">
            Place your products into custom scenes with AI-powered contextual integration. This feature will be available soon!
          </p>
        </div>
      </div>
    </div>
  )
}

export default ProductPlacement
