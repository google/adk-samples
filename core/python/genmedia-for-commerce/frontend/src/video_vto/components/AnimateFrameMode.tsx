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

import { ChangeEvent, useState, useRef, DragEvent } from 'react'
import PromptTextarea from '../../components/PromptTextarea'

interface AnimateFrameModeProps {
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
}

function AnimateFrameMode({
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
  isEnhancingPrompt = false
}: AnimateFrameModeProps) {
  const [isDragOver, setIsDragOver] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)

  const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      onFileUpload(file)
    }
  }

  const handleDragOver = (e: DragEvent<HTMLLabelElement>) => {
    e.preventDefault()
    setIsDragOver(true)
  }

  const handleDragLeave = (e: DragEvent<HTMLLabelElement>) => {
    e.preventDefault()
    setIsDragOver(false)
  }

  const handleDrop = (e: DragEvent<HTMLLabelElement>) => {
    e.preventDefault()
    setIsDragOver(false)
    const file = e.dataTransfer.files[0]
    if (file) {
      onFileUpload(file)
    }
  }

  return (
    <div className="space-y-6">
      {/* Upload Section */}
      <section>
        <div className="flex items-center justify-between mb-3">
          <div>
            <h3 className="text-[15px] font-semibold">Upload Image</h3>
            <p className="text-caption text-gm-text-tertiary-light dark:text-gm-text-tertiary mt-0.5">
              Select a model image to animate
            </p>
          </div>
        </div>

        {uploadedImage ? (
          <div className="relative w-32 h-32 rounded-xl overflow-hidden bg-white dark:bg-white/10 group">
            <img
              src={uploadedImage}
              alt="Uploaded asset"
              className="w-full h-full object-contain p-2"
            />
            <button
              onClick={onRemoveFile}
              className="absolute top-2 right-2 w-7 h-7 bg-black/50 hover:bg-black/70 backdrop-blur-sm rounded-full flex items-center justify-center transition-all opacity-0 group-hover:opacity-100"
            >
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none">
                <path d="M18 6L6 18M6 6L18 18" stroke="white" strokeWidth="2" strokeLinecap="round"/>
              </svg>
            </button>
          </div>
        ) : (
          <label
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            className={`upload-zone-compact cursor-pointer ${isDragOver ? 'border-gm-accent bg-gm-accent/10' : ''}`}
          >
            <input
              ref={inputRef}
              type="file"
              accept="image/png,image/jpeg,image/tiff"
              onChange={handleFileChange}
              className="hidden"
            />
            <div className={`w-10 h-10 rounded-lg flex items-center justify-center flex-shrink-0 transition-colors ${
              isDragOver ? 'bg-gm-accent/30' : 'bg-gm-accent/10 dark:bg-gm-accent/20'
            }`}>
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" className="text-gm-accent">
                <path d="M12 15V3M12 3L8 7M12 3L16 7" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
                <path d="M2 17v2a2 2 0 002 2h16a2 2 0 002-2v-2" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
              </svg>
            </div>
            <div>
              <p className="text-button font-medium">{isDragOver ? 'Release to upload' : 'Drop file or click to upload'}</p>
              <p className="text-[11px] text-gm-text-tertiary-light dark:text-gm-text-tertiary mt-0.5">PNG, JPG, TIFF supported</p>
            </div>
          </label>
        )}
      </section>

      {/* Divider */}
      <div className="divider" />

      {/* Animation Description */}
      <section>
        <div className="mb-3">
          <h3 className="text-[15px] font-semibold">Animation Description</h3>
          <p className="text-caption text-gm-text-tertiary-light dark:text-gm-text-tertiary mt-0.5">
            Describe movements, timing, and expressions
          </p>
        </div>
        <PromptTextarea
          value={animationDescription}
          onChange={onAnimationDescriptionChange}
          placeholder="The model slowly rotates head to camera, stares briefly, then turns to show side profile..."
          readOnly={!!generatedVideo}
          className="h-28"
          showGeminiButton={!!onEnhancePrompt && !generatedVideo}
          onGeminiClick={onEnhancePrompt}
          isEnhancing={isEnhancingPrompt}
        />
      </section>

      {/* Generate Button */}
      {generatedVideo ? (
        <button
          onClick={onRegenerate}
          className="btn-secondary w-full py-3.5 rounded-xl text-button font-medium flex items-center justify-center gap-2"
        >
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
            <path d="M1 4v6h6M23 20v-6h-6" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
            <path d="M20.49 9A9 9 0 005.64 5.64L1 10m22 4l-4.64 4.36A9 9 0 013.51 15" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
          </svg>
          Regenerate
        </button>
      ) : (
        <button
          onClick={onGenerate}
          disabled={isGenerating || !uploadedImage}
          className="btn-primary w-full py-3.5 rounded-xl text-button font-medium flex items-center justify-center gap-2"
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
                <path d="M12 2L2 7l10 5 10-5-10-5z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                <path d="M2 17l10 5 10-5M2 12l10 5 10-5" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
              </svg>
              Generate Animation
            </>
          )}
        </button>
      )}
    </div>
  )
}

export default AnimateFrameMode
