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

import { useState, useRef, DragEvent, ChangeEvent } from 'react'

interface UploadZoneProps {
  onFileSelect: (file: File) => void
  accept?: string
  label?: string
  sublabel?: string
  className?: string
  previewUrl?: string | null
  onRemove?: () => void
}

function UploadZone({
  onFileSelect,
  accept = 'image/png,image/jpeg,image/tiff',
  label = 'Drop your file here',
  sublabel,
  className = '',
  previewUrl,
  onRemove
}: UploadZoneProps) {
  const [isDragOver, setIsDragOver] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)

  const handleDragOver = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    setIsDragOver(true)
  }

  const handleDragLeave = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    setIsDragOver(false)
  }

  const handleDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    setIsDragOver(false)
    const file = e.dataTransfer.files[0]
    if (file) {
      onFileSelect(file)
    }
  }

  const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      onFileSelect(file)
    }
  }

  const handleClick = () => {
    inputRef.current?.click()
  }

  // Parse accepted file types for display
  const getAcceptedTypes = () => {
    return accept
      .split(',')
      .map(type => type.split('/')[1]?.toUpperCase())
      .filter(Boolean)
  }

  if (previewUrl) {
    return (
      <div className={`relative rounded-xl overflow-hidden bg-gradient-to-b from-black/[0.02] to-black/[0.04] dark:from-white/[0.02] dark:to-white/[0.04] group animate-scale-in ${className}`}>
        <img
          src={previewUrl}
          alt="Uploaded file"
          className="w-full aspect-square object-contain"
        />

        {/* Hover overlay */}
        <div className="absolute inset-0 bg-gradient-to-t from-black/60 via-transparent to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300" />

        {onRemove && (
          <button
            onClick={onRemove}
            className="absolute top-3 right-3 w-8 h-8 bg-black/40 hover:bg-black/60 backdrop-blur-sm rounded-full flex items-center justify-center transition-all duration-200 opacity-0 group-hover:opacity-100 hover:scale-110"
          >
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none">
              <path d="M18 6L6 18M6 6L18 18" stroke="white" strokeWidth="2" strokeLinecap="round"/>
            </svg>
          </button>
        )}

        {/* File info badge */}
        <div className="absolute bottom-3 left-3 px-3 py-1.5 bg-black/40 backdrop-blur-sm rounded-lg text-white text-caption opacity-0 group-hover:opacity-100 transition-opacity duration-300">
          Image uploaded
        </div>
      </div>
    )
  }

  return (
    <div
      onClick={handleClick}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
      className={`upload-zone h-40 cursor-pointer ${isDragOver ? 'drag-over' : ''} ${className}`}
    >
      <input
        ref={inputRef}
        type="file"
        accept={accept}
        onChange={handleFileChange}
        className="hidden"
      />

      <div className="flex flex-col items-center gap-3 relative z-10">
        {/* Upload icon with animated background */}
        <div className={`w-12 h-12 rounded-xl flex items-center justify-center transition-all duration-300 ${
          isDragOver
            ? 'bg-gm-accent/30 scale-110'
            : 'bg-gm-accent/10 dark:bg-gm-accent/20'
        }`}>
          <svg
            width="24"
            height="24"
            viewBox="0 0 24 24"
            fill="none"
            className={`text-gm-accent transition-transform duration-300 ${isDragOver ? 'scale-110 -translate-y-1' : ''}`}
          >
            <path d="M12 15V3M12 3L8 7M12 3L16 7" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
            <path d="M2 17L2 19C2 20.1046 2.89543 21 4 21L20 21C21.1046 21 22 20.1046 22 19V17" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
          </svg>
        </div>

        {/* Label text */}
        <div className="text-center">
          <p className={`text-button font-medium mb-1 transition-colors duration-200 ${
            isDragOver ? 'text-gm-accent' : ''
          }`}>
            {isDragOver ? 'Release to upload' : label}
          </p>
          <p className="text-caption text-gm-text-tertiary-light dark:text-gm-text-tertiary">
            {sublabel || 'or click to browse'}
          </p>
        </div>

        {/* Accepted file type badges */}
        <div className="flex items-center gap-1.5 mt-1">
          {getAcceptedTypes().map((type, index) => (
            <span
              key={index}
              className="px-2 py-0.5 rounded bg-black/[0.04] dark:bg-white/[0.06] text-[10px] font-medium text-gm-text-tertiary-light dark:text-gm-text-tertiary"
            >
              {type}
            </span>
          ))}
        </div>
      </div>
    </div>
  )
}

export default UploadZone
