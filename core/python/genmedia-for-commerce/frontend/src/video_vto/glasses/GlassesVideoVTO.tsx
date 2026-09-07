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

import { useState, useEffect } from 'react'
import { PRODUCTS, CAPABILITIES } from '../../config/featureConstraints'
import type { Product } from '../../config/featureConstraints'
import VideoVTOForm from '../components/VideoVTOForm'
import BackgroundVideo from '../../components/BackgroundVideo'
import { openFeedbackForm } from '../../components/FeedbackButton'
import {
  generateAnimatedVideo,
  generateAnimationPrompt,
  dataUrlToBlob,
  base64ToVideoUrl,
  downloadVideoFromUrl,
} from '../services/videoVtoApi'

interface GlassesVideoVTOProps {
  prefilledImage?: string | null
  prefilledPrompt?: string
  prefilledSubMode?: 'animate-frame' | 'custom-scene'
  showVideo?: boolean
  currentProduct?: Product
  availableProducts?: Product[]
  onProductChange?: (product: Product) => void
}

function GlassesVideoVTO({
  prefilledImage,
  prefilledPrompt = '',
  prefilledSubMode = 'animate-frame',
  showVideo = true,
  currentProduct,
  availableProducts,
  onProductChange
}: GlassesVideoVTOProps) {
  const [subMode, setSubMode] = useState<'animate-frame' | 'custom-scene'>(prefilledSubMode)
  const [uploadedImage, setUploadedImage] = useState<string | null>(prefilledImage || null)
  const [animationDescription, setAnimationDescription] = useState(prefilledPrompt)
  const [isGenerating, setIsGenerating] = useState(false)
  const [generatedVideos, setGeneratedVideos] = useState<string[]>([])
  const [videoFilenames, setVideoFilenames] = useState<string[]>([])
  const [selectedVideoIndex, setSelectedVideoIndex] = useState(0)
  const [error, setError] = useState<string | null>(null)
  const [isEnhancingPrompt, setIsEnhancingPrompt] = useState(false)
  const [modelImage, setModelImage] = useState<string | null>(null)

  // For backwards compatibility with VideoVTOForm
  const generatedVideo = generatedVideos.length > 0 ? generatedVideos[selectedVideoIndex] : null

  // Cleanup blob URLs on unmount
  useEffect(() => {
    return () => {
      generatedVideos.forEach(url => URL.revokeObjectURL(url))
    }
  }, [generatedVideos])

  const handleFileUpload = (file: File) => {
    const reader = new FileReader()
    reader.onloadend = () => {
      setUploadedImage(reader.result as string)
    }
    reader.readAsDataURL(file)
  }

  const handleRemoveFile = () => {
    setUploadedImage(null)
  }

  const handleGenerate = async () => {
    // For animate-frame mode: need uploadedImage (the frame to animate)
    // For custom-scene mode: need uploadedImage (product/glasses), modelImage is optional (face photo)
    console.log('handleGenerate called', { subMode, uploadedImage: !!uploadedImage, modelImage: !!modelImage })

    if (subMode === 'animate-frame' && !uploadedImage) {
      console.log('Validation failed: animate-frame mode requires uploadedImage')
      return
    }
    if (subMode === 'custom-scene' && !uploadedImage) {
      console.log('Validation failed: custom-scene mode requires uploadedImage (product image)', { uploadedImage: !!uploadedImage, modelImage: !!modelImage })
      return
    }

    console.log('Starting generation...')
    setIsGenerating(true)
    setError(null)

    try {
      // In animate-frame mode: uploadedImage is the model/frame to animate
      // In custom-scene mode: uploadedImage is product image, modelImage is face photo
      const isAnimationMode = subMode === 'animate-frame'
      console.log('Generation mode:', isAnimationMode ? 'animate-frame' : 'custom-scene')

      let response
      if (isAnimationMode) {
        const imageBlob = dataUrlToBlob(uploadedImage!)
        response = await generateAnimatedVideo({
          prompt: animationDescription,
          modelImage: imageBlob,
          isAnimationMode: true,
          numberOfVideos: 4,
        })
      } else {
        // Custom scene mode: send product image and optionally model image
        const productImageBlob = dataUrlToBlob(uploadedImage!)
        const modelImageBlob = modelImage ? dataUrlToBlob(modelImage) : undefined
        response = await generateAnimatedVideo({
          prompt: animationDescription,
          modelImage: modelImageBlob,
          productImage: productImageBlob,
          isAnimationMode: false,
          numberOfVideos: 4,
        })
      }

      // Clean up old blob URLs
      generatedVideos.forEach(url => URL.revokeObjectURL(url))

      // Convert base64 videos to blob URLs
      const videoUrls = response.videos.map(base64ToVideoUrl)

      setGeneratedVideos(videoUrls)
      setVideoFilenames(response.filenames)
      setSelectedVideoIndex(0)
    } catch (err) {
      console.error('Error generating video:', err)
      setError(err instanceof Error ? err.message : 'Failed to generate video')
    } finally {
      setIsGenerating(false)
    }
  }

  const handleRegenerate = () => {
    // Clean up blob URLs
    generatedVideos.forEach(url => URL.revokeObjectURL(url))
    setGeneratedVideos([])
    setVideoFilenames([])
    setSelectedVideoIndex(0)
    setUploadedImage(null)
    setModelImage(null)
    setError(null)
  }

  const handleSubModeChange = (newSubMode: 'animate-frame' | 'custom-scene') => {
    setSubMode(newSubMode)
    // Reset all state when switching sub-modes
    generatedVideos.forEach(url => URL.revokeObjectURL(url))
    setGeneratedVideos([])
    setVideoFilenames([])
    setSelectedVideoIndex(0)
    setUploadedImage(null)
    setModelImage(null)
    setAnimationDescription('')
    setError(null)
  }

  const handleDownload = (index: number) => {
    const url = generatedVideos[index]
    const filename = videoFilenames[index] || `animation_${index + 1}.mp4`
    downloadVideoFromUrl(url, filename)
  }

  const handleEnhancePrompt = async () => {
    if (!animationDescription.trim()) return

    setIsEnhancingPrompt(true)
    try {
      const imageBlob = uploadedImage ? dataUrlToBlob(uploadedImage) : undefined
      const response = await generateAnimationPrompt({
        text: animationDescription,
        modelImage: imageBlob,
      })
      setAnimationDescription(response.enhanced_prompt)
    } catch (err) {
      console.error('Error enhancing prompt:', err)
    } finally {
      setIsEnhancingPrompt(false)
    }
  }

  return (
    <div className="grid grid-cols-[480px_1fr] gap-6">
      <VideoVTOForm
        product={PRODUCTS.GLASSES}
        capability={CAPABILITIES.VIDEO_VTO}
        subMode={subMode}
        onSubModeChange={handleSubModeChange}
        uploadedImage={uploadedImage}
        animationDescription={animationDescription}
        onAnimationDescriptionChange={setAnimationDescription}
        onFileUpload={handleFileUpload}
        onRemoveFile={handleRemoveFile}
        onGenerate={handleGenerate}
        isGenerating={isGenerating}
        generatedVideo={generatedVideo}
        onRegenerate={handleRegenerate}
        onEnhancePrompt={handleEnhancePrompt}
        isEnhancingPrompt={isEnhancingPrompt}
        modelImage={modelImage}
        onModelImageChange={setModelImage}
        currentProduct={currentProduct}
        availableProducts={availableProducts}
        onProductChange={onProductChange}
      />

      {/* Preview Panel */}
      <div className={`${showVideo ? 'glass-panel' : 'border border-black/[0.08] dark:border-white/10'} rounded-lg flex items-center justify-center min-h-[600px] overflow-hidden relative`}>
        {error ? (
          <div className="text-center px-8">
            <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-red-500/10 flex items-center justify-center">
              <svg width="32" height="32" viewBox="0 0 24 24" fill="none" className="text-red-500">
                <path d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
              </svg>
            </div>
            <h3 className="text-xl font-semibold mb-2">Generation Failed</h3>
            <p className="text-gm-text-secondary mb-3">{error}</p>
            <button
              onClick={() => openFeedbackForm({ feedbackType: 'Bug / Error', capability: 'Video VTO', errorMessage: error || undefined })}
              className="text-sm text-gm-accent hover:text-gm-accent-hover transition-colors underline underline-offset-2 mb-4 block"
            >
              Report this issue
            </button>
            <button
              onClick={() => setError(null)}
              className="btn-secondary px-4 py-2 rounded-lg"
            >
              Try Again
            </button>
          </div>
        ) : generatedVideos.length > 0 ? (
          <div className="w-full h-full p-6 flex flex-col">
            {/* Primary Video Player */}
            <div className="flex-1 mb-4 relative rounded-lg overflow-hidden bg-black">
              <video
                key={generatedVideos[selectedVideoIndex]}
                controls
                autoPlay
                className="w-full h-full object-contain rounded-lg"
              >
                <source src={generatedVideos[selectedVideoIndex]} type="video/mp4" />
              </video>
              <div className="absolute top-4 right-4 flex gap-2">
                <button
                  onClick={() => handleDownload(selectedVideoIndex)}
                  className="w-10 h-10 bg-black/70 hover:bg-black/90 rounded-lg flex items-center justify-center transition-all"
                  title="Download video"
                >
                  <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
                    <path d="M12 15V3M12 15L8 11M12 15L16 11" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                    <path d="M2 17L2 19C2 20.1046 2.89543 21 4 21L20 21C21.1046 21 22 20.1046 22 19V17" stroke="white" strokeWidth="2" strokeLinecap="round"/>
                  </svg>
                </button>
              </div>
            </div>

            {/* Video Thumbnails Grid */}
            <div className="grid grid-cols-4 gap-3">
              {generatedVideos.map((videoUrl, index) => (
                <div key={index} className="relative">
                  <button
                    onClick={() => setSelectedVideoIndex(index)}
                    className={`w-full aspect-video bg-black/30 rounded-lg overflow-hidden mb-2 transition-all ${
                      selectedVideoIndex === index
                        ? 'ring-2 ring-gm-accent ring-offset-2 ring-offset-black/20'
                        : 'hover:ring-1 hover:ring-white/30'
                    }`}
                  >
                    <video
                      className="w-full h-full object-cover"
                      muted
                      playsInline
                      onMouseEnter={(e) => e.currentTarget.play()}
                      onMouseLeave={(e) => {
                        e.currentTarget.pause()
                        e.currentTarget.currentTime = 0
                      }}
                    >
                      <source src={videoUrl} type="video/mp4" />
                    </video>
                    {selectedVideoIndex === index && (
                      <div className="absolute inset-0 flex items-center justify-center bg-black/30">
                        <div className="w-6 h-6 rounded-full bg-gm-accent flex items-center justify-center">
                          <svg width="12" height="12" viewBox="0 0 24 24" fill="none">
                            <path d="M20 6L9 17l-5-5" stroke="white" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round"/>
                          </svg>
                        </div>
                      </div>
                    )}
                  </button>
                  <div className="flex gap-2">
                    <span className="flex-1 text-xs text-center text-gm-text-tertiary">
                      Video {index + 1}
                    </span>
                    <button
                      onClick={() => handleDownload(index)}
                      className="w-8 h-8 bg-black/[0.06] hover:bg-black/[0.1] dark:bg-white/10 dark:hover:bg-white/20 rounded-md flex items-center justify-center transition-all"
                      title="Download"
                    >
                      <svg width="14" height="14" viewBox="0 0 24 24" fill="none">
                        <path d="M12 15V3M12 15L8 11M12 15L16 11" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                        <path d="M2 17L2 19C2 20.1046 2.89543 21 4 21L20 21C21.1046 21 22 20.1046 22 19V17" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
                      </svg>
                    </button>
                  </div>
                </div>
              ))}
            </div>
          </div>
        ) : (
          <>
            {showVideo && <BackgroundVideo />}

            <div className="absolute inset-0 bg-gradient-to-t from-gm-bg-light/50 via-gm-bg-light/20 to-transparent dark:from-gm-bg/50 dark:via-gm-bg/20"></div>

            {isGenerating ? (
              <div className="text-center px-8 relative z-10">
                <div className="relative mb-6">
                  <div className="w-16 h-16 rounded-full border-2 border-gm-accent/30 animate-ping absolute inset-0 mx-auto" />
                  <div className="w-16 h-16 rounded-full border-2 border-gm-accent/20 animate-pulse mx-auto" />
                  <div className="absolute inset-0 flex items-center justify-center">
                    <div className="w-3 h-3 rounded-full bg-gm-accent animate-pulse" />
                  </div>
                </div>
                <h2 className="text-[32px] font-bold mb-3 leading-tight">
                  Generating your <span className="text-gm-accent" style={{ textShadow: '0 0 30px rgba(138, 180, 248, 0.5), 0 0 60px rgba(138, 180, 248, 0.3)' }}>video</span>...
                </h2>
                <p className="text-lg text-gm-text-secondary-light dark:text-gm-text-secondary">
                  Our AI is crafting your animation. This may take a moment.
                </p>
              </div>
            ) : (
              <div className="text-center px-8 relative z-10 max-w-lg animate-fade-in-up">
                <div className="w-16 h-16 mx-auto mb-6 rounded-2xl bg-gm-accent/10 dark:bg-gm-accent/20 flex items-center justify-center glow-accent">
                  <svg width="28" height="28" viewBox="0 0 24 24" fill="none" className="text-gm-accent">
                    <path d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14v-4z" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
                    <rect x="3" y="6" width="12" height="12" rx="2" stroke="currentColor" strokeWidth="1.5" />
                  </svg>
                </div>

                <h2 className="text-[40px] lg:text-[48px] font-display font-bold mb-4 leading-tight tracking-tight">
                  Glasses{' '}
                  <span className="text-gm-accent" style={{ textShadow: '0 0 40px rgba(138, 180, 248, 0.4), 0 0 80px rgba(138, 180, 248, 0.2)' }}>
                    Video VTO
                  </span>
                </h2>
                <p className="text-body-lg text-gm-text-secondary-light dark:text-gm-text-secondary leading-relaxed">
                  Upload a model image and eyewear to generate an animated video showcasing the glasses.
                </p>

                <div className="flex flex-wrap items-center justify-center gap-2 mt-6">
                  <span className="px-3 py-1.5 rounded-full bg-black/[0.08] dark:bg-white/[0.12] text-caption text-gm-text-secondary-light dark:text-gm-text-secondary">
                    Veo 3.1
                  </span>
                  <span className="px-3 py-1.5 rounded-full bg-black/[0.08] dark:bg-white/[0.12] text-caption text-gm-text-secondary-light dark:text-gm-text-secondary">
                    Animate Frame
                  </span>
                  <span className="px-3 py-1.5 rounded-full bg-black/[0.08] dark:bg-white/[0.12] text-caption text-gm-text-secondary-light dark:text-gm-text-secondary">
                    Eyewear Fitting
                  </span>
                </div>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  )
}

export default GlassesVideoVTO
