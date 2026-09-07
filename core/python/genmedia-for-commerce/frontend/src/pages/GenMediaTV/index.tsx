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

import { useState, useRef, useEffect } from 'react'
import { useNavigate, useLocation } from 'react-router-dom'
import ControlBar from './ControlBar'
import ImageVTODisplay from './ImageVTODisplay'

interface VideoPrompt {
  input_subject?: string
  subject?: string
  action?: string
  scene?: string
  camera_angles_and_movements?: string
  lighting?: string
  negative_prompt?: string
}

interface Template {
  video_path: string
  product_img?: string
  product_images?: string[]
  video_prompt?: VideoPrompt
}

interface ImageVTOTemplate {
  model_image: string
  garment_images: string[]
  vto_result: string
  description: string
  scenario: string
}

interface VideoVTOTemplate {
  model_image: string
  garment_images: string[]
  vto_result: string
  video_path: string
  description: string
  prompt: string
}

interface GlassesImageVTOTemplate {
  model_image: string
  glasses_image: string
  vto_result: string
  description: string
}

interface GlassesTemplatesData {
  men: Template[]
  women: Template[]
}

interface ImageVTOTemplatesData {
  men: ImageVTOTemplate[]
  women: ImageVTOTemplate[]
}

interface VideoVTOTemplatesData {
  men: VideoVTOTemplate[]
  women: VideoVTOTemplate[]
}

interface ProcessedTemplate {
  video_path: string
  product_img: string
  product_images?: string[]
  prompt: string
}

interface ProcessedImageVTO {
  model_image: string
  garment_images: string[]
  vto_result: string
  description: string
}

interface VideoAsset {
  id: number
  src: string
  prompt: string
  productImg: string
  productImages?: string[]
  aspectRatio: string
  channel?: string
  contentType?: 'video' | 'image'
  imageVTOData?: ProcessedImageVTO
  videoVTOData?: { model_image: string; garment_images: string[]; vto_result: string; prompt: string }
}

function GenMediaTV() {
  const navigate = useNavigate()
  const location = useLocation()
  const videoRef = useRef<HTMLVideoElement>(null)
  const [isPlaying, setIsPlaying] = useState(true)
  const [showPrompt, setShowPrompt] = useState(true)
  const [isShuffleMode, setIsShuffleMode] = useState(() => {
    const locationState = location.state as { shuffle?: boolean; selectedCategory?: string } | null
    return locationState?.shuffle || false
  })
  const [selectedChannel, setSelectedChannel] = useState(() => {
    const locationState = location.state as { shuffle?: boolean; selectedCategory?: string } | null
    const category = locationState?.selectedCategory || 'product-360'

    // Map specific categories to main channels
    if (category.endsWith('-360')) {
      return 'product-360'
    }
    return category
  })
  const [selectedCategory, setSelectedCategory] = useState(() => {
    const locationState = location.state as { shuffle?: boolean; selectedCategory?: string } | null
    return locationState?.selectedCategory || 'product-360'
  })
  const [isGridView, setIsGridView] = useState(false)
  const [currentVideoIndex, setCurrentVideoIndex] = useState(0)
  const [glassesTemplatesData, setGlassesTemplatesData] = useState<GlassesTemplatesData>({ men: [], women: [] })
  const [product360TemplatesData, setProduct360TemplatesData] = useState<Template[]>([])
  const [imageVTOTemplatesData, setImageVTOTemplatesData] = useState<ImageVTOTemplatesData>({ men: [], women: [] })
  const [videoVTOTemplatesData, setVideoVTOTemplatesData] = useState<VideoVTOTemplatesData>({ men: [], women: [] })
  const [glassesImageVTOTemplatesData, setGlassesImageVTOTemplatesData] = useState<GlassesImageVTOTemplate[]>([])
  const [isDarkMode, setIsDarkMode] = useState(() => {
    const saved = localStorage.getItem('darkMode')
    return saved !== null ? JSON.parse(saved) : true
  })

  const mergePrompt = (videoPrompt?: VideoPrompt): string => {
    if (!videoPrompt) return "No prompt available"

    const parts: string[] = []

    if (videoPrompt.input_subject) parts.push(videoPrompt.input_subject)
    if (videoPrompt.subject) parts.push(`[Subject]: ${videoPrompt.subject}`)
    if (videoPrompt.action) parts.push(`[Action]: ${videoPrompt.action}`)
    if (videoPrompt.scene) parts.push(videoPrompt.scene)
    if (videoPrompt.camera_angles_and_movements) parts.push(videoPrompt.camera_angles_and_movements)
    if (videoPrompt.lighting) parts.push(videoPrompt.lighting)
    if (videoPrompt.negative_prompt) parts.push(`[Negative Prompt]: ${videoPrompt.negative_prompt}`)

    return parts.join(' ')
  }

  useEffect(() => {
    const safeFetch = (url: string) => fetch(url)
      .then(r => {
        const ct = r.headers.get('content-type') || ''
        return r.ok && ct.includes('application/json') ? r.json() : []
      })
      .catch(() => [])

    Promise.all([
      safeFetch('/products/glasses/templates.json'),
      safeFetch('/templates/360-templates.json'),
      safeFetch('/products/clothes/image-vto-templates.json'),
      safeFetch('/products/clothes/video-vto-templates.json'),
      safeFetch('/products/glasses/image-vto-templates.json')
    ])
      .then(([glassesData, product360Data, imageVTOData, videoVTOData, glassesImageVTOData]) => {
        setGlassesTemplatesData(glassesData)
        setProduct360TemplatesData(product360Data)
        setImageVTOTemplatesData(imageVTOData)
        setVideoVTOTemplatesData(videoVTOData)
        setGlassesImageVTOTemplatesData(glassesImageVTOData)
      })
      .catch(error => {
        console.error('Error loading templates:', error)
      })
  }, [])

  useEffect(() => {
    localStorage.setItem('darkMode', JSON.stringify(isDarkMode))
    if (isDarkMode) {
      document.documentElement.classList.add('dark')
    } else {
      document.documentElement.classList.remove('dark')
    }
  }, [isDarkMode])

  const getProductImage = (template: Template): string => {
    if (template.product_images && template.product_images.length > 1) {
      return template.product_images[1]
    }
    return template.product_img || ''
  }

  const glassesTemplates: { men: ProcessedTemplate[], women: ProcessedTemplate[] } = {
    men: glassesTemplatesData.men.map(template => ({
      video_path: template.video_path,
      product_img: getProductImage(template),
      product_images: template.product_images,
      prompt: mergePrompt(template.video_prompt)
    })),
    women: glassesTemplatesData.women.map(template => ({
      video_path: template.video_path,
      product_img: getProductImage(template),
      product_images: template.product_images,
      prompt: mergePrompt(template.video_prompt)
    }))
  }

  const product360Templates: ProcessedTemplate[] = product360TemplatesData
    .filter(template => {
      if (!template.video_path) return false

      // Filter by specific category if selected
      if (selectedCategory !== 'product-360') {
        const productImg = getProductImage(template)
        if (selectedCategory === 'shoes-360' && !productImg.toLowerCase().includes('shoe')) return false
        if (selectedCategory === 'clothes-360' && !(productImg.toLowerCase().includes('shirt') || productImg.toLowerCase().includes('clothes'))) return false
        if (selectedCategory === 'cars-360' && !productImg.toLowerCase().includes('car')) return false
        if (selectedCategory === 'accessories-360' && !(productImg.toLowerCase().includes('smartphone') || productImg.toLowerCase().includes('phone'))) return false
        if (selectedCategory === 'glasses-360' && !productImg.toLowerCase().includes('glass')) return false
      }

      return true
    })
    .map(template => ({
      video_path: template.video_path,
      product_img: getProductImage(template),
      product_images: template.product_images,
      prompt: mergePrompt(template.video_prompt)
    }))

  const imageVTOTemplates: { men: ProcessedImageVTO[], women: ProcessedImageVTO[] } = {
    men: imageVTOTemplatesData.men.map(template => ({
      model_image: template.model_image,
      garment_images: template.garment_images,
      vto_result: template.vto_result,
      description: `${template.description} - ${template.scenario}`
    })),
    women: imageVTOTemplatesData.women.map(template => ({
      model_image: template.model_image,
      garment_images: template.garment_images,
      vto_result: template.vto_result,
      description: `${template.description} - ${template.scenario}`
    }))
  }

  const videoVTOTemplates = {
    men: videoVTOTemplatesData.men.map(template => ({
      model_image: template.model_image,
      garment_images: template.garment_images,
      vto_result: template.vto_result,
      video_path: template.video_path,
      description: template.description,
      prompt: template.prompt
    })),
    women: videoVTOTemplatesData.women.map(template => ({
      model_image: template.model_image,
      garment_images: template.garment_images,
      vto_result: template.vto_result,
      video_path: template.video_path,
      description: template.description,
      prompt: template.prompt
    }))
  }

  // Create video assets for all channels
  const allChannelVideos = {
    'glasses-vto': [
      ...glassesTemplates.women.map((template, index) => ({
        id: index + 1,
        src: template.video_path,
        prompt: template.prompt,
        productImg: template.product_img,
        productImages: template.product_images,
        aspectRatio: 'aspect-[16/9]',
        channel: 'glasses-vto' as const,
        contentType: 'video' as const
      })),
      ...glassesTemplates.men.map((template, index) => ({
        id: index + 100,
        src: template.video_path,
        prompt: template.prompt,
        productImg: template.product_img,
        productImages: template.product_images,
        aspectRatio: 'aspect-[16/9]',
        channel: 'glasses-vto' as const,
        contentType: 'video' as const
      }))
    ],
    'product-360': product360Templates.map((template, index) => ({
      id: index + 200,
      src: template.video_path,
      prompt: template.prompt,
      productImg: template.product_img,
      productImages: template.product_images,
      aspectRatio: 'aspect-[16/9]',
      channel: 'product-360' as const,
      contentType: 'video' as const
    })),
    'clothes-image-vto': [
      ...imageVTOTemplates.women.map((template, index) => ({
        id: index + 300,
        src: template.vto_result,
        prompt: template.description,
        productImg: template.garment_images[0],
        productImages: template.garment_images,
        aspectRatio: 'aspect-[16/9]',
        channel: 'clothes-image-vto' as const,
        contentType: 'image' as const,
        imageVTOData: template
      })),
      ...imageVTOTemplates.men.map((template, index) => ({
        id: index + 400,
        src: template.vto_result,
        prompt: template.description,
        productImg: template.garment_images[0],
        productImages: template.garment_images,
        aspectRatio: 'aspect-[16/9]',
        channel: 'clothes-image-vto' as const,
        contentType: 'image' as const,
        imageVTOData: template
      }))
    ],
    'clothes-video-vto': [
      ...videoVTOTemplates.men.map((template, index) => ({
        id: index + 500,
        src: template.video_path,
        prompt: template.description,
        productImg: template.vto_result,
        productImages: template.garment_images,
        aspectRatio: 'aspect-[16/9]',
        channel: 'clothes-video-vto' as const,
        contentType: 'video' as const,
        videoVTOData: template
      })),
      ...videoVTOTemplates.women.map((template, index) => ({
        id: index + 600,
        src: template.video_path,
        prompt: template.description,
        productImg: template.vto_result,
        productImages: template.garment_images,
        aspectRatio: 'aspect-[16/9]',
        channel: 'clothes-video-vto' as const,
        contentType: 'video' as const,
        videoVTOData: template
      }))
    ],
    'glasses-image-vto': glassesImageVTOTemplatesData.map((template, index) => ({
      id: index + 700,
      src: template.vto_result,
      prompt: template.description,
      productImg: template.glasses_image,
      productImages: [template.glasses_image],
      aspectRatio: 'aspect-[16/9]',
      channel: 'glasses-image-vto' as const,
      contentType: 'image' as const,
      imageVTOData: {
        model_image: template.model_image,
        garment_images: [template.glasses_image],
        vto_result: template.vto_result,
        description: template.description
      }
    }))
  }

  const videoAssets: VideoAsset[] = isShuffleMode
    ? (() => {
        // Combine all videos and shuffle
        const allVideos = [
          // TODO: Re-enable when glasses video VTO quality is improved
          // ...allChannelVideos['glasses-vto'],
          ...allChannelVideos['product-360'],
          ...allChannelVideos['clothes-image-vto'],
          ...allChannelVideos['clothes-video-vto'],
          ...allChannelVideos['glasses-image-vto']
        ]
        return allVideos.sort(() => Math.random() - 0.5)
      })()
    : selectedChannel === 'product-360'
    ? allChannelVideos['product-360']
    : selectedChannel === 'clothes-image-vto'
    ? allChannelVideos['clothes-image-vto']
    : selectedChannel === 'clothes-video-vto'
    ? allChannelVideos['clothes-video-vto']
    : selectedChannel === 'glasses-image-vto'
    ? allChannelVideos['glasses-image-vto']
    : []

  const currentVideo = videoAssets[currentVideoIndex] || videoAssets[0]
  const currentPrompt = currentVideo?.prompt || "No prompt available"

  // Update selected channel to match current video when in shuffle mode
  useEffect(() => {
    if (isShuffleMode && currentVideo?.channel && selectedChannel !== currentVideo.channel) {
      setSelectedChannel(currentVideo.channel)
    }
  }, [currentVideoIndex, isShuffleMode, currentVideo, selectedChannel])

  // Auto-advance slideshow for image VTO content
  useEffect(() => {
    // Only set up timer for image content type and when playing
    if (currentVideo?.contentType === 'image' && isPlaying) {
      const slideshowInterval = setInterval(() => {
        const newIndex = (currentVideoIndex + 1) % videoAssets.length
        setCurrentVideoIndex(newIndex)

        // In shuffle mode, update the selected channel to match the video's channel
        if (isShuffleMode && videoAssets[newIndex]?.channel) {
          setSelectedChannel(videoAssets[newIndex].channel!)
        }
      }, 6000) // 6 seconds per image

      return () => clearInterval(slideshowInterval)
    }
  }, [currentVideoIndex, currentVideo?.contentType, isPlaying, videoAssets.length, isShuffleMode, videoAssets])

  const channels = [
    { id: 'product-360', name: 'PRODUCT 360', thumbnail: '/products/360-videos/shoes-360.mp4' },
    { id: 'clothes-video-vto', name: 'CLOTHES VIDEO VTO', thumbnail: '/products/clothes/videos/women_vto_001.mp4' },
    { id: 'clothes-image-vto', name: 'CLOTHES IMAGE VTO', thumbnail: '/assets/models/european_woman.png' },
    // TODO: Re-enable when glasses video VTO quality is improved
    // { id: 'glasses-vto', name: 'GLASSES VIDEO VTO', thumbnail: '/products/glasses/videos/women/w_pose.mp4' },
    { id: 'glasses-image-vto', name: 'GLASSES IMAGE VTO', thumbnail: '/products/glasses/images/glasses_4.jpeg' },
  ]

  const togglePlay = () => {
    // For image content, just toggle the playing state (controls slideshow)
    if (currentVideo?.contentType === 'image') {
      setIsPlaying(!isPlaying)
    }
    // For video content, control the video element
    else if (videoRef.current) {
      if (isPlaying) {
        videoRef.current.pause()
      } else {
        videoRef.current.play()
      }
      setIsPlaying(!isPlaying)
    }
  }

  const nextVideo = () => {
    const newIndex = (currentVideoIndex + 1) % videoAssets.length
    setCurrentVideoIndex(newIndex)

    // In shuffle mode, update the selected channel to match the video's channel
    if (isShuffleMode && videoAssets[newIndex]?.channel) {
      setSelectedChannel(videoAssets[newIndex].channel!)
    }
  }

  const prevVideo = () => {
    const newIndex = (currentVideoIndex - 1 + videoAssets.length) % videoAssets.length
    setCurrentVideoIndex(newIndex)

    // In shuffle mode, update the selected channel to match the video's channel
    if (isShuffleMode && videoAssets[newIndex]?.channel) {
      setSelectedChannel(videoAssets[newIndex].channel!)
    }
  }

  const handleVideoEnded = () => {
    const newIndex = (currentVideoIndex + 1) % videoAssets.length
    setCurrentVideoIndex(newIndex)

    // In shuffle mode, update the selected channel to match the video's channel
    if (isShuffleMode && videoAssets[newIndex]?.channel) {
      setSelectedChannel(videoAssets[newIndex].channel!)
    }
  }

  const handleChannelChange = (direction: 'prev' | 'next') => {
    const currentIndex = channels.findIndex(c => c.id === selectedChannel)
    const newIndex = direction === 'next'
      ? (currentIndex + 1) % channels.length
      : (currentIndex - 1 + channels.length) % channels.length
    setSelectedChannel(channels[newIndex].id)
    setSelectedCategory(channels[newIndex].id)
    setCurrentVideoIndex(0)
    setIsPlaying(true) // Ensure playback starts when switching channels
  }

  const handleExitShuffle = () => {
    setIsShuffleMode(false)
    setCurrentVideoIndex(0)
    setIsPlaying(true) // Ensure playback starts when exiting shuffle
  }

  const handleOpenInCreate = () => {
    let capability = 'video-vto'
    let product = 'glasses'

    if (selectedChannel === 'product-360') {
      capability = 'product-360'
      const productImg = currentVideo?.productImg || ''
      if (productImg.includes('shoe')) product = 'shoes'
      else product = 'other'
    } else if (selectedChannel === 'clothes-video-vto') {
      capability = 'video-vto'
      product = 'clothes'
    } else if (selectedChannel === 'clothes-image-vto') {
      capability = 'image-vto'
      product = 'clothes'
    } else if (selectedChannel === 'glasses-vto') {
      capability = 'video-vto'
      product = 'glasses'
    } else if (selectedChannel === 'glasses-image-vto') {
      capability = 'image-vto'
      product = 'glasses'
    }

    const stateData: {
      prefilledPrompt: string
      prefilledProduct: string
      prefilledSubMode: string
      prefilledSpinMode?: string
      prefilledImages?: string[]
      prefilledImage?: string
      prefilledModelImage?: string
    } = {
      prefilledPrompt: currentPrompt,
      prefilledProduct: product,
      prefilledSubMode: 'custom-scene'
    }

    // Auto-select spin mode for product-360: cars/phones → interpolation, others → r2v
    if (selectedChannel === 'product-360' && product === 'other') {
      const productImg = currentVideo?.productImg || ''
      if (productImg.includes('car') || productImg.includes('smartphone') || productImg.includes('phone')) {
        stateData.prefilledSpinMode = 'interpolation'
      } else {
        stateData.prefilledSpinMode = 'r2v-standard'
      }
    }

    // For Video VTO Clothes, pass original model image and garment images
    if (selectedChannel === 'clothes-video-vto' && currentVideo?.videoVTOData) {
      stateData.prefilledImage = currentVideo.videoVTOData.model_image
      stateData.prefilledImages = currentVideo.videoVTOData.garment_images
      stateData.prefilledPrompt = currentVideo.videoVTOData.prompt
    }
    // For Image VTO Clothes, pass model image and garment images separately
    else if (selectedChannel === 'clothes-image-vto' && currentVideo?.imageVTOData) {
      stateData.prefilledModelImage = currentVideo.imageVTOData.model_image
      stateData.prefilledImages = currentVideo.imageVTOData.garment_images
    }
    // For Image VTO Glasses, pass model image and glasses image
    else if (selectedChannel === 'glasses-image-vto' && currentVideo?.imageVTOData) {
      stateData.prefilledModelImage = currentVideo.imageVTOData.model_image
      stateData.prefilledImages = currentVideo.imageVTOData.garment_images
    } else if (currentVideo?.productImages && currentVideo.productImages.length > 1) {
      stateData.prefilledImages = currentVideo.productImages
    } else {
      stateData.prefilledImage = currentVideo?.productImg
    }

    navigate(`/create/${capability}`, { state: stateData })
  }

  return (
    <div className="min-h-screen bg-gm-bg-light dark:bg-gm-bg flex flex-col">
      {/* Header */}
      <header className="px-8 py-4 flex items-center justify-between">
        <div className="flex items-center gap-4">
          {/* Back Button */}
          <button
            onClick={() => navigate('/')}
            className="w-10 h-10 rounded-full bg-black/5 dark:bg-white/10 hover:bg-black/10 dark:hover:bg-white/20 flex items-center justify-center transition-all"
          >
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" className="text-gm-text-primary-light dark:text-gm-text-primary">
              <path d="M19 12H5M5 12L12 19M5 12L12 5" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
          </button>
          <h1 className="text-xl font-bold">GenMedia TV</h1>
        </div>

        {/* Search Bar — hidden for now, re-enable when search is implemented
        <div className="flex-1 max-w-xl mx-8">
          <div className="relative">
            <svg
              className="absolute left-4 top-1/2 -translate-y-1/2 text-gm-text-secondary-light dark:text-gm-text-secondary"
              width="20"
              height="20"
              viewBox="0 0 24 24"
              fill="none"
            >
              <circle cx="11" cy="11" r="7" stroke="currentColor" strokeWidth="2" />
              <path d="M20 20L16.5 16.5" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
            </svg>
            <input
              type="text"
              placeholder="Search assets, tools, or projects..."
              className="w-full bg-white/5 dark:bg-white/5 bg-black/5 border border-black/15 dark:border-white/10 rounded-full py-2.5 pl-12 pr-4 text-sm text-gm-text-primary-light dark:text-gm-text-primary placeholder-gm-text-tertiary-light dark:placeholder-gm-text-secondary focus:outline-none focus:border-gm-accent/50"
            />
          </div>
        </div>
        */}

        {/* Right Side */}
        <div className="flex items-center gap-4">
          {/* Dark/Light Mode Toggle */}
          <button
            onClick={() => setIsDarkMode(!isDarkMode)}
            className={`w-12 h-6 rounded-full relative transition-all ${
              isDarkMode ? 'bg-white/20' : 'bg-gm-accent'
            }`}
          >
            <div
              className={`absolute top-1 w-4 h-4 rounded-full bg-white transition-all flex items-center justify-center ${
                isDarkMode ? 'left-1' : 'left-7'
              }`}
            >
              {isDarkMode ? (
                <svg width="10" height="10" viewBox="0 0 24 24" fill="none">
                  <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z" fill="currentColor" className="text-gray-800"/>
                </svg>
              ) : (
                <svg width="10" height="10" viewBox="0 0 24 24" fill="none">
                  <circle cx="12" cy="12" r="5" fill="currentColor" className="text-yellow-500"/>
                  <path d="M12 1v2m0 18v2M4.22 4.22l1.42 1.42m12.72 12.72l1.42 1.42M1 12h2m18 0h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42" stroke="currentColor" strokeWidth="2" strokeLinecap="round" className="text-yellow-500"/>
                </svg>
              )}
            </div>
          </button>

          <button
            onClick={() => navigate('/create/video-vto')}
            className="btn-primary px-5 py-2.5 rounded-full text-sm font-medium"
          >
            Create with GenMedia
          </button>
        </div>
      </header>

      {/* Main Content */}
      <main className={`flex-1 px-8 pb-40 py-8 overflow-y-auto transition-all duration-500 ${
        isGridView ? '' : 'flex items-center justify-center'
      }`}>
        <div className={`mx-auto transition-all duration-500 ${
          isGridView ? 'max-w-[1400px] w-full' : 'w-full max-w-[1100px] px-16'
        }`}>
          <div className={`transition-all duration-500 ${
            isGridView ? 'grid grid-cols-3 gap-6' : 'grid grid-cols-1'
          }`}>
            {isGridView ? (
              videoAssets.map((asset, index) => (
                <div
                  key={`${selectedChannel}-${asset.id}`}
                  onClick={() => {
                    setCurrentVideoIndex(index)
                    setIsGridView(false)
                    // In shuffle mode, update channel to match the selected video
                    if (isShuffleMode && asset.channel) {
                      setSelectedChannel(asset.channel)
                    }
                  }}
                  className={`relative ${asset.aspectRatio} rounded-2xl overflow-hidden bg-black/20 dark:bg-black/20 bg-black/5 border border-black/10 dark:border-white/10 hover:border-black/15 dark:hover:border-white/20 transition-all duration-300 cursor-pointer group animate-in fade-in zoom-in-95`}
                  style={{ animationDelay: `${index * 50}ms`, animationFillMode: 'backwards' }}
                >
                  {asset.contentType === 'image' && asset.imageVTOData ? (
                    <div className="w-full h-full bg-gradient-to-br from-amber-50 via-white to-emerald-50 dark:from-gray-800 dark:via-gray-900 dark:to-gray-800 flex items-center justify-center p-2">
                      <img
                        src={asset.imageVTOData.vto_result}
                        alt="VTO Result"
                        className="max-w-full max-h-full object-contain"
                      />
                    </div>
                  ) : (
                    <video
                      key={asset.src}
                      autoPlay
                      loop
                      muted
                      playsInline
                      className="w-full h-full object-cover"
                    >
                      <source src={asset.src} type="video/mp4" />
                    </video>
                  )}
                  <div className="absolute inset-0 bg-black/0 group-hover:bg-black/20 transition-all"></div>
                </div>
              ))
            ) : (
              <div className="relative w-full aspect-[16/10] rounded-2xl overflow-hidden shadow-2xl transition-all duration-500 animate-in fade-in zoom-in-95">
                {currentVideo?.contentType === 'image' && currentVideo?.imageVTOData ? (
                  <ImageVTODisplay
                    key={currentVideo.imageVTOData.vto_result}
                    modelImage={currentVideo.imageVTOData.model_image}
                    garmentImages={currentVideo.imageVTOData.garment_images}
                    vtoResult={currentVideo.imageVTOData.vto_result}
                    description={currentVideo.imageVTOData.description}
                    isPlaying={isPlaying}
                  />
                ) : (
                  <video
                    ref={videoRef}
                    autoPlay
                    muted
                    playsInline
                    className="w-full h-full object-cover"
                    key={currentVideo?.src}
                    onEnded={handleVideoEnded}
                  >
                    <source src={currentVideo?.src || '/w_pose.mp4'} type="video/mp4" />
                  </video>
                )}
              </div>
            )}
          </div>
        </div>
      </main>

      {/* Control Bar */}
      <ControlBar
        channels={channels}
        selectedChannel={selectedChannel}
        isShuffleMode={isShuffleMode}
        onExitShuffle={handleExitShuffle}
        onChannelChange={handleChannelChange}
        isPlaying={isPlaying}
        onTogglePlay={togglePlay}
        onPrevVideo={prevVideo}
        onNextVideo={nextVideo}
        showPrompt={showPrompt}
        onTogglePrompt={() => setShowPrompt(!showPrompt)}
        isGridView={isGridView}
        onToggleGridView={() => setIsGridView(!isGridView)}
        currentVideo={currentVideo}
        currentPrompt={currentPrompt}
        onOpenInCreate={handleOpenInCreate}
        onNavigateToCategories={() => navigate('/genmedia-categories')}
      />
    </div>
  )
}

export default GenMediaTV
