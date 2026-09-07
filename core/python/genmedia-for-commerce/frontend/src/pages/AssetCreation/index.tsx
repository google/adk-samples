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
import { useParams, useNavigate, useLocation } from 'react-router-dom'
import TopNav from '../../components/TopNav'
import {
  CAPABILITIES,
  getAvailableCapabilities,
  getAvailableProducts,
  isValidCombination,
  getDefaultCapability,
  getDefaultProduct
} from '../../config/featureConstraints'
import type { Product, Capability } from '../../config/featureConstraints'

// Import capability modules
import GlassesImageVTO from '../../image_vto/glasses/GlassesImageVTO'
import ClothesImageVTO from '../../image_vto/clothes/ClothesImageVTO'
import GlassesVideoVTO from '../../video_vto/glasses/GlassesVideoVTO'
import ClothesVideoVTO from '../../video_vto/clothes/ClothesVideoVTO'
import ShoesSpinning from '../../spinning/r2v/shoes/ShoesSpinning'
import OtherSpinning from '../../spinning/r2v/other/OtherSpinning'
import BackgroundChanger from '../../other/background_changer/BackgroundChanger'
import AssetTools from '../../other/AssetTools'
import FeedbackButton from '../../components/FeedbackButton'

interface LocationState {
  prefilledPrompt?: string
  prefilledProduct?: string
  prefilledSubMode?: 'animate-frame' | 'custom-scene'
  prefilledSpinMode?: 'r2v-standard' | 'interpolation'
  prefilledImage?: string
  prefilledImages?: string[]
  prefilledModelImage?: string
}

// Capability icons
const CapabilityIcon = ({ capability }: { capability: Capability }) => {
  switch (capability) {
    case 'image-vto':
      return (
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" className="opacity-70">
          <rect x="3" y="3" width="18" height="18" rx="2" stroke="currentColor" strokeWidth="2"/>
          <circle cx="8.5" cy="8.5" r="1.5" fill="currentColor"/>
          <path d="M21 15l-5-5L5 21" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
        </svg>
      )
    case 'video-vto':
      return (
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" className="opacity-70">
          <path d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14v-4z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
          <rect x="3" y="6" width="12" height="12" rx="2" stroke="currentColor" strokeWidth="2"/>
        </svg>
      )
    case 'product-360':
      return (
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" className="opacity-70">
          <path d="M23 4v6h-6" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"/>
          <path d="M1 20v-6h6" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"/>
          <path d="M3.51 9a9 9 0 0114.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0020.49 15" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"/>
        </svg>
      )
    case 'asset-tools':
      return (
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none">
          <path d="M12 2L14.5 9.5L22 12L14.5 14.5L12 22L9.5 14.5L2 12L9.5 9.5L12 2Z" fill="url(#gradient-main)" />
          <path d="M19 4L20 7L23 8L20 9L19 12L18 9L15 8L18 7L19 4Z" fill="url(#gradient-accent)" />
          <path d="M7 14L8 17L11 18L8 19L7 22L6 19L3 18L6 17L7 14Z" fill="url(#gradient-secondary)" />
          <defs>
            <linearGradient id="gradient-main" x1="12" y1="2" x2="12" y2="22" gradientUnits="userSpaceOnUse">
              <stop offset="0%" stopColor="#8AB4F8" />
              <stop offset="100%" stopColor="#6B9EF0" />
            </linearGradient>
            <linearGradient id="gradient-accent" x1="19" y1="4" x2="19" y2="12" gradientUnits="userSpaceOnUse">
              <stop offset="0%" stopColor="#A8C7FA" />
              <stop offset="100%" stopColor="#8AB4F8" />
            </linearGradient>
            <linearGradient id="gradient-secondary" x1="7" y1="14" x2="7" y2="22" gradientUnits="userSpaceOnUse">
              <stop offset="0%" stopColor="#6B9EF0" />
              <stop offset="100%" stopColor="#5A8DE8" />
            </linearGradient>
          </defs>
        </svg>
      )
    default:
      return null
  }
}

function AssetCreation() {
  const { capability: urlCapability } = useParams<{ capability: string }>()
  const navigate = useNavigate()
  const location = useLocation()
  const initialPrefilledData = (location.state as LocationState) || {}

  const [capability, setCapability] = useState<Capability>(
    (urlCapability as Capability) || CAPABILITIES.PRODUCT_360
  )
  // Set initial product based on URL capability to ensure valid combination
  const initialCapability = (urlCapability as Capability) || CAPABILITIES.PRODUCT_360
  const [product, setProduct] = useState<Product>(
    (initialPrefilledData.prefilledProduct as Product) || getDefaultProduct(initialCapability)
  )

  // Manage prefilled data in state so we can clear it when switching tabs
  const [prefilledData, setPrefilledData] = useState<LocationState>(initialPrefilledData)

  useEffect(() => {
    if (urlCapability) {
      const newCapability = urlCapability as Capability
      setCapability(newCapability)
      // Also update product if current product doesn't support the new capability
      const availableProducts = getAvailableProducts(newCapability)
      if (!availableProducts.includes(product)) {
        setProduct(availableProducts[0])
      }
    }
  }, [urlCapability])

  // Validate capability/product combination and auto-correct if needed
  useEffect(() => {
    if (!isValidCombination(product, capability)) {
      const newCapability = getDefaultCapability(product)
      setCapability(newCapability)
      navigate(`/create/${newCapability}`)
    }
  }, [product, capability, navigate])

  const handleCapabilityChange = (newCapability: Capability) => {
    setCapability(newCapability)
    navigate(`/create/${newCapability}`)

    // Always reset to default product when switching capabilities
    setProduct(getDefaultProduct(newCapability))

    // Clear prefilled data when switching tabs
    setPrefilledData({})
  }

  const handleProductChange = (newProduct: Product) => {
    setProduct(newProduct)

    // Auto-switch to first valid capability if current capability doesn't support this product
    const availableCapabilities = getAvailableCapabilities(newProduct)
    if (!availableCapabilities.includes(capability)) {
      const newCapability = availableCapabilities[0]
      setCapability(newCapability)
      navigate(`/create/${newCapability}`)
    }
  }

  // Render the appropriate capability component
  // Using key={`${capability}-${product}`} forces React to unmount/remount on switch, resetting all state
  const renderCapabilityContent = () => {
    const componentKey = `${capability}-${product}`

    if (capability === 'product-360') {
      if (product === 'shoes') {
        return (
          <ShoesSpinning
            key={componentKey}
            prefilledImages={prefilledData.prefilledImages}
            currentProduct={product}
            availableProducts={getAvailableProducts(capability)}
            onProductChange={handleProductChange}
          />
        )
      }
      // Use OtherSpinning for all other products (cars, smartphones, other, etc.)
      return (
        <OtherSpinning
          key={componentKey}
          prefilledImages={prefilledData.prefilledImages}
          prefilledSpinMode={prefilledData.prefilledSpinMode}
          currentProduct={product}
          availableProducts={getAvailableProducts(capability)}
          onProductChange={handleProductChange}
        />
      )
    }

    if (capability === 'image-vto') {
      if (product === 'glasses') {
        return (
          <GlassesImageVTO
            key={componentKey}
            uploadedImage={prefilledData.prefilledImage}
            prefilledModelImage={prefilledData.prefilledModelImage}
            prefilledGlassesImage={prefilledData.prefilledImages?.[0]}
            currentProduct={product}
            availableProducts={getAvailableProducts(capability)}
            onProductChange={handleProductChange}
          />
        )
      }
      if (product === 'clothes') {
        return (
          <ClothesImageVTO
            key={componentKey}
            uploadedImage={prefilledData.prefilledImage}
            prefilledModelImage={prefilledData.prefilledModelImage}
            prefilledGarmentImages={prefilledData.prefilledImages}
            currentProduct={product}
            availableProducts={getAvailableProducts(capability)}
            onProductChange={handleProductChange}
          />
        )
      }
    }

    if (capability === 'video-vto') {
      if (product === 'glasses') {
        return (
          <GlassesVideoVTO
            key={componentKey}
            prefilledImage={prefilledData.prefilledImage}
            prefilledPrompt={prefilledData.prefilledPrompt}
            prefilledSubMode={prefilledData.prefilledSubMode}
            currentProduct={product}
            availableProducts={getAvailableProducts(capability)}
            onProductChange={handleProductChange}
          />
        )
      }
      if (product === 'clothes') {
        return (
          <ClothesVideoVTO
            key={componentKey}
            prefilledImage={prefilledData.prefilledImage}
            prefilledPrompt={prefilledData.prefilledPrompt}
            prefilledSubMode={prefilledData.prefilledSubMode}
            prefilledGarmentImages={prefilledData.prefilledImages}
            currentProduct={product}
            availableProducts={getAvailableProducts(capability)}
            onProductChange={handleProductChange}
          />
        )
      }
    }

    if (capability === 'background-change') {
      return <BackgroundChanger key={componentKey} />
    }

    if (capability === 'asset-tools') {
      return <AssetTools key={componentKey} />
    }

    return null
  }

  return (
    <div className="min-h-screen bg-gm-bg-light dark:bg-gm-bg">
      <TopNav />

      <main className="px-6 lg:px-8 py-6 lg:py-8">
        <div className="max-w-[1440px] mx-auto">
          {/* Capability Selector - Centered */}
          <div className="flex justify-center items-center mb-8 animate-fade-in relative">
            {/* Capability Tabs */}
            <div className="glass-panel rounded-xl p-1.5 inline-flex gap-1">
              <button
                onClick={() => handleCapabilityChange(CAPABILITIES.PRODUCT_360)}
                className={`px-5 lg:px-6 py-2.5 rounded-lg text-button font-medium transition-all duration-300 flex items-center gap-2 ${
                  capability === 'product-360'
                    ? 'glass-toggle active'
                    : 'glass-toggle'
                }`}
              >
                <CapabilityIcon capability="product-360" />
                <span className="hidden sm:inline">Product 360</span>
                <span className="sm:hidden">360</span>
              </button>
              <button
                onClick={() => handleCapabilityChange(CAPABILITIES.IMAGE_VTO)}
                className={`px-5 lg:px-6 py-2.5 rounded-lg text-button font-medium transition-all duration-300 flex items-center gap-2 ${
                  capability === 'image-vto'
                    ? 'glass-toggle active'
                    : 'glass-toggle'
                }`}
              >
                <CapabilityIcon capability="image-vto" />
                <span className="hidden sm:inline">Image VTO</span>
                <span className="sm:hidden">Image</span>
              </button>
              <button
                onClick={() => handleCapabilityChange(CAPABILITIES.VIDEO_VTO)}
                className={`px-5 lg:px-6 py-2.5 rounded-lg text-button font-medium transition-all duration-300 flex items-center gap-2 ${
                  capability === 'video-vto'
                    ? 'glass-toggle active'
                    : 'glass-toggle'
                }`}
              >
                <CapabilityIcon capability="video-vto" />
                <span className="hidden sm:inline">Video VTO</span>
                <span className="sm:hidden">Video</span>
              </button>
              <button
                onClick={() => handleCapabilityChange(CAPABILITIES.ASSET_TOOLS)}
                className={`px-5 lg:px-6 py-2.5 rounded-lg text-button font-medium transition-all duration-300 flex items-center gap-2 ${
                  capability === 'asset-tools'
                    ? 'glass-toggle active'
                    : 'glass-toggle'
                }`}
              >
                <CapabilityIcon capability="asset-tools" />
                <span className="hidden sm:inline">Catalogue Enrichment</span>
                <span className="sm:hidden">AI</span>
              </button>
            </div>
          </div>

          {/* Main Content Area */}
          <div className="animate-fade-in delay-100">
            {renderCapabilityContent()}
          </div>
        </div>
      </main>

      <FeedbackButton capability={(() => {
        const base: Record<string, string> = {
          'product-360': 'Product 360',
          'image-vto': 'Image VTO',
          'video-vto': 'Video VTO',
          'asset-tools': 'Product Fitting',
          'background-change': 'Background Changer',
        }
        const cap = base[capability] || 'Other'
        if (capability === 'product-360' && product) {
          return `${cap} - ${product}`
        }
        return cap
      })()} />
    </div>
  )
}

export default AssetCreation
