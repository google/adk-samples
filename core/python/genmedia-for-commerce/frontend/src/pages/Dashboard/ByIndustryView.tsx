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

import { NavigateFunction } from 'react-router-dom'
import { ReactNode } from 'react'

interface IndustryCapability {
  id: string
  name: string
  description: string
  route: string
  icon: ReactNode
}

interface Industry {
  id: string
  title: string
  description: string
  icon: string
  image: string
  fallbackGradient: string
  capabilities: IndustryCapability[]
}

interface ByIndustryViewProps {
  navigate: NavigateFunction
  onIndustryClick: (industry: Industry) => void
}

function ByIndustryView({ navigate, onIndustryClick }: ByIndustryViewProps) {
  const industries: Industry[] = [
    {
      id: 'fashion',
      title: 'Fashion & Apparel',
      description: 'Image VTO and Video VTO for apparel, accessories, and footwear.',
      icon: '👗',
      image: '/assets/ui/industries/fashion.jpg',
      fallbackGradient: 'from-slate-800 to-slate-900',
      capabilities: [
        {
          id: 'image-vto',
          name: 'Image VTO',
          description: 'Virtual try-on for glasses, clothes, shoes, and accessories.',
          route: '/create/image-vto',
          icon: (
            <svg width="24" height="24" viewBox="0 0 20 20" fill="none" className="text-gm-accent">
              <rect x="2" y="3" width="16" height="14" rx="2" stroke="currentColor" strokeWidth="1.5"/>
              <circle cx="7" cy="7.5" r="1.5" fill="currentColor"/>
              <path d="M2 13L6 9L10 13L14 9L18 13" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
          )
        },
        {
          id: 'video-vto',
          name: 'Video VTO',
          description: 'Dynamic virtual try-on with realistic fabric movement.',
          route: '/create/video-vto',
          icon: (
            <svg width="24" height="24" viewBox="0 0 20 20" fill="none" className="text-gm-accent">
              <circle cx="10" cy="10" r="7" stroke="currentColor" strokeWidth="1.5"/>
              <path d="M8 7L13 10L8 13V7Z" fill="currentColor"/>
            </svg>
          )
        }
      ]
    },
    {
      id: 'automotive',
      title: 'Automotive',
      description: 'Product 360 for showroom-ready vehicle presentations.',
      icon: '🚗',
      image: '/assets/ui/industries/automotive.png',
      fallbackGradient: 'from-blue-900 to-slate-900',
      capabilities: [
        {
          id: 'product-360',
          name: 'Product 360',
          description: 'Full rotation sequences for vehicles.',
          route: '/create/product-360',
          icon: (
            <svg width="24" height="24" viewBox="0 0 20 20" fill="none" className="text-gm-accent">
              <circle cx="10" cy="10" r="7" stroke="currentColor" strokeWidth="1.5"/>
              <path d="M14 6L16 4M4 16L6 14" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
              <path d="M10 3V10L14 14" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
          )
        }
      ]
    },
    {
      id: 'home',
      title: 'Home & Appliances',
      description: 'Product 360 and background customization for home goods.',
      icon: '🏠',
      image: '/assets/ui/industries/home-appliances.png',
      fallbackGradient: 'from-orange-900 to-slate-900',
      capabilities: [
        {
          id: 'product-360',
          name: 'Product 360',
          description: 'Full rotation sequences for home products.',
          route: '/create/product-360',
          icon: (
            <svg width="24" height="24" viewBox="0 0 20 20" fill="none" className="text-gm-accent">
              <circle cx="10" cy="10" r="7" stroke="currentColor" strokeWidth="1.5"/>
              <path d="M14 6L16 4M4 16L6 14" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
              <path d="M10 3V10L14 14" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
          )
        },
        {
          id: 'background-change',
          name: 'Background Change',
          description: 'Customize and replace product backgrounds.',
          route: '/create/background-change',
          icon: (
            <svg width="24" height="24" viewBox="0 0 20 20" fill="none" className="text-gm-accent">
              <rect x="2" y="2" width="16" height="16" rx="2" stroke="currentColor" strokeWidth="1.5"/>
              <path d="M2 14L6 10L10 14L14 10L18 14" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
          )
        }
      ]
    },
    {
      id: 'toys',
      title: 'Toys',
      description: 'Product 360 and background customization for toy products.',
      icon: '🧸',
      image: '/assets/ui/industries/toys.png',
      fallbackGradient: 'from-rose-900 to-slate-900',
      capabilities: [
        {
          id: 'product-360',
          name: 'Product 360',
          description: 'Full rotation sequences for toys.',
          route: '/create/product-360',
          icon: (
            <svg width="24" height="24" viewBox="0 0 20 20" fill="none" className="text-gm-accent">
              <circle cx="10" cy="10" r="7" stroke="currentColor" strokeWidth="1.5"/>
              <path d="M14 6L16 4M4 16L6 14" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
              <path d="M10 3V10L14 14" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
          )
        },
        {
          id: 'background-change',
          name: 'Background Change',
          description: 'Customize and replace product backgrounds.',
          route: '/create/background-change',
          icon: (
            <svg width="24" height="24" viewBox="0 0 20 20" fill="none" className="text-gm-accent">
              <rect x="2" y="2" width="16" height="16" rx="2" stroke="currentColor" strokeWidth="1.5"/>
              <path d="M2 14L6 10L10 14L14 10L18 14" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
          )
        }
      ]
    }
  ]

  const handleIndustryClick = (industry: Industry) => {
    if (industry.capabilities.length === 1) {
      // Direct navigation for single capability
      const state: { prefilledProduct?: string } = {}

      // Set default product based on industry
      if (industry.id === 'automotive') {
        state.prefilledProduct = 'other'
      }

      navigate(industry.capabilities[0].route, { state })
    } else if (industry.id === 'fashion') {
      // Direct navigation to Image VTO with clothes for fashion
      navigate('/create/image-vto', { state: { prefilledProduct: 'clothes' } })
    } else {
      // Show modal for multiple capabilities
      onIndustryClick(industry)
    }
  }

  return (
    <div className="grid grid-cols-3 gap-4">
      {/* Fashion & Apparel - Tall */}
      <div onClick={() => handleIndustryClick(industries[0])} className="glass-panel rounded-lg overflow-hidden group cursor-pointer hover:shadow-level-3 transition-all duration-300 hover:-translate-y-1 row-span-2 h-[480px]">
        <div className={`h-full bg-gradient-to-br ${industries[0].fallbackGradient} relative flex items-end p-6`}>
          <div
            className="absolute inset-0 bg-cover bg-center transition-all duration-700 ease-out group-hover:scale-110 group-hover:blur-sm"
            style={{ backgroundImage: `url(${industries[0].image})` }}
          />
          <div className="absolute inset-0 bg-gradient-to-t from-black/60 to-transparent" />
          <div className="absolute top-4 left-4 w-10 h-10 bg-black/10 dark:bg-white/10 rounded-lg flex items-center justify-center backdrop-blur-glass transition-all duration-300 group-hover:bg-gm-accent/20 group-hover:scale-110 z-10">
            <svg width="20" height="20" viewBox="0 0 20 20" fill="none" className="text-gm-text-primary-light dark:text-gm-text-primary">
              <path d="M4 6L10 3L16 6" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
              <path d="M4 6V17H16V6" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
              <path d="M7 10H13" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
            </svg>
          </div>
          <div className="transition-transform duration-300 group-hover:translate-x-1 relative z-10">
            <h3 className="text-h3 font-bold mb-2 text-white">{industries[0].title}</h3>
            <p className="text-body text-white/90">{industries[0].description}</p>
          </div>
        </div>
      </div>

      {/* Automotive - Tall */}
      <div onClick={() => handleIndustryClick(industries[1])} className="glass-panel rounded-lg overflow-hidden group cursor-pointer hover:shadow-level-3 transition-all duration-300 hover:-translate-y-1 row-span-2 h-[480px]">
        <div className={`h-full bg-gradient-to-br ${industries[1].fallbackGradient} relative flex items-end p-6`}>
          <div
            className="absolute inset-0 bg-cover bg-center transition-all duration-700 ease-out group-hover:scale-110 group-hover:blur-sm"
            style={{ backgroundImage: `url(${industries[1].image})` }}
          />
          <div className="absolute inset-0 bg-gradient-to-t from-black/60 to-transparent" />
          <div className="absolute top-4 left-4 w-10 h-10 bg-black/10 dark:bg-white/10 rounded-lg flex items-center justify-center backdrop-blur-glass transition-all duration-300 group-hover:bg-gm-accent/20 group-hover:scale-110 z-10">
            <svg width="20" height="20" viewBox="0 0 20 20" fill="none" className="text-gm-text-primary-light dark:text-gm-text-primary">
              <rect x="2" y="8" width="16" height="8" rx="1" stroke="currentColor" strokeWidth="1.5"/>
              <circle cx="6" cy="16" r="2" stroke="currentColor" strokeWidth="1.5"/>
              <circle cx="14" cy="16" r="2" stroke="currentColor" strokeWidth="1.5"/>
              <path d="M5 8L7 4H13L15 8" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
          </div>
          <div className="transition-transform duration-300 group-hover:translate-x-1 relative z-10">
            <h3 className="text-h3 font-bold mb-2 text-white">{industries[1].title}</h3>
            <p className="text-body text-white/90">{industries[1].description}</p>
          </div>
        </div>
      </div>

      {/* Right column - Home & Appliances and Toys */}
      <div className="flex flex-col gap-4 row-span-2">
        {/* Home & Appliances */}
        <div className="flex-1 glass-panel rounded-lg overflow-hidden group">
          <div className={`h-full bg-gradient-to-br ${industries[2].fallbackGradient} relative flex items-end p-6`}>
            <div
              className="absolute inset-0 bg-cover bg-center transition-all duration-700 ease-out group-hover:scale-110 group-hover:blur-sm"
              style={{ backgroundImage: `url(${industries[2].image})` }}
            />
            <div className="absolute inset-0 bg-gradient-to-t from-black/60 to-transparent" />
            <div className="absolute top-4 left-4 w-10 h-10 bg-black/10 dark:bg-white/10 rounded-lg flex items-center justify-center backdrop-blur-glass transition-all duration-300 group-hover:bg-gm-accent/20 group-hover:scale-110 z-10">
              <svg width="20" height="20" viewBox="0 0 20 20" fill="none" className="text-gm-text-primary-light dark:text-gm-text-primary">
                <path d="M3 10L10 3L17 10V17H3V10Z" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
                <rect x="7" y="12" width="6" height="5" stroke="currentColor" strokeWidth="1.5"/>
              </svg>
            </div>
            <div className="transition-transform duration-300 group-hover:translate-x-1 relative z-10">
              <h3 className="text-h3 font-bold mb-2 text-white">{industries[2].title}</h3>
              <p className="text-body text-white/90 text-sm">{industries[2].description}</p>
            </div>
          </div>
        </div>

        {/* Toys */}
        <div className="flex-1 glass-panel rounded-lg overflow-hidden group">
          <div className={`h-full bg-gradient-to-br ${industries[3].fallbackGradient} relative flex items-end p-6`}>
            <div
              className="absolute inset-0 bg-cover bg-center transition-all duration-700 ease-out group-hover:scale-110 group-hover:blur-sm"
              style={{ backgroundImage: `url(${industries[3].image})` }}
            />
            <div className="absolute inset-0 bg-gradient-to-t from-black/60 to-transparent" />
            <div className="absolute top-4 left-4 w-10 h-10 bg-black/10 dark:bg-white/10 rounded-lg flex items-center justify-center backdrop-blur-glass transition-all duration-300 group-hover:bg-gm-accent/20 group-hover:scale-110 z-10">
              <svg width="20" height="20" viewBox="0 0 20 20" fill="none" className="text-gm-text-primary-light dark:text-gm-text-primary">
                <circle cx="10" cy="8" r="3" stroke="currentColor" strokeWidth="1.5"/>
                <path d="M6 11C6 11 6 16 10 16C14 16 14 11 14 11" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
                <circle cx="7.5" cy="7.5" r="0.5" fill="currentColor"/>
                <circle cx="12.5" cy="7.5" r="0.5" fill="currentColor"/>
              </svg>
            </div>
            <div className="transition-transform duration-300 group-hover:translate-x-1 relative z-10">
              <h3 className="text-h3 font-bold mb-2 text-white">{industries[3].title}</h3>
              <p className="text-body text-white/90 text-sm">{industries[3].description}</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export type { Industry }
export default ByIndustryView
