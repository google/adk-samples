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

interface Capability {
  id: string
  title: string
  description: string
  icon: string
  image: string
  large: boolean
}

interface ByCapabilityViewProps {
  navigate: NavigateFunction
}

function ByCapabilityView({ navigate }: ByCapabilityViewProps) {
  const capabilities: Capability[] = [
    {
      id: 'image-vto',
      title: 'Image VTO',
      description: 'Virtual try-on for glasses, clothes, shoes, and accessories on static model images.\nHigh fidelity texture mapping and lighting adaptation.',
      icon: '👤',
      image: '/assets/ui/capabilities/image-vto.png',
      large: true
    },
    {
      id: 'video-vto',
      title: 'Video VTO',
      description: 'Dynamic virtual try-on for glasses and apparel with realistic fabric movement in video.',
      icon: '🎥',
      image: '/assets/ui/capabilities/video-vto.png',
      large: false
    },
    {
      id: 'product-360',
      title: '360 Spin',
      description: 'Product 360 rotation for shoes, cars, toys, home goods, and more.',
      icon: '🔄',
      image: '/assets/ui/capabilities/360-spin.png',
      large: false
    },
    {
      id: 'asset-tools',
      title: 'Catalogue Enrichment',
      description: 'Background changer and product placement—the building blocks for creative product photography.',
      icon: '✨',
      image: '/assets/ui/capabilities/asset-tools.png',
      large: false
    }
  ]

  return (
    <div className="grid grid-cols-2 gap-4">
      {/* Large Card - Image VTO */}
      <div onClick={() => navigate('/create/image-vto')} className="glass-panel rounded-lg overflow-hidden group cursor-pointer hover:shadow-level-3 transition-all duration-300 hover:-translate-y-1">
        <div className="h-[480px] bg-gradient-to-br from-white/10 to-white/5 relative flex items-end p-6">
          {/* Background Image Layer with Zoom Animation */}
          <div
            className="absolute inset-0 bg-cover bg-center transition-all duration-700 ease-out group-hover:scale-110 group-hover:blur-sm"
            style={{ backgroundImage: `url(${capabilities[0].image})` }}
          />
          {/* Gradient Overlay */}
          <div
            className="absolute inset-0 bg-gradient-to-t from-black/60 to-transparent transition-all duration-700 ease-out group-hover:scale-110"
          />
          {/* Icon */}
          <div className="absolute top-4 left-4 w-10 h-10 bg-white/10 rounded-lg flex items-center justify-center backdrop-blur-glass transition-all duration-300 group-hover:bg-gm-accent/20 group-hover:scale-110 z-10">
            <svg width="20" height="20" viewBox="0 0 20 20" fill="none" className="text-white">
              <rect x="2" y="3" width="16" height="14" rx="2" stroke="currentColor" strokeWidth="1.5"/>
              <circle cx="7" cy="7.5" r="1.5" fill="currentColor"/>
              <path d="M2 13L6 9L10 13L14 9L18 13" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
          </div>
          {/* Content */}
          <div className="transition-transform duration-300 group-hover:translate-x-1 relative z-10">
            <h3 className="text-h2 font-bold mb-2 text-white">{capabilities[0].title}</h3>
            <p className="text-body text-white/90 max-w-xs">
              {capabilities[0].description.split('\n')[0]}
            </p>
          </div>
        </div>
      </div>

      {/* Right Column - Three smaller cards */}
      <div className="flex flex-col gap-4">
        {/* Video VTO */}
        <div onClick={() => navigate('/create/video-vto')} className="flex-1 glass-panel rounded-lg overflow-hidden group cursor-pointer hover:shadow-level-3 transition-all duration-300 hover:-translate-y-1">
          <div className="h-full bg-gradient-to-br from-white/10 to-white/5 relative flex items-end p-4">
            <div
              className="absolute inset-0 bg-cover bg-center transition-all duration-700 ease-out group-hover:scale-110 group-hover:blur-sm"
              style={{ backgroundImage: `url(${capabilities[1].image})` }}
            />
            <div className="absolute inset-0 bg-gradient-to-t from-black/70 via-black/30 to-transparent" />
            <div className="absolute top-3 left-3 w-8 h-8 bg-white/10 rounded-lg flex items-center justify-center backdrop-blur-glass transition-all duration-300 group-hover:bg-gm-accent/20 group-hover:scale-110 z-10">
              <svg width="16" height="16" viewBox="0 0 20 20" fill="none" className="text-white">
                <circle cx="10" cy="10" r="7" stroke="currentColor" strokeWidth="1.5"/>
                <path d="M8 7L13 10L8 13V7Z" fill="currentColor"/>
              </svg>
            </div>
            <div className="transition-transform duration-300 group-hover:translate-x-1 relative z-10">
              <h3 className="text-base font-bold text-white">{capabilities[1].title}</h3>
              <p className="text-xs text-white/70 line-clamp-1">
                {capabilities[1].description}
              </p>
            </div>
          </div>
        </div>

        {/* 360 Spin */}
        <div onClick={() => navigate('/create/product-360')} className="flex-1 glass-panel rounded-lg overflow-hidden group cursor-pointer hover:shadow-level-3 transition-all duration-300 hover:-translate-y-1">
          <div className="h-full bg-gradient-to-br from-white/10 to-white/5 relative flex items-end p-4">
            <div
              className="absolute inset-0 bg-cover bg-center transition-all duration-700 ease-out group-hover:scale-110 group-hover:blur-sm"
              style={{ backgroundImage: `url(${capabilities[2].image})` }}
            />
            <div className="absolute inset-0 bg-gradient-to-t from-black/70 via-black/30 to-transparent" />
            <div className="absolute top-3 left-3 w-8 h-8 bg-white/10 rounded-lg flex items-center justify-center backdrop-blur-glass transition-all duration-300 group-hover:bg-gm-accent/20 group-hover:scale-110 z-10">
              <svg width="16" height="16" viewBox="0 0 20 20" fill="none" className="text-white">
                <circle cx="10" cy="10" r="7" stroke="currentColor" strokeWidth="1.5"/>
                <path d="M14 6L16 4M4 16L6 14" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
                <path d="M10 3V10L14 14" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
              </svg>
            </div>
            <div className="transition-transform duration-300 group-hover:translate-x-1 relative z-10">
              <h3 className="text-base font-bold text-white">{capabilities[2].title}</h3>
              <p className="text-xs text-white/70 line-clamp-1">
                {capabilities[2].description}
              </p>
            </div>
          </div>
        </div>

        {/* Asset Tools */}
        <div onClick={() => navigate('/create/asset-tools')} className="flex-1 glass-panel rounded-lg overflow-hidden group cursor-pointer hover:shadow-level-3 transition-all duration-300 hover:-translate-y-1">
          <div className="h-full bg-gradient-to-br from-white/10 to-white/5 relative flex items-end p-4">
            <div
              className="absolute inset-0 bg-cover bg-center transition-all duration-700 ease-out group-hover:scale-110 group-hover:blur-sm"
              style={{ backgroundImage: `url(${capabilities[3].image})` }}
            />
            <div className="absolute inset-0 bg-gradient-to-t from-black/70 via-black/30 to-transparent" />
            <div className="absolute top-3 left-3 w-8 h-8 bg-white/10 rounded-lg flex items-center justify-center backdrop-blur-glass transition-all duration-300 group-hover:bg-gm-accent/20 group-hover:scale-110 z-10">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" className="text-white">
                <path d="M12 2L14.5 9.5L22 12L14.5 14.5L12 22L9.5 14.5L2 12L9.5 9.5L12 2Z" fill="currentColor" opacity="0.6"/>
                <path d="M19 4L20 7L23 8L20 9L19 12L18 9L15 8L18 7L19 4Z" fill="currentColor" opacity="0.8"/>
                <path d="M7 14L8 17L11 18L8 19L7 22L6 19L3 18L6 17L7 14Z" fill="currentColor" opacity="0.5"/>
              </svg>
            </div>
            <div className="transition-transform duration-300 group-hover:translate-x-1 relative z-10">
              <h3 className="text-base font-bold text-white">{capabilities[3].title}</h3>
              <p className="text-xs text-white/70 line-clamp-1">
                {capabilities[3].description}
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default ByCapabilityView
