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
import BackgroundChanger from './background_changer/BackgroundChanger'
import ProductPlacement from './product_placement/ProductPlacement'
import ProductFitting from '../product_enrichment/product_fitting/ProductFitting'

interface AssetToolsProps {
  prefilledSubMode?: 'background-changer' | 'product-placement' | 'product-fitting'
  showVideo?: boolean
}

function AssetTools({
  prefilledSubMode = 'product-fitting',
  showVideo = true
}: AssetToolsProps) {
  const [subMode, setSubMode] = useState<'background-changer' | 'product-placement' | 'product-fitting'>(prefilledSubMode)

  return (
    <div className="space-y-6">
      {/* Mode Toggle */}
      <div className="flex justify-center">
        <div className="flex items-center gap-1 p-1 rounded-xl bg-black/[0.03] dark:bg-white/[0.04] glass-panel inline-flex">
          <button
            onClick={() => setSubMode('background-changer')}
            className={`px-4 py-2.5 rounded-lg text-button font-medium transition-all duration-200 flex items-center gap-2 ${
              subMode === 'background-changer'
                ? 'bg-white dark:bg-white/10 shadow-sm text-gm-text-primary-light dark:text-gm-text-primary'
                : 'text-gm-text-tertiary-light dark:text-gm-text-tertiary hover:text-gm-text-primary-light dark:hover:text-gm-text-primary'
            }`}
          >
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" className="opacity-70">
              <rect x="2" y="2" width="20" height="20" rx="3" stroke="currentColor" strokeWidth="1.5" opacity="0.3"/>
              <path d="M2 14l4-4 4 4 6-6 6 6" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
              <circle cx="7" cy="7" r="1.5" fill="currentColor"/>
            </svg>
            Background Changer
          </button>
          <button
            onClick={() => setSubMode('product-fitting')}
            className={`px-4 py-2.5 rounded-lg text-button font-medium transition-all duration-200 flex items-center gap-2 ${
              subMode === 'product-fitting'
                ? 'bg-white dark:bg-white/10 shadow-sm text-gm-text-primary-light dark:text-gm-text-primary'
                : 'text-gm-text-tertiary-light dark:text-gm-text-tertiary hover:text-gm-text-primary-light dark:hover:text-gm-text-primary'
            }`}
          >
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" className="opacity-70">
              <path d="M20 21v-2a4 4 0 00-4-4H8a4 4 0 00-4 4v2" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
              <circle cx="12" cy="7" r="4" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
            Product Fitting
          </button>
          <button
            onClick={() => setSubMode('product-placement')}
            disabled
            className={`px-4 py-2.5 rounded-lg text-button font-medium transition-all duration-200 flex items-center gap-2 relative ${
              subMode === 'product-placement'
                ? 'bg-white dark:bg-white/10 shadow-sm text-gm-text-primary-light dark:text-gm-text-primary'
                : 'text-gm-text-tertiary-light dark:text-gm-text-tertiary hover:text-gm-text-primary-light dark:hover:text-gm-text-primary opacity-50 cursor-not-allowed'
            }`}
          >
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" className="opacity-70">
              <rect x="2" y="2" width="20" height="20" rx="3" stroke="currentColor" strokeWidth="1.5" opacity="0.3"/>
              <rect x="8" y="8" width="8" height="8" rx="1.5" fill="currentColor" opacity="0.5"/>
              <path d="M12 8V3M12 21v-5M16 12h5M3 12h5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
            </svg>
            Product Placement
            <div className="absolute -top-1 -right-1">
              <div className="bg-gradient-to-r from-gm-accent to-blue-500 text-white px-1.5 py-0.5 rounded-full text-[9px] font-bold tracking-wide shadow-md">
                SOON
              </div>
            </div>
          </button>
        </div>
      </div>

      {/* Content based on mode */}
      <div className="animate-fade-in">
        {subMode === 'background-changer' ? (
          <BackgroundChanger showVideo={showVideo} />
        ) : subMode === 'product-fitting' ? (
          <ProductFitting showVideo={showVideo} />
        ) : (
          <ProductPlacement showVideo={showVideo} />
        )}
      </div>
    </div>
  )
}

export default AssetTools
