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
import type { JSX } from 'react'
import type { Product } from '../config/featureConstraints'

interface ProductSelectorProps {
  currentProduct: Product
  availableProducts: Product[]
  onProductChange: (product: Product) => void
  compact?: boolean
}

const PRODUCT_CONFIG: Record<Product, { label: string; icon: JSX.Element }> = {
  glasses: {
    label: 'Glasses',
    icon: (
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
        <circle cx="6" cy="12" r="4" stroke="currentColor" strokeWidth="1.5"/>
        <circle cx="18" cy="12" r="4" stroke="currentColor" strokeWidth="1.5"/>
        <path d="M10 12h4" stroke="currentColor" strokeWidth="1.5"/>
        <path d="M2 12h0M22 12h0" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
      </svg>
    )
  },
  clothes: {
    label: 'Clothes',
    icon: (
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
        <path d="M6.5 2L2 6l3 2v12a1 1 0 001 1h12a1 1 0 001-1V8l3-2-4.5-4" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
        <path d="M12 2c1.5 0 2.5 1.5 2.5 3s-1 3-2.5 3-2.5-1.5-2.5-3 1-3 2.5-3z" stroke="currentColor" strokeWidth="1.5"/>
      </svg>
    )
  },
  shoes: {
    label: 'Shoes',
    icon: (
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
        <path d="M3 17h18v2a2 2 0 01-2 2H5a2 2 0 01-2-2v-2z" stroke="currentColor" strokeWidth="1.5"/>
        <path d="M3 17l2-8a2 2 0 012-2h6l2 4h4a2 2 0 012 2v4" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
      </svg>
    )
  },
  cars: {
    label: 'Cars',
    icon: (
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
        <path d="M5 17h14v2a1 1 0 01-1 1H6a1 1 0 01-1-1v-2z" stroke="currentColor" strokeWidth="1.5"/>
        <path d="M5 17l1.5-6.5A2 2 0 018.44 9h7.12a2 2 0 011.94 1.5L19 17" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
        <circle cx="7.5" cy="17" r="1.5" stroke="currentColor" strokeWidth="1.5"/>
        <circle cx="16.5" cy="17" r="1.5" stroke="currentColor" strokeWidth="1.5"/>
        <path d="M8 9l1-4h6l1 4" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
      </svg>
    )
  },
  smartphones: {
    label: 'Smartphones',
    icon: (
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
        <rect x="5" y="2" width="14" height="20" rx="2" stroke="currentColor" strokeWidth="1.5"/>
        <path d="M12 18h.01" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
        <path d="M9 5h6" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
      </svg>
    )
  },
  other: {
    label: 'Other',
    icon: (
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
        <rect x="3" y="3" width="18" height="18" rx="2" stroke="currentColor" strokeWidth="1.5"/>
        <path d="M12 8v8M8 12h8" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
      </svg>
    )
  }
}

function ProductSelector({
  currentProduct,
  availableProducts,
  onProductChange,
  compact = false
}: ProductSelectorProps) {
  const [isOpen, setIsOpen] = useState(false)
  const dropdownRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setIsOpen(false)
      }
    }

    if (isOpen) {
      document.addEventListener('mousedown', handleClickOutside)
    }

    return () => {
      document.removeEventListener('mousedown', handleClickOutside)
    }
  }, [isOpen])

  // If only one product available, show as static badge
  if (availableProducts.length === 1) {
    return (
      <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-lg bg-black/[0.04] dark:bg-white/[0.06] border border-black/[0.06] dark:border-white/[0.08]">
        <span className="text-gm-text-tertiary-light dark:text-gm-text-tertiary opacity-60">
          {PRODUCT_CONFIG[currentProduct].icon}
        </span>
        <span className="text-caption font-medium text-gm-text-secondary-light dark:text-gm-text-secondary">
          {PRODUCT_CONFIG[currentProduct].label}
        </span>
      </div>
    )
  }

  return (
    <div className="relative inline-block" ref={dropdownRef}>
      <button
        onClick={() => setIsOpen(!isOpen)}
        className={`inline-flex items-center gap-2 ${
          compact ? 'px-2.5 py-1' : 'px-3 py-1.5'
        } rounded-lg bg-black/[0.04] dark:bg-white/[0.06] hover:bg-black/[0.06] dark:hover:bg-white/[0.08] border border-black/[0.06] dark:border-white/[0.08] transition-all duration-200 group`}
      >
        <span className="text-gm-text-tertiary-light dark:text-gm-text-tertiary group-hover:text-gm-text-secondary-light dark:group-hover:text-gm-text-secondary transition-colors opacity-60">
          {PRODUCT_CONFIG[currentProduct].icon}
        </span>
        <span className={`${compact ? 'text-[11px]' : 'text-caption'} font-medium text-gm-text-secondary-light dark:text-gm-text-secondary`}>
          {PRODUCT_CONFIG[currentProduct].label}
        </span>
        <svg
          width="10"
          height="10"
          viewBox="0 0 12 12"
          fill="none"
          className={`text-gm-text-tertiary-light dark:text-gm-text-tertiary transition-transform duration-200 ${
            isOpen ? 'rotate-180' : ''
          }`}
        >
          <path d="M3 4.5L6 7.5L9 4.5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
        </svg>
      </button>

      {isOpen && (
        <div
          className="absolute top-full left-0 mt-1.5 min-w-[160px] rounded-xl p-1.5 shadow-level-3 border backdrop-blur-xl z-20 animate-slide-down"
          style={{
            background: 'var(--dropdown-bg)',
            borderColor: 'var(--dropdown-border)'
          }}
        >
          {availableProducts.map((product) => (
            <button
              key={product}
              onClick={() => {
                onProductChange(product)
                setIsOpen(false)
              }}
              className={`w-full flex items-center gap-2.5 px-3 py-2 rounded-lg transition-all duration-150 ${
                product === currentProduct
                  ? 'bg-gm-accent/10 text-gm-accent'
                  : 'text-gm-text-primary-light dark:text-gm-text-primary hover:bg-black/[0.04] dark:hover:bg-white/[0.06]'
              }`}
            >
              <span className="opacity-70">
                {PRODUCT_CONFIG[product].icon}
              </span>
              <span className="text-button font-medium flex-1 text-left">
                {PRODUCT_CONFIG[product].label}
              </span>
              {product === currentProduct && (
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" className="text-gm-accent">
                  <path d="M20 6L9 17l-5-5" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                </svg>
              )}
            </button>
          ))}
        </div>
      )}
    </div>
  )
}

export default ProductSelector
