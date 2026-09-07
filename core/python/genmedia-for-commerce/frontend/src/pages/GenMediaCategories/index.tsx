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
import { useNavigate } from 'react-router-dom'

interface Category {
  id: string
  name: string
  image?: string
  video?: string
  isSpecial?: boolean
}

function GenMediaCategories() {
  const navigate = useNavigate()
  const [isDarkMode, setIsDarkMode] = useState(() => {
    const saved = localStorage.getItem('darkMode')
    return saved !== null ? JSON.parse(saved) : true
  })

  useEffect(() => {
    localStorage.setItem('darkMode', JSON.stringify(isDarkMode))
    if (isDarkMode) {
      document.documentElement.classList.add('dark')
    } else {
      document.documentElement.classList.remove('dark')
    }
  }, [isDarkMode])

  const categories: Category[] = [
    {
      id: 'shuffle-all',
      name: 'SHUFFLE ALL',
      video: '/assets/ui/categories/loop_shuffle.mp4',
      isSpecial: true
    },
    {
      id: 'accessories-vto',
      name: 'ACCESSORIES VTO',
      image: '/assets/ui/categories/watch-color.png'
    },
    {
      id: 'glasses-vto',
      name: 'GLASSES VTO',
      image: '/assets/ui/categories/glasses-color.png'
    },
    {
      id: 'bags-vto',
      name: 'BAGS VTO',
      image: '/assets/ui/categories/bag-color.png'
    },
    {
      id: 'clothes-vto',
      name: 'CLOTHES VTO',
      image: '/assets/ui/categories/shirt-color.png'
    },
    {
      id: 'shoes-360',
      name: 'SHOES 360',
      image: '/assets/ui/categories/shoe-color.png'
    },
    {
      id: 'clothes-360',
      name: 'CLOTHES 360',
      image: '/assets/ui/categories/shirt-color.png'
    },
    {
      id: 'accessories-360',
      name: 'ACCESSORIES 360',
      image: '/assets/ui/categories/watch-color.png'
    },
    {
      id: 'cars-360',
      name: 'CARS 360',
      image: '/assets/ui/categories/car-360-color.png'
    },
    {
      id: 'glasses-360',
      name: 'GLASSES 360',
      image: '/assets/ui/categories/glasses-360-color.png'
    }
  ]

  const isClickable = (categoryId: string) => {
    const clickableCategories = ['shuffle-all', 'glasses-vto', 'clothes-vto', 'shoes-360', 'clothes-360', 'accessories-360', 'cars-360', 'glasses-360']
    return clickableCategories.includes(categoryId)
  }

  const handleCategoryClick = (categoryId: string) => {
    if (!isClickable(categoryId)) return

    // Navigate to GenMedia TV - shuffle will show all content
    if (categoryId === 'shuffle-all') {
      navigate('/genmedia-tv', { state: { shuffle: true } })
    } else {
      // Map category ID to GenMedia TV channel ID
      const channelId = categoryId === 'clothes-vto' ? 'clothes-image-vto' : categoryId
      navigate('/genmedia-tv', { state: { selectedCategory: channelId } })
    }
  }

  return (
    <div className="min-h-screen bg-gm-bg-light dark:bg-gm-bg flex flex-col">
      {/* Header */}
      <header className="px-8 py-4 flex items-center justify-between">
        <div className="flex items-center gap-4">
          {/* Back Button */}
          <button
            onClick={() => navigate('/genmedia-tv')}
            className="w-10 h-10 rounded-full bg-black/5 dark:bg-white/10 hover:bg-black/10 dark:hover:bg-white/20 flex items-center justify-center transition-all"
          >
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" className="text-gm-text-primary-light dark:text-gm-text-primary">
              <path d="M19 12H5M5 12L12 19M5 12L12 5" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
          </button>
          <h1 className="text-xl font-bold">GenMedia TV</h1>
        </div>

        {/* Search Bar */}
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

      {/* Main Content - Category Grid */}
      <main className="flex-1 px-8 py-8">
        <div className="max-w-[1400px] mx-auto">
          <div className="grid grid-cols-4 gap-6">
            {categories.map((category) => (
              <button
                key={category.id}
                onClick={() => handleCategoryClick(category.id)}
                disabled={!isClickable(category.id)}
                style={{
                  boxShadow: isDarkMode
                    ? undefined
                    : isClickable(category.id)
                      ? '0 2px 8px rgba(0, 0, 0, 0.04), 0 8px 24px rgba(0, 0, 0, 0.06), inset 0 1px 0 rgba(255, 255, 255, 0.8)'
                      : '0 2px 8px rgba(0, 0, 0, 0.04)'
                }}
                className={`relative rounded-2xl overflow-hidden transition-all duration-300 group ${
                  isClickable(category.id) ? 'cursor-pointer' : 'cursor-default opacity-50'
                } ${
                  category.isSpecial
                    ? `col-span-2 row-span-1 aspect-[8/3]
                       bg-black
                       dark:from-black/80 dark:via-black/70 dark:to-black/80 dark:bg-gradient-to-br
                       border border-gray-800
                       dark:border-white/10
                       ${isClickable(category.id) ? 'hover:shadow-2xl hover:-translate-y-1 hover:border-gm-accent/50 dark:hover:border-gm-accent/50' : ''}`
                    : `aspect-[4/3]
                       bg-gradient-to-br from-white via-gray-50 to-white
                       dark:from-black/80 dark:via-black/70 dark:to-black/80
                       border border-gray-200/80
                       dark:border-white/10
                       ${isClickable(category.id) ? 'hover:shadow-2xl hover:-translate-y-1 hover:border-gm-accent/30 dark:hover:border-gm-accent/50' : ''}`
                }`}
              >
                {/* Inner gradient border effect for light mode (not on shuffle all) */}
                {!category.isSpecial && (
                  <div className="absolute inset-0 rounded-2xl dark:hidden"
                       style={{
                         background: 'linear-gradient(135deg, rgba(138, 180, 248, 0.03) 0%, transparent 50%, rgba(138, 180, 248, 0.03) 100%)',
                         pointerEvents: 'none'
                       }}
                  />
                )}

                {/* Category Image/Video */}
                <div className={`relative w-full h-full flex items-center justify-center ${
                  category.isSpecial ? 'p-4' : 'p-8'
                }`}
                style={{
                  filter: isDarkMode ? undefined : 'drop-shadow(0 4px 12px rgba(0, 0, 0, 0.08))'
                }}>
                  {category.video ? (
                    <video
                      autoPlay
                      loop
                      muted
                      playsInline
                      className="max-w-full max-h-full object-contain transition-all duration-500 group-hover:scale-110"
                      style={{
                        filter: category.isSpecial && !isDarkMode ? 'brightness(0.95) contrast(1.05)' : undefined
                      }}
                    >
                      <source src={category.video} type="video/mp4" />
                    </video>
                  ) : (
                    <img
                      src={category.image}
                      alt={category.name}
                      className={`max-w-full max-h-full object-contain transition-all duration-500 ${
                        isClickable(category.id) ? 'group-hover:scale-110' : ''
                      }`}
                      style={{
                        filter: !isDarkMode ? 'brightness(0.95) contrast(1.05) saturate(1.1)' : undefined
                      }}
                      onError={(e) => {
                        (e.target as HTMLImageElement).style.display = 'none'
                      }}
                    />
                  )}
                </div>

                {/* Category Label - Refined for light mode */}
                <div className={`absolute bottom-0 left-0 right-0 p-4 ${
                  category.isSpecial
                    ? 'bg-transparent'
                    : 'bg-gradient-to-t from-white/95 via-white/80 to-transparent dark:from-black/80 dark:to-transparent'
                }`}>
                  <p className={`font-semibold text-center tracking-wide transition-colors ${
                    category.isSpecial ? 'text-lg' : 'text-sm'
                  } ${
                    category.isSpecial
                      ? 'text-white'
                      : isDarkMode
                        ? 'text-white'
                        : 'text-gray-800 group-hover:text-gm-accent'
                  }`}>
                    {category.name}
                  </p>
                </div>

                {/* Hover Overlay - Subtle glow effect */}
                {isClickable(category.id) && (
                  <div
                    className="absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity duration-300 pointer-events-none"
                    style={{
                      background: isDarkMode
                        ? 'radial-gradient(circle at 50% 50%, rgba(138, 180, 248, 0.1) 0%, transparent 70%)'
                        : 'radial-gradient(circle at 50% 50%, rgba(138, 180, 248, 0.08) 0%, transparent 70%)'
                    }}
                  />
                )}
              </button>
            ))}
          </div>
        </div>
      </main>
    </div>
  )
}

export default GenMediaCategories
