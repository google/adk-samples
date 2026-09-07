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

import { useEffect, useRef, useState, useCallback } from 'react'
import { useNavigate } from 'react-router-dom'

interface CarouselItem {
  id: number
  title: string
  thumbnail: string
  video?: string
  isClothes?: boolean
  isGlasses?: boolean
}

const items: CarouselItem[] = [
  { id: 1, title: 'SHOES 360', thumbnail: '/products/shoes/images/product_4/image_1.webp', video: '/products/shoes/videos/product_4_360.mp4' },
  { id: 2, title: 'GLASSES VTO', thumbnail: '/products/glasses/images/blue.png', video: '/products/glasses/videos/women/w_blue_glaze.mp4', isGlasses: true },
  { id: 3, title: 'CLOTHES VIDEO VTO', thumbnail: '/products/clothes/results/women/result_001.png', video: '/products/clothes/videos/women_vto_001.mp4', isClothes: true },
  { id: 4, title: 'CAR 360', thumbnail: '/products/cars/images/car-1.png', video: '/products/360-videos/360-car-spin.mp4' },
  { id: 5, title: 'CLOTHES IMAGE VTO', thumbnail: '/products/clothes/results/men/result_003.png', isClothes: true },
  { id: 6, title: 'HIKING BOOT 360', thumbnail: '/products/shoes/images/product_2/image_1.png', video: '/products/360-videos/shoes-360.mp4' },
  { id: 7, title: 'GLASSES VTO', thumbnail: '/products/glasses/images/brown.png', video: '/products/glasses/videos/men/m_walking2.mp4', isGlasses: true },
  { id: 8, title: 'CLOTHES VIDEO VTO', thumbnail: '/products/clothes/results/men/result_001.png', video: '/products/clothes/videos/men_vto_001.mp4', isClothes: true },
  { id: 9, title: 'SMARTPHONE 360', thumbnail: '/products/phones/images/phone-1.png', video: '/products/360-videos/accessories-360-2.mp4' },
  { id: 10, title: 'RUNNING SHOE 360', thumbnail: '/products/shoes/images/product_5/image_1.webp', video: '/products/shoes/videos/product_5_360.mp4' },
  { id: 11, title: 'AURA SHOES', thumbnail: '/products/shoes/images/aura/aura_1.png', video: '/products/shoes/videos/w_aura_shoes.mp4' },
  { id: 12, title: 'CLOTHES IMAGE VTO', thumbnail: '/products/clothes/results/women/result_002.png', isClothes: true },
  { id: 13, title: 'SUNGLASSES VTO', thumbnail: '/products/glasses/images/green.png', video: '/products/glasses/videos/women/w_pose.mp4', isGlasses: true },
  { id: 14, title: 'CLOTHES VIDEO VTO', thumbnail: '/products/clothes/results/men/result_002.png', video: '/products/clothes/videos/men_vto_002.mp4', isClothes: true },
  { id: 15, title: 'ACCESSORIES 360', thumbnail: '/products/phones/images/phone-1.png', video: '/products/360-videos/accessories-360.mp4' },
  { id: 16, title: 'GLASSES VTO', thumbnail: '/products/glasses/images/black.png', video: '/products/glasses/videos/men/m_turn.mp4', isGlasses: true },
  { id: 17, title: 'CLOTHES IMAGE VTO', thumbnail: '/products/clothes/results/women/result_003.png', isClothes: true },
  { id: 18, title: 'CLOTHES VIDEO VTO', thumbnail: '/products/clothes/results/women/result_002.png', video: '/products/clothes/videos/women_vto_002.mp4', isClothes: true },
  { id: 19, title: 'SHOE 360', thumbnail: '/products/shoes/images/product_3/image_1.png', video: '/products/360-videos/shoe-2-360.mp4' },
  { id: 20, title: 'CLOTHES VIDEO VTO', thumbnail: '/products/clothes/results/men/result_003.png', video: '/products/clothes/videos/men_vto_003.mp4', isClothes: true },
]

const GAP = 16
const VISIBLE_COUNT = 5

function CarouselCard({ item, cardWidth }: { item: CarouselItem; cardWidth: number }) {
  const videoRef = useRef<HTMLVideoElement>(null)
  const [isHovered, setIsHovered] = useState(false)
  const objectPos = item.isClothes ? 'object-top' : ''

  useEffect(() => {
    if (!videoRef.current || !item.video) return
    if (isHovered) {
      videoRef.current.play().catch(() => {})
    } else {
      videoRef.current.pause()
      videoRef.current.currentTime = 0
    }
  }, [isHovered, item.video])

  return (
    <div
      className="flex-shrink-0 group cursor-pointer"
      style={{ width: cardWidth }}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
    >
      <div className="aspect-[4/3] rounded-lg overflow-hidden mb-1.5 bg-gradient-to-br from-black/[0.04] to-black/[0.02] dark:from-white/10 dark:to-white/5 relative">
        {item.isGlasses && item.video ? (
          <video
            ref={videoRef}
            src={item.video}
            muted
            loop
            playsInline
            preload="metadata"
            className={`w-full h-full object-cover ${objectPos} transition-transform duration-500 group-hover:scale-105`}
          />
        ) : item.video && isHovered ? (
          <video
            ref={videoRef}
            src={item.video}
            muted
            loop
            playsInline
            className={`w-full h-full object-cover ${objectPos} transition-transform duration-500 group-hover:scale-105`}
          />
        ) : (
          <img
            src={item.thumbnail}
            alt={item.title}
            className={`w-full h-full object-cover ${objectPos} transition-transform duration-500 group-hover:scale-105`}
            loading="lazy"
          />
        )}
      </div>
      <h4 className="text-[9px] font-semibold uppercase tracking-wider text-gm-text-primary-light dark:text-white/90 transition-colors duration-200 group-hover:text-gm-accent dark:group-hover:text-gm-accent truncate">
        {item.title}
      </h4>
    </div>
  )
}

function MadeWithGenMedia() {
  const navigate = useNavigate()
  const containerRef = useRef<HTMLDivElement>(null)
  const scrollRef = useRef<HTMLDivElement>(null)
  const hoverRef = useRef<HTMLDivElement>(null)
  const isPausedRef = useRef(false)
  const animationRef = useRef<number | null>(null)
  const scrollPosRef = useRef(0)
  const [cardWidth, setCardWidth] = useState(130)

  const duplicatedItems = [...items, ...items]

  const measureCards = useCallback(() => {
    if (!containerRef.current) return
    const w = containerRef.current.offsetWidth
    setCardWidth(Math.floor((w - (VISIBLE_COUNT - 1) * GAP) / VISIBLE_COUNT))
  }, [])

  useEffect(() => {
    measureCards()
    window.addEventListener('resize', measureCards)
    return () => window.removeEventListener('resize', measureCards)
  }, [measureCards])

  // Attach pointer events on DOM directly — more reliable than React synthetic events
  useEffect(() => {
    const el = hoverRef.current
    if (!el) return
    const pause = () => { isPausedRef.current = true }
    const resume = () => { isPausedRef.current = false }
    el.addEventListener('pointerenter', pause)
    el.addEventListener('pointerleave', resume)
    return () => {
      el.removeEventListener('pointerenter', pause)
      el.removeEventListener('pointerleave', resume)
    }
  }, [])

  useEffect(() => {
    const container = scrollRef.current
    if (!container) return

    const speed = 0.25

    const animate = () => {
      if (!isPausedRef.current) {
        const singleSetWidth = container.scrollWidth / 2
        scrollPosRef.current += speed
        if (scrollPosRef.current >= singleSetWidth) {
          scrollPosRef.current -= singleSetWidth
        }
        container.scrollLeft = scrollPosRef.current
      }
      animationRef.current = requestAnimationFrame(animate)
    }

    animationRef.current = requestAnimationFrame(animate)
    return () => {
      if (animationRef.current) cancelAnimationFrame(animationRef.current)
    }
  }, [cardWidth])

  return (
    <div className="mt-6 animate-fade-in-up delay-300" ref={containerRef}>
      <div className="flex items-center justify-between mb-2.5">
        <h2 className="text-sm font-bold">
          Made with <span className="text-red-500">❤️</span> using Genmedia
        </h2>
        <button
          onClick={() => navigate('/genmedia-tv')}
          className="text-xs text-gm-text-secondary-light dark:text-white/90 hover:text-gm-accent dark:hover:text-gm-text-primary transition-all duration-200 flex items-center gap-1 group/btn"
        >
          View All
          <svg width="16" height="16" viewBox="0 0 16 16" fill="none" className="text-current transition-transform duration-200 group-hover/btn:translate-x-1">
            <path d="M6 4L10 8L6 12" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
          </svg>
        </button>
      </div>

      <div className="relative overflow-hidden" ref={hoverRef}>
        <div
          ref={scrollRef}
          className="flex overflow-x-hidden scrollbar-hide"
          style={{ gap: GAP }}
        >
          {duplicatedItems.map((item, index) => (
            <CarouselCard key={`${item.id}-${index}`} item={item} cardWidth={cardWidth} />
          ))}
        </div>
      </div>
    </div>
  )
}

export default MadeWithGenMedia
