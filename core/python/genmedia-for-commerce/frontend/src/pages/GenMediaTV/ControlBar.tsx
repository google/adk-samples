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

import { useRef, useEffect, useState } from 'react'
import Tooltip from './Tooltip'

interface Channel {
  id: string
  name: string
  thumbnail: string
}

interface VideoAsset {
  id: number
  src: string
  prompt: string
  productImg: string
  productImages?: string[]
  aspectRatio: string
  contentType?: 'video' | 'image'
}

interface ControlBarProps {
  channels: Channel[]
  selectedChannel: string
  isShuffleMode?: boolean
  onExitShuffle?: () => void
  onChannelChange: (direction: 'prev' | 'next') => void
  isPlaying: boolean
  onTogglePlay: () => void
  onPrevVideo: () => void
  onNextVideo: () => void
  showPrompt: boolean
  onTogglePrompt: () => void
  isGridView: boolean
  onToggleGridView: () => void
  currentVideo?: VideoAsset
  currentPrompt: string
  onOpenInCreate: () => void
  onNavigateToCategories: () => void
}

function ControlBar({
  channels,
  selectedChannel,
  isShuffleMode = false,
  onExitShuffle,
  onChannelChange,
  isPlaying,
  onTogglePlay,
  onPrevVideo,
  onNextVideo,
  showPrompt,
  onTogglePrompt,
  isGridView,
  onToggleGridView,
  currentVideo,
  currentPrompt,
  onOpenInCreate,
  onNavigateToCategories
}: ControlBarProps) {
  const channelNameRef = useRef<HTMLDivElement>(null)
  const [shouldScroll, setShouldScroll] = useState(false)

  // Detect if channel name text overflows and needs scrolling
  useEffect(() => {
    const checkOverflow = () => {
      if (channelNameRef.current) {
        const element = channelNameRef.current
        const isOverflowing = element.scrollWidth > element.clientWidth
        setShouldScroll(isOverflowing)
      }
    }

    checkOverflow()
    // Recheck on channel change
    const timer = setTimeout(checkOverflow, 100)
    return () => clearTimeout(timer)
  }, [selectedChannel, channels])

  return (
    <>
    <div className="fixed bottom-8 left-1/2 -translate-x-1/2 glass-panel rounded-2xl p-4 flex items-center gap-4">
      {/* Screen Mode */}
      <Tooltip text="Browse Categories" position="top">
        <button
          onClick={onNavigateToCategories}
          className="w-12 h-12 rounded-xl bg-black/5 dark:bg-white/10 hover:bg-black/10 dark:hover:bg-white/20 flex items-center justify-center transition-all"
        >
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" className="text-gm-text-primary-light dark:text-gm-text-primary">
            <rect x="2" y="4" width="20" height="14" rx="2" stroke="currentColor" strokeWidth="2"/>
            <path d="M8 21H16" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
            <path d="M12 18V21" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
          </svg>
        </button>
      </Tooltip>

      <div className="h-8 w-px bg-black/10 dark:bg-white/10"></div>

      {/* Playback Controls */}
      <div className="flex items-center gap-2">
        <Tooltip text="Previous Video" position="top">
          <button
            onClick={onPrevVideo}
            className="w-10 h-10 rounded-lg hover:bg-black/5 dark:hover:bg-white/10 flex items-center justify-center transition-all"
          >
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" className="text-gm-text-primary-light dark:text-gm-text-primary">
              <path d="M19 20L9 12L19 4V20Z" fill="currentColor"/>
              <path d="M5 4V20" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
            </svg>
          </button>
        </Tooltip>

        <Tooltip text={isPlaying ? "Pause" : "Play"} position="top">
          <button
            onClick={onTogglePlay}
            className="w-10 h-10 rounded-lg hover:bg-black/5 dark:hover:bg-white/10 flex items-center justify-center transition-all"
          >
            {isPlaying ? (
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" className="text-gm-text-primary-light dark:text-gm-text-primary">
                <rect x="6" y="4" width="4" height="16" rx="1" fill="currentColor"/>
                <rect x="14" y="4" width="4" height="16" rx="1" fill="currentColor"/>
              </svg>
            ) : (
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" className="text-gm-text-primary-light dark:text-gm-text-primary">
                <path d="M5 4L19 12L5 20V4Z" fill="currentColor"/>
              </svg>
            )}
          </button>
        </Tooltip>

        <Tooltip text="Next Video" position="top">
          <button
            onClick={onNextVideo}
            className="w-10 h-10 rounded-lg hover:bg-black/5 dark:hover:bg-white/10 flex items-center justify-center transition-all"
          >
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" className="text-gm-text-primary-light dark:text-gm-text-primary">
              <path d="M5 4L15 12L5 20V4Z" fill="currentColor"/>
              <path d="M19 4V20" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
            </svg>
          </button>
        </Tooltip>
      </div>

      <div className="h-8 w-px bg-black/10 dark:bg-white/10"></div>

      {/* Hide Prompt Toggle */}
      <div className="flex items-center gap-2">
        <span className="text-xs text-gm-text-secondary-light dark:text-gm-text-secondary">Hide Prompt</span>
        <button
          onClick={onTogglePrompt}
          className={`w-12 h-6 rounded-full relative transition-all ${
            showPrompt ? 'bg-white/20' : 'bg-gm-accent'
          }`}
        >
          <div
            className={`absolute top-1 w-4 h-4 rounded-full bg-white transition-all ${
              showPrompt ? 'left-1' : 'right-1'
            }`}
          ></div>
        </button>
      </div>

      <div className="h-8 w-px bg-black/10 dark:bg-white/10"></div>

      {/* Additional Controls */}
      <div className="flex items-center gap-1">
        <Tooltip text="Fullscreen" position="top">
          <button className="w-10 h-10 rounded-lg hover:bg-black/5 dark:hover:bg-white/10 flex items-center justify-center transition-all">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" className="text-gm-text-primary-light dark:text-gm-text-primary">
              <path d="M8 3H5C3.89543 3 3 3.89543 3 5V8" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
              <path d="M16 3H19C20.1046 3 21 3.89543 21 5V8" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
              <path d="M8 21H5C3.89543 21 3 20.1046 3 19V16" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
              <path d="M16 21H19C20.1046 21 21 20.1046 21 19V16" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
            </svg>
          </button>
        </Tooltip>
      </div>

      <div className="h-8 w-px bg-black/10 dark:bg-white/10"></div>

      {/* Prompt Section */}
      {showPrompt && (
        <div className="flex items-center gap-4 max-w-xs">
          <div className="flex -space-x-3">
            <div className="w-16 h-16 rounded-lg overflow-hidden border-2 border-gm-bg-light dark:border-gm-bg shadow-md">
              <img
                src={currentVideo?.productImages && currentVideo.productImages.length > 1
                  ? currentVideo.productImages[1]
                  : currentVideo?.productImg}
                alt="Product"
                className="w-full h-full object-cover"
              />
            </div>
            <div className="w-16 h-16 rounded-lg overflow-hidden border-2 border-gm-bg-light dark:border-gm-bg shadow-md">
              {currentVideo?.productImages && currentVideo.productImages.length > 2 ? (
                <img
                  src={currentVideo.productImages[2]}
                  alt="Product angle 2"
                  className="w-full h-full object-cover"
                />
              ) : currentVideo?.contentType === 'image' ? (
                <img
                  src={currentVideo?.src}
                  alt="Result"
                  className="w-full h-full object-cover"
                />
              ) : (
                <video
                  className="w-full h-full object-cover"
                  muted
                  playsInline
                  key={`thumb-${currentVideo?.src}`}
                >
                  <source src={currentVideo?.src} type="video/mp4" />
                </video>
              )}
            </div>
          </div>
          <div className="flex-1 overflow-hidden">
            <p className="text-xs text-gm-text-secondary-light dark:text-gm-text-secondary font-medium mb-1">Prompt</p>
            <div className="h-10 overflow-hidden relative">
              <div className="broadcast-ticker-vertical">
                <div className="text-xs text-gm-text-primary-light dark:text-gm-text-primary leading-relaxed pb-2">
                  {currentPrompt}
                </div>
                {/* Duplicate text for seamless loop */}
                <div className="text-xs text-gm-text-primary-light dark:text-gm-text-primary leading-relaxed py-2 opacity-30">•••</div>
                <div className="text-xs text-gm-text-primary-light dark:text-gm-text-primary leading-relaxed">
                  {currentPrompt}
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      <div className="h-8 w-px bg-black/10 dark:bg-white/10"></div>

      {/* Action Buttons */}
      <Tooltip text="Open in Create" position="top">
        <button
          onClick={onOpenInCreate}
          className="w-10 h-10 rounded-lg bg-black/5 dark:bg-white/10 hover:bg-black/10 dark:hover:bg-white/20 flex items-center justify-center transition-all"
        >
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" className="text-gm-text-primary-light dark:text-gm-text-primary">
            <path d="M18 13V19C18 19.5304 17.7893 20.0391 17.4142 20.4142C17.0391 20.7893 16.5304 21 16 21H5C4.46957 21 3.96086 20.7893 3.58579 20.4142C3.21071 20.0391 3 19.5304 3 19V8C3 7.46957 3.21071 6.96086 3.58579 6.58579C3.96086 6.21071 4.46957 6 5 6H11" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
            <path d="M15 3H21V9" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
            <path d="M10 14L21 3" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
          </svg>
        </button>
      </Tooltip>

      <Tooltip text={isGridView ? "Exit Grid View" : "Grid View"} position="top">
        <button
          onClick={onToggleGridView}
          className={`w-10 h-10 rounded-lg flex items-center justify-center transition-all ${
            isGridView ? 'bg-gm-accent text-white' : 'bg-black/5 dark:bg-white/10 hover:bg-black/10 dark:hover:bg-white/20'
          }`}
        >
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" className={isGridView ? 'text-white' : 'text-gm-text-primary-light dark:text-gm-text-primary'}>
            <rect x="3" y="3" width="7" height="7" stroke="currentColor" strokeWidth="2"/>
            <rect x="14" y="3" width="7" height="7" stroke="currentColor" strokeWidth="2"/>
            <rect x="3" y="14" width="7" height="7" stroke="currentColor" strokeWidth="2"/>
            <rect x="14" y="14" width="7" height="7" stroke="currentColor" strokeWidth="2"/>
          </svg>
        </button>
      </Tooltip>

      <div className="h-8 w-px bg-black/10 dark:bg-white/10"></div>

      {/* Channel Selector */}
      <div className="flex items-center gap-3 bg-black/5 dark:bg-white/5 rounded-xl p-2 relative">
        {/* Shuffle indicator badge */}
        {isShuffleMode && onExitShuffle && (
          <Tooltip text="Click to Exit Shuffle Mode" position="top">
            <button
              onClick={onExitShuffle}
              className="absolute -top-2 -right-2 bg-gm-accent hover:bg-gm-accent/80 rounded-full p-1.5 shadow-lg z-10 animate-pulse hover:animate-none transition-all group"
            >
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" className="text-white">
                <path d="M16 3H21V8" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                <path d="M4 20L21 3" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                <path d="M21 16V21H16" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                <path d="M15 15L21 21" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                <path d="M4 4L9 9" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
              </svg>
            </button>
          </Tooltip>
        )}
        <p className="text-[10px] text-gm-text-secondary-light dark:text-gm-text-secondary font-medium px-2">CHANNEL</p>
        <div className="flex items-center gap-2">
          <div className="w-14 h-14 rounded-full overflow-hidden bg-gradient-to-br from-amber-50 via-white to-emerald-50 dark:from-gray-700 dark:via-gray-800 dark:to-gray-700 flex items-center justify-center shadow-md border border-black/5 dark:border-white/10">
            {(() => {
              const currentChannel = channels.find(c => c.id === selectedChannel)
              const thumbnail = currentChannel?.thumbnail || ''
              const isImage = thumbnail.match(/\.(webp|png|jpg|jpeg)$/i)

              return isImage ? (
                <img
                  src={thumbnail}
                  alt={currentChannel?.name}
                  className="w-full h-full object-cover scale-100"
                  style={{ objectPosition: 'center top' }}
                />
              ) : (
                <video
                  className="w-full h-full object-cover"
                  muted
                  autoPlay
                  loop
                  playsInline
                  key={selectedChannel}
                >
                  <source src={thumbnail} type="video/mp4" />
                </video>
              )
            })()}
          </div>
          <div className="flex flex-col items-center">
            <Tooltip text="Previous Channel" position="top">
              <button
                onClick={() => onChannelChange('prev')}
                className="hover:bg-black/5 dark:hover:bg-white/10 rounded p-0.5 transition-all"
              >
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" className="text-gm-text-secondary-light dark:text-gm-text-secondary">
                  <path d="M18 15L12 9L6 15" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                </svg>
              </button>
            </Tooltip>
            {/* Broadcast Ticker for Channel Name */}
            <div className="w-24 overflow-hidden relative">
              <div
                ref={channelNameRef}
                className={`flex whitespace-nowrap ${shouldScroll ? 'broadcast-ticker-animate' : 'justify-center'}`}
              >
                <span className="text-xs font-medium text-gm-text-primary-light dark:text-gm-text-primary tracking-wide">
                  {channels.find(c => c.id === selectedChannel)?.name || 'GLASSES VTO'}
                </span>
                {/* Duplicate text for seamless loop when scrolling */}
                {shouldScroll && (
                  <>
                    <span className="text-xs font-medium text-gm-text-primary-light dark:text-gm-text-primary tracking-wide px-4 opacity-30">•</span>
                    <span className="text-xs font-medium text-gm-text-primary-light dark:text-gm-text-primary tracking-wide">
                      {channels.find(c => c.id === selectedChannel)?.name || 'GLASSES VTO'}
                    </span>
                  </>
                )}
              </div>
            </div>
            <Tooltip text="Next Channel" position="top">
              <button
                onClick={() => onChannelChange('next')}
                className="hover:bg-black/5 dark:hover:bg-white/10 rounded p-0.5 transition-all"
              >
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" className="text-gm-text-secondary-light dark:text-gm-text-secondary">
                  <path d="M6 9L12 15L18 9" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                </svg>
              </button>
            </Tooltip>
          </div>
        </div>
      </div>
    </div>

    {/* Broadcast Ticker Animation Styles */}
    <style>{`
      @keyframes broadcast-ticker {
        0% {
          transform: translateX(0);
        }
        100% {
          transform: translateX(-50%);
        }
      }

      .broadcast-ticker-animate {
        animation: broadcast-ticker 60s linear infinite;
        will-change: transform;
      }

      /* Slower animation for prompts (longer text) */
      .broadcast-ticker-animate-slow {
        animation: broadcast-ticker 60s linear infinite;
        will-change: transform;
      }

      /* Vertical scrolling animation for prompts */
      @keyframes broadcast-ticker-vertical {
        0% {
          transform: translateY(0);
        }
        100% {
          transform: translateY(-50%);
        }
      }

      .broadcast-ticker-vertical {
        animation: broadcast-ticker-vertical 60s linear infinite;
        will-change: transform;
      }

      /* Pause animation on hover for readability */
      .broadcast-ticker-animate:hover,
      .broadcast-ticker-animate-slow:hover,
      .broadcast-ticker-vertical:hover {
        animation-play-state: paused;
      }

      /* Smooth fade at edges for polished broadcast look */
      .broadcast-ticker-animate::before,
      .broadcast-ticker-animate::after {
        content: '';
        position: absolute;
        top: 0;
        bottom: 0;
        width: 12px;
        pointer-events: none;
        z-index: 1;
      }
    `}</style>
  </>
  )
}

export default ControlBar
