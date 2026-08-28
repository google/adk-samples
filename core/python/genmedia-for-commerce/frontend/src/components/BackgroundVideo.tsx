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

import { useEffect, useRef } from 'react'

// Global variable to preserve video playback position across all components
let globalVideoTime = 0

interface BackgroundVideoProps {
  className?: string
  opacity?: string
}

function BackgroundVideo({ className = '', opacity = '' }: BackgroundVideoProps) {
  const videoRef = useRef<HTMLVideoElement>(null)

  useEffect(() => {
    const video = videoRef.current
    if (!video) return

    // Restore saved position
    if (globalVideoTime > 0) {
      video.currentTime = globalVideoTime
    }

    // Save position periodically
    const saveTime = () => {
      if (video && !video.paused) {
        globalVideoTime = video.currentTime
      }
    }

    const interval = setInterval(saveTime, 250)

    // Also save on time update for more accuracy
    video.addEventListener('timeupdate', saveTime)

    return () => {
      clearInterval(interval)
      video.removeEventListener('timeupdate', saveTime)
      // Save final position
      if (video) {
        globalVideoTime = video.currentTime
      }
    }
  }, [])

  return (
    <video
      ref={videoRef}
      autoPlay
      loop
      muted
      playsInline
      className={`absolute inset-0 w-full h-full object-cover animate-video-bg ${opacity} ${className}`}
    >
      <source src="/assets/loaders/background-loop.mov" type="video/mp4" />
    </video>
  )
}

export default BackgroundVideo
