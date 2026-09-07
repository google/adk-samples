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

import { useState, useRef, useEffect, useCallback } from 'react'

// FaceMesh is loaded globally from script tag in index.html
// eslint-disable-next-line @typescript-eslint/no-explicit-any
declare const FaceMesh: any

interface CameraCaptureProps {
  isOpen: boolean
  onCapture: (imageDataUrl: string) => void
  onClose: () => void
}

type LightingStatus = 'good' | 'poor' | 'checking'
type FacePosition = 'centered' | 'off-center' | 'too-close' | 'too-far' | 'not-detected'

function CameraCapture({ isOpen, onCapture, onClose }: CameraCaptureProps) {
  const [cameraStream, setCameraStream] = useState<MediaStream | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [lightingStatus, setLightingStatus] = useState<LightingStatus>('checking')
  const [facePosition, setFacePosition] = useState<FacePosition>('not-detected')

  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const faceMeshRef = useRef<any>(null)
  const analysisIntervalRef = useRef<number | null>(null)
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const lastResultsRef = useRef<any>(null)

  // Stop camera and cleanup
  const stopCamera = useCallback(() => {
    console.log('🛑 Stopping camera...')
    if (analysisIntervalRef.current) {
      clearInterval(analysisIntervalRef.current)
      analysisIntervalRef.current = null
    }
    if (cameraStream) {
      cameraStream.getTracks().forEach(track => track.stop())
      setCameraStream(null)
    }
    if (faceMeshRef.current) {
      faceMeshRef.current.close()
      faceMeshRef.current = null
    }
    setLightingStatus('checking')
    setFacePosition('not-detected')
    setIsLoading(true)
  }, [cameraStream])

  // Analyze lighting from center of frame
  const analyzeLighting = useCallback((ctx: CanvasRenderingContext2D, width: number, height: number): LightingStatus => {
    try {
      const centerX = Math.floor(width / 2)
      const centerY = Math.floor(height / 2)
      const faceSize = Math.floor(Math.min(width, height) * 0.15)

      const faceData = ctx.getImageData(
        Math.max(0, centerX - faceSize / 2),
        Math.max(0, centerY - faceSize / 2),
        faceSize,
        faceSize
      )

      let total = 0, count = 0
      for (let i = 0; i < faceData.data.length; i += 16) {
        const brightness = (faceData.data[i] + faceData.data[i + 1] + faceData.data[i + 2]) / 3
        total += brightness
        count++
      }
      const faceBrightness = count > 0 ? total / count : 128

      if (faceBrightness < 60) return 'poor'
      if (faceBrightness > 220) return 'poor'
      if (faceBrightness >= 80 && faceBrightness <= 200) return 'good'
      return 'poor'
    } catch {
      return 'checking'
    }
  }, [])

  // Analyze face position from MediaPipe landmarks
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const analyzeFacePosition = useCallback((landmarks: any[], width: number, height: number): FacePosition => {
    if (!landmarks || landmarks.length === 0) return 'not-detected'

    // MediaPipe FaceMesh landmark indices
    const noseTip = landmarks[1]
    const leftEye = landmarks[159]
    const rightEye = landmarks[386]

    if (!noseTip || !leftEye || !rightEye) return 'not-detected'

    const centerX = width / 2
    const centerY = height / 2
    const noseX = noseTip.x * width
    const noseY = noseTip.y * height

    // Calculate eye distance
    const leftEyeX = leftEye.x * width
    const leftEyeY = leftEye.y * height
    const rightEyeX = rightEye.x * width
    const rightEyeY = rightEye.y * height
    const eyeDistance = Math.sqrt(Math.pow(rightEyeX - leftEyeX, 2) + Math.pow(rightEyeY - leftEyeY, 2))

    // Check distance from camera
    const idealEyeDistance = width * 0.2
    const eyeDistanceRatio = eyeDistance / idealEyeDistance

    if (eyeDistanceRatio > 2.2) return 'too-close'
    if (eyeDistanceRatio < 0.4) return 'too-far'

    // Check if face is centered
    const horizontalTolerance = width * 0.2
    const verticalTolerance = height * 0.2

    if (Math.abs(noseX - centerX) > horizontalTolerance || Math.abs(noseY - centerY) > verticalTolerance) {
      return 'off-center'
    }

    return 'centered'
  }, [])

  // Start the analysis loop
  const startAnalysis = useCallback(() => {
    const video = videoRef.current
    const canvas = canvasRef.current
    if (!video || !canvas) return
    if (video.videoWidth === 0 || video.videoHeight === 0) return

    console.log('🔍 Starting analysis loop...')

    // Set canvas size to match video
    canvas.width = video.videoWidth
    canvas.height = video.videoHeight

    // Clear any existing interval
    if (analysisIntervalRef.current) {
      clearInterval(analysisIntervalRef.current)
    }

    const analyze = () => {
      if (!video || video.paused || video.ended) return
      if (video.videoWidth === 0 || video.videoHeight === 0) return

      const ctx = canvas.getContext('2d', { willReadFrequently: true })
      if (!ctx) return

      ctx.drawImage(video, 0, 0, canvas.width, canvas.height)

      // Lighting analysis
      const lighting = analyzeLighting(ctx, canvas.width, canvas.height)
      setLightingStatus(lighting)

      // Face position analysis from MediaPipe results
      if (lastResultsRef.current?.multiFaceLandmarks?.[0]) {
        const landmarks = lastResultsRef.current.multiFaceLandmarks[0]
        const position = analyzeFacePosition(landmarks, canvas.width, canvas.height)
        setFacePosition(position)
      } else {
        setFacePosition('not-detected')
      }
    }

    analysisIntervalRef.current = window.setInterval(analyze, 400)
  }, [analyzeLighting, analyzeFacePosition])

  // Initialize MediaPipe FaceMesh (in background)
  const initFaceDetector = useCallback(async () => {
    if (faceMeshRef.current) return

    try {
      console.log('🚀 Initializing Face Detection...')

      const faceMesh = new FaceMesh({
        locateFile: (file: string) => {
          const url = `/mediapipe/face_mesh/${file}`
          console.log(`📁 Loading Face Mesh file: ${url}`)
          return url
        }
      })

      faceMesh.setOptions({
        maxNumFaces: 1,
        refineLandmarks: true,
        minDetectionConfidence: 0.5,
        minTrackingConfidence: 0.5
      })

      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      faceMesh.onResults((results: any) => {
        lastResultsRef.current = results
      })

      await faceMesh.initialize()
      faceMeshRef.current = faceMesh

      // Start processing video frames
      const video = videoRef.current
      if (video) {
        const processFrame = async () => {
          if (faceMeshRef.current && video.readyState >= 2) {
            await faceMeshRef.current.send({ image: video })
          }
          if (faceMeshRef.current) {
            requestAnimationFrame(processFrame)
          }
        }
        processFrame()
      }

      console.log('✅ Face Detection initialized')
    } catch (err) {
      console.log('ℹ️ Face detection disabled - lighting analysis only', err)
    }
  }, [])

  // Start camera (matching frontend_dev pattern)
  const startCamera = useCallback(async () => {
    console.log('🎥 Starting camera...')
    setIsLoading(true)
    setError(null)

    if (!navigator.mediaDevices?.getUserMedia) {
      setError('Camera not supported in this browser')
      setIsLoading(false)
      return
    }

    try {
      let stream: MediaStream
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: {
            facingMode: 'user',
            width: { ideal: 1280 },
            height: { ideal: 720 }
          }
        })
      } catch {
        stream = await navigator.mediaDevices.getUserMedia({ video: true })
      }

      console.log('✅ Camera stream obtained')
      setCameraStream(stream)

      const video = videoRef.current
      if (!video) {
        console.error('❌ Video ref is null!')
        return
      }

      video.srcObject = stream

      // Following frontend_dev pattern: onloadedmetadata -> play() -> start analysis
      video.onloadedmetadata = () => {
        console.log('📹 Video metadata loaded:', video.videoWidth, 'x', video.videoHeight)

        video.play().then(() => {
          console.log('▶️ Video playing successfully')
          setIsLoading(false)

          // Start analysis with multiple attempts (like frontend_dev)
          startAnalysis()
          setTimeout(() => startAnalysis(), 500)
          setTimeout(() => startAnalysis(), 1000)

          // Init Face Detector in background
          initFaceDetector()
        }).catch(err => {
          console.error('❌ Video play error:', err)
          setError('Could not start video playback')
          setIsLoading(false)
        })
      }

      video.onerror = (err) => {
        console.error('❌ Video error:', err)
        setError('Video error occurred')
        setIsLoading(false)
      }

    } catch (err) {
      console.error('❌ Camera error:', err)
      if (err instanceof Error) {
        if (err.name === 'NotAllowedError') {
          setError('Camera access denied. Please allow camera permissions.')
        } else if (err.name === 'NotFoundError') {
          setError('No camera found on this device.')
        } else {
          setError('Could not access camera: ' + err.message)
        }
      } else {
        setError('Could not access camera')
      }
      setIsLoading(false)
    }
  }, [startAnalysis, initFaceDetector])

  // Handle open/close
  useEffect(() => {
    if (isOpen) {
      startCamera()
    }
    return () => {
      if (cameraStream) {
        cameraStream.getTracks().forEach(track => track.stop())
      }
      if (analysisIntervalRef.current) {
        clearInterval(analysisIntervalRef.current)
      }
      if (faceMeshRef.current) {
        faceMeshRef.current.close()
        faceMeshRef.current = null
      }
    }
  }, [isOpen]) // Only depend on isOpen to avoid re-running

  const handleCapture = () => {
    const video = videoRef.current
    const canvas = canvasRef.current
    if (!video || !canvas) return

    const ctx = canvas.getContext('2d', { willReadFrequently: true })
    if (!ctx) return

    canvas.width = video.videoWidth
    canvas.height = video.videoHeight
    ctx.drawImage(video, 0, 0)

    const imageDataUrl = canvas.toDataURL('image/png')
    stopCamera()
    onCapture(imageDataUrl)
  }

  const handleClose = () => {
    stopCamera()
    onClose()
  }

  const getPositionMessage = () => {
    switch (facePosition) {
      case 'centered': return 'Face centered'
      case 'off-center': return 'Move face to center'
      case 'too-close': return 'Move further away'
      case 'too-far': return 'Move closer'
      case 'not-detected': return 'No face detected'
    }
  }

  const getLightingMessage = () => {
    switch (lightingStatus) {
      case 'good': return 'Good lighting'
      case 'poor': return 'Improve lighting'
      case 'checking': return 'Checking...'
    }
  }

  const isReadyToCapture = facePosition === 'centered' && lightingStatus === 'good'

  if (!isOpen) return null

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm">
      <div className="relative bg-gm-bg-primary rounded-2xl overflow-hidden shadow-2xl max-w-2xl w-full mx-4">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-white/10">
          <h3 className="text-lg font-semibold">Take Photo</h3>
          <button
            onClick={handleClose}
            className="w-8 h-8 rounded-full bg-white/10 hover:bg-white/20 flex items-center justify-center transition-colors"
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
              <path d="M18 6L6 18M6 6L18 18" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
            </svg>
          </button>
        </div>

        {/* Camera View */}
        <div className="relative aspect-[4/3] bg-black">
          {isLoading && (
            <div className="absolute inset-0 flex items-center justify-center z-10 bg-black">
              <div className="flex flex-col items-center gap-3">
                <svg className="animate-spin h-8 w-8 text-gm-accent" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none"/>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"/>
                </svg>
                <p className="text-sm text-gm-text-secondary">Starting camera...</p>
              </div>
            </div>
          )}

          {error && (
            <div className="absolute inset-0 flex items-center justify-center z-10 bg-black">
              <div className="flex flex-col items-center gap-3 text-center px-6">
                <svg width="48" height="48" viewBox="0 0 24 24" fill="none" className="text-red-400">
                  <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="2"/>
                  <path d="M12 8v4M12 16h.01" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
                </svg>
                <p className="text-sm text-red-400">{error}</p>
                <button
                  onClick={startCamera}
                  className="px-4 py-2 bg-gm-accent hover:bg-gm-accent-hover rounded-lg text-sm font-medium transition-colors"
                >
                  Try Again
                </button>
              </div>
            </div>
          )}

          <video
            ref={videoRef}
            playsInline
            muted
            className="w-full h-full object-cover"
            style={{ transform: 'scaleX(-1)' }}
          />

          {/* Face guide overlay */}
          {!isLoading && !error && (
            <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
              <div className={`w-48 h-64 border-2 rounded-[50%] border-dashed transition-colors ${
                facePosition === 'centered' ? 'border-green-400' :
                facePosition === 'not-detected' ? 'border-white/30' : 'border-yellow-400'
              }`} />
            </div>
          )}

          {/* Status indicators */}
          {!isLoading && !error && (
            <div className="absolute bottom-4 left-1/2 -translate-x-1/2 flex gap-2">
              <div className={`px-2 py-1 rounded-full backdrop-blur-sm text-[10px] font-medium flex items-center gap-1 ${
                facePosition === 'centered' ? 'bg-green-500/20 text-green-400' :
                facePosition === 'not-detected' ? 'bg-white/10 text-white/60' : 'bg-yellow-500/20 text-yellow-400'
              }`}>
                <div className={`w-1.5 h-1.5 rounded-full ${
                  facePosition === 'centered' ? 'bg-green-400' :
                  facePosition === 'not-detected' ? 'bg-white/40' : 'bg-yellow-400'
                }`} />
                {getPositionMessage()}
              </div>

              <div className={`px-2 py-1 rounded-full backdrop-blur-sm text-[10px] font-medium flex items-center gap-1 ${
                lightingStatus === 'good' ? 'bg-green-500/20 text-green-400' :
                lightingStatus === 'checking' ? 'bg-white/10 text-white/60' : 'bg-yellow-500/20 text-yellow-400'
              }`}>
                <div className={`w-1.5 h-1.5 rounded-full ${
                  lightingStatus === 'good' ? 'bg-green-400' :
                  lightingStatus === 'checking' ? 'bg-white/40' : 'bg-yellow-400'
                }`} />
                {getLightingMessage()}
              </div>
            </div>
          )}

          <canvas ref={canvasRef} style={{ display: 'none' }} />
        </div>

        {/* Controls */}
        <div className="p-4 flex items-center justify-center gap-4">
          <button
            onClick={handleClose}
            className="px-6 py-2.5 rounded-xl bg-white/10 hover:bg-white/20 text-sm font-medium transition-colors"
          >
            Cancel
          </button>
          <button
            onClick={handleCapture}
            disabled={isLoading || !!error}
            className={`px-6 py-2.5 rounded-xl text-sm font-medium transition-colors flex items-center gap-2 ${
              isReadyToCapture ? 'bg-green-500 hover:bg-green-600' : 'bg-gm-accent hover:bg-gm-accent-hover'
            } disabled:opacity-50 disabled:cursor-not-allowed`}
          >
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
              <circle cx="12" cy="12" r="3" stroke="currentColor" strokeWidth="2"/>
              <path d="M3 9a2 2 0 012-2h1.5a1 1 0 00.8-.4l1.4-1.8a1 1 0 01.8-.4h5a1 1 0 01.8.4l1.4 1.8a1 1 0 00.8.4H19a2 2 0 012 2v9a2 2 0 01-2 2H5a2 2 0 01-2-2V9z" stroke="currentColor" strokeWidth="2"/>
            </svg>
            {isReadyToCapture ? 'Capture' : 'Capture Anyway'}
          </button>
        </div>
      </div>
    </div>
  )
}

export default CameraCapture
