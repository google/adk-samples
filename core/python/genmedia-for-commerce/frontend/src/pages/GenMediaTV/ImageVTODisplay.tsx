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

interface ImageVTODisplayProps {
  modelImage: string
  garmentImages: string[]
  vtoResult: string
  description?: string
  isPlaying?: boolean
}

function ImageVTODisplay({
  modelImage,
  garmentImages,
  vtoResult
}: ImageVTODisplayProps) {
  return (
    <div className="relative w-full h-full rounded-2xl overflow-hidden bg-gray-200 dark:bg-black p-1">
      {/* Main Layout: Left (Inputs) + Right (Result) */}
      <div className="flex gap-1 h-full">
        {/* LEFT COLUMN - Inputs (Model + Garments) */}
        <div className="flex flex-col gap-1 w-[40%]">
          {/* Model Image - Top Half */}
          <div className="relative bg-white rounded-tl-2xl overflow-hidden flex items-center justify-center flex-1 animate-slide-in-left">
            <img
              src={modelImage}
              alt="Model"
              className="max-w-[90%] max-h-[90%] object-contain animate-zoom-in"
              style={{ animationDelay: '0.2s' }}
            />
            <div className="absolute bottom-3 left-3 px-2.5 py-1 bg-black/70 backdrop-blur-sm rounded text-[11px] font-semibold text-white/90 uppercase tracking-wide animate-fade-in" style={{ animationDelay: '0.5s' }}>
              Model
            </div>
          </div>

          {/* Garment Images - Bottom Half */}
          <div className="relative bg-white rounded-bl-2xl overflow-hidden flex items-center justify-center flex-1 animate-slide-in-left" style={{ animationDelay: '0.3s' }}>
            {garmentImages.length === 1 ? (
              <>
                <img
                  src={garmentImages[0]}
                  alt="Garment"
                  className="max-w-[90%] max-h-[90%] object-contain animate-zoom-in"
                  style={{ animationDelay: '0.5s' }}
                />
                <div className="absolute bottom-3 left-3 px-2.5 py-1 bg-black/70 backdrop-blur-sm rounded text-[11px] font-semibold text-white/90 uppercase tracking-wide animate-fade-in" style={{ animationDelay: '0.8s' }}>
                  Garment
                </div>
              </>
            ) : garmentImages.length === 2 ? (
              <div className="grid grid-cols-2 gap-2 w-full h-full p-3">
                {garmentImages.map((garment, idx) => (
                  <div key={idx} className="relative bg-gray-50 rounded overflow-hidden flex items-center justify-center animate-pop-in" style={{ animationDelay: `${0.5 + idx * 0.15}s` }}>
                    <img
                      src={garment}
                      alt={`Garment ${idx + 1}`}
                      className="max-w-[85%] max-h-[85%] object-contain"
                    />
                  </div>
                ))}
                <div className="absolute bottom-2 left-2 px-2 py-0.5 bg-black/70 backdrop-blur-sm rounded text-[9px] font-semibold text-white/90 uppercase animate-fade-in" style={{ animationDelay: '0.9s' }}>
                  Garments (2)
                </div>
              </div>
            ) : (
              <div className="grid grid-cols-3 gap-1.5 w-full h-full p-2.5">
                {garmentImages.map((garment, idx) => (
                  <div key={idx} className="relative bg-gray-50 rounded overflow-hidden flex items-center justify-center animate-pop-in" style={{ animationDelay: `${0.5 + idx * 0.1}s` }}>
                    <img
                      src={garment}
                      alt={`Garment ${idx + 1}`}
                      className="max-w-[85%] max-h-[85%] object-contain"
                    />
                  </div>
                ))}
                <div className="absolute bottom-2 left-2 px-2 py-0.5 bg-black/70 backdrop-blur-sm rounded text-[9px] font-semibold text-white/90 uppercase animate-fade-in" style={{ animationDelay: '0.9s' }}>
                  Garments ({garmentImages.length})
                </div>
              </div>
            )}
          </div>
        </div>

        {/* RIGHT COLUMN - VTO Result */}
        <div className="relative bg-gradient-to-br from-amber-50 via-white to-emerald-50 rounded-r-2xl overflow-hidden flex items-center justify-center flex-1 animate-slide-in-right" style={{ animationDelay: '0.6s' }}>
          <div className="w-full h-full flex items-center justify-center p-4 animate-result-reveal" style={{ animationDelay: '0.9s' }}>
            <img
              src={vtoResult}
              alt="Virtual Try-On Result"
              className="max-w-full max-h-full object-contain"
            />
          </div>

          {/* Result Label */}
          <div className="absolute top-4 right-4 px-3 py-1.5 bg-gm-accent/90 backdrop-blur-sm rounded-lg shadow-lg animate-slide-down" style={{ animationDelay: '1.2s' }}>
            <span className="text-xs font-bold text-white uppercase tracking-wider">
              Virtual Try-On
            </span>
          </div>

          {/* Subtle corner accents */}
          <div className="absolute top-0 right-0 w-16 h-16 border-t-2 border-r-2 border-gm-accent/20 rounded-tr-2xl animate-draw-border" style={{ animationDelay: '1.3s' }} />
          <div className="absolute bottom-0 right-0 w-16 h-16 border-b-2 border-r-2 border-gm-accent/20 rounded-br-2xl animate-draw-border" style={{ animationDelay: '1.4s' }} />
        </div>
      </div>

      <style>{`
        @keyframes slide-in-left {
          from {
            opacity: 0;
            transform: translateX(-40px);
          }
          to {
            opacity: 1;
            transform: translateX(0);
          }
        }

        @keyframes slide-in-right {
          from {
            opacity: 0;
            transform: translateX(40px);
          }
          to {
            opacity: 1;
            transform: translateX(0);
          }
        }

        @keyframes zoom-in {
          from {
            opacity: 0;
            transform: scale(0.85);
          }
          to {
            opacity: 1;
            transform: scale(1);
          }
        }

        @keyframes fade-in {
          from {
            opacity: 0;
          }
          to {
            opacity: 1;
          }
        }

        @keyframes pop-in {
          from {
            opacity: 0;
            transform: scale(0.7) rotate(-5deg);
          }
          to {
            opacity: 1;
            transform: scale(1) rotate(0deg);
          }
        }

        @keyframes result-reveal {
          0% {
            opacity: 0;
            transform: scale(0.9) rotateY(-15deg);
          }
          60% {
            opacity: 1;
            transform: scale(1.02) rotateY(2deg);
          }
          100% {
            opacity: 1;
            transform: scale(1) rotateY(0deg);
          }
        }

        @keyframes slide-down {
          from {
            opacity: 0;
            transform: translateY(-20px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }

        @keyframes draw-border {
          from {
            opacity: 0;
            clip-path: polygon(0 0, 0 0, 0 100%, 0 100%);
          }
          to {
            opacity: 1;
            clip-path: polygon(0 0, 100% 0, 100% 100%, 0 100%);
          }
        }

        @keyframes gentle-float {
          0%, 100% {
            transform: translateY(0px) scale(1);
          }
          50% {
            transform: translateY(-8px) scale(1.01);
          }
        }

        .animate-slide-in-left {
          animation: slide-in-left 0.7s cubic-bezier(0.16, 1, 0.3, 1) both;
        }

        .animate-slide-in-right {
          animation: slide-in-right 0.8s cubic-bezier(0.16, 1, 0.3, 1) both;
        }

        .animate-zoom-in {
          animation: zoom-in 0.6s cubic-bezier(0.16, 1, 0.3, 1) both;
        }

        .animate-fade-in {
          animation: fade-in 0.5s ease-out both;
        }

        .animate-pop-in {
          animation: pop-in 0.5s cubic-bezier(0.34, 1.56, 0.64, 1) both;
        }

        .animate-result-reveal {
          animation: result-reveal 1s cubic-bezier(0.16, 1, 0.3, 1) both;
          transform-style: preserve-3d;
          perspective: 1000px;
        }

        .animate-slide-down {
          animation: slide-down 0.6s cubic-bezier(0.16, 1, 0.3, 1) both;
        }

        .animate-draw-border {
          animation: draw-border 0.8s ease-out both;
        }

        .animate-gentle-float {
          animation: gentle-float 4s ease-in-out infinite;
          animation-delay: 1.5s;
        }
      `}</style>
    </div>
  )
}

export default ImageVTODisplay
