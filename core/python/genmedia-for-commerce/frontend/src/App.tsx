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
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import Dashboard from './pages/Dashboard'
import AssetCreation from './pages/AssetCreation'
import GenMediaTV from './pages/GenMediaTV'
import GenMediaCategories from './pages/GenMediaCategories'
// import PersonalAssistant from './pages/PersonalAssistant'  // WIP

interface StatusWarning {
  title: string
  message: string
}

function App() {
  const [warnings, setWarnings] = useState<StatusWarning[]>([])
  const [dismissedWarnings, setDismissedWarnings] = useState<Set<number>>(new Set())

  useEffect(() => {
    fetch('/api/status')
      .then(response => response.json())
      .then(data => {
        const newWarnings: StatusWarning[] = []
        if (data.vertex_segmentation_enabled === false) {
          newWarnings.push({
            title: 'Vertex AI Segmentation Not Enabled',
            message: 'Background removal is using the local rembg fallback. For best results, enable the image-segmentation-001 model in your project\'s Vertex AI Model Garden.',
          })
        }
        setWarnings(newWarnings)
      })
      .catch(error => console.error('Error fetching status:', error))
  }, [])

  const dismissWarning = (index: number) => {
    setDismissedWarnings(prev => new Set(prev).add(index))
  }

  const activeWarnings = warnings.filter((_, i) => !dismissedWarnings.has(i))

  return (
    <Router>
      {/* Status warnings */}
      {activeWarnings.length > 0 && (
        <div className="fixed top-4 right-4 z-[9999] flex flex-col gap-3 max-w-md">
          {warnings.map((warning, index) => (
            !dismissedWarnings.has(index) && (
              <div
                key={index}
                className="rounded-xl p-4 shadow-lg border border-yellow-500/30"
                style={{
                  background: 'var(--dropdown-bg, #1a1a2e)',
                }}
              >
                <div className="flex items-start gap-3">
                  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" className="text-yellow-500 flex-shrink-0 mt-0.5">
                    <path d="M12 9v4m0 4h.01M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                  </svg>
                  <div className="flex-1">
                    <h4 className="text-sm font-semibold mb-1">{warning.title}</h4>
                    <p className="text-xs text-gm-text-secondary-light dark:text-gm-text-secondary leading-relaxed">
                      {warning.message}
                    </p>
                  </div>
                  <button
                    onClick={() => dismissWarning(index)}
                    className="w-6 h-6 rounded flex items-center justify-center hover:bg-white/10 transition-colors flex-shrink-0"
                  >
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none">
                      <path d="M18 6L6 18M6 6L18 18" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
                    </svg>
                  </button>
                </div>
              </div>
            )
          ))}
        </div>
      )}

      <Routes>
        <Route path="/" element={<Dashboard />} />
        <Route path="/create/:capability" element={<AssetCreation />} />
        <Route path="/genmedia-tv" element={<GenMediaTV />} />
        <Route path="/genmedia-categories" element={<GenMediaCategories />} />
        {/* <Route path="/personal-assistant" element={<PersonalAssistant />} /> */}
      </Routes>
    </Router>
  )
}

export default App
