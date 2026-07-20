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
import { useNavigate } from 'react-router-dom'
import TopNav from '../../components/TopNav'
import Modal from '../../components/Modal'
import ByCapabilityView from './ByCapabilityView'
import ByIndustryView, { type Industry } from './ByIndustryView'

function Dashboard() {
  const [viewMode, setViewMode] = useState<'capability' | 'industry'>('capability')
  const [showIndustryModal, setShowIndustryModal] = useState(false)
  const [selectedIndustry, setSelectedIndustry] = useState<Industry | null>(null)
  const navigate = useNavigate()

  return (
    <div className="min-h-screen bg-gm-bg-light dark:bg-gm-bg">
      <TopNav />

      <main className="px-8 py-8">
        <div className="max-w-[1440px] mx-auto">
          {/* Title */}
          <h1 className="text-title text-center mb-6 animate-fade-in">
            Create <span className="text-gm-accent" style={{ textShadow: '0 0 30px rgba(138, 180, 248, 0.5), 0 0 60px rgba(138, 180, 248, 0.3)' }}>New Asset</span>
          </h1>

          {/* Toggle Switch */}
          <div className="flex justify-center mb-8 animate-fade-in-up delay-100">
            <div className="glass-panel rounded-pill p-1.5 inline-flex relative">
              {/* Sliding Background */}
              <div
                className={`absolute top-1.5 h-[calc(100%-12px)] bg-white dark:bg-white/10 rounded-pill shadow-sm transition-all duration-300 ease-out ${
                  viewMode === 'industry' ? 'left-1.5' : 'left-[calc(50%)]'
                }`}
                style={{ width: 'calc(50% - 6px)' }}
              />

              {/* Industry Button */}
              <button
                onClick={() => setViewMode('industry')}
                className={`relative px-6 py-2.5 rounded-pill text-button font-medium transition-all duration-300 z-10 ${
                  viewMode === 'industry'
                    ? 'text-gm-text-primary-light dark:text-gm-text-primary'
                    : 'text-gm-text-secondary-light dark:text-gm-text-secondary hover:text-gm-text-primary-light dark:hover:text-gm-text-primary'
                }`}
              >
                By Industry
              </button>

              {/* Capability Button */}
              <button
                onClick={() => setViewMode('capability')}
                className={`relative px-6 py-2.5 rounded-pill text-button font-medium transition-all duration-300 z-10 ${
                  viewMode === 'capability'
                    ? 'text-gm-text-primary-light dark:text-gm-text-primary'
                    : 'text-gm-text-secondary-light dark:text-gm-text-secondary hover:text-gm-text-primary-light dark:hover:text-gm-text-primary'
                }`}
              >
                By Capability
              </button>
            </div>
          </div>

          {/* Content Area */}
          <div className="max-w-6xl mx-auto relative">
            {viewMode === 'capability' ? (
              <div key="capability" className="animate-fade-scale">
                <ByCapabilityView navigate={navigate} />
              </div>
            ) : (
              <div key="industry" className="animate-fade-scale">
                <ByIndustryView
                  navigate={navigate}
                  onIndustryClick={(industry) => {
                    setSelectedIndustry(industry)
                    setShowIndustryModal(true)
                  }}
                />
              </div>
            )}
          </div>

        </div>
      </main>

      {/* Industry Capability Selection Modal */}
      <Modal
        isOpen={showIndustryModal && selectedIndustry !== null}
        onClose={() => setShowIndustryModal(false)}
        title={selectedIndustry?.title}
      >
        <p className="text-body text-gm-text-secondary-light dark:text-gm-text-secondary mb-8">
          Choose a capability to get started:
        </p>
        <div className="grid grid-cols-2 gap-4">
          {selectedIndustry?.capabilities.map((capability) => (
            <button
              key={capability.id}
              onClick={() => {
                navigate(capability.route)
                setShowIndustryModal(false)
              }}
              className="glass-panel rounded-lg p-6 text-left hover:shadow-level-2 transition-all hover:-translate-y-1 group"
            >
              <div className="w-12 h-12 rounded-lg bg-gm-accent/10 flex items-center justify-center mb-4 group-hover:bg-gm-accent/20 transition-all">
                {capability.icon}
              </div>
              <h3 className="text-h3 font-bold mb-2">{capability.name}</h3>
              <p className="text-sm text-gm-text-secondary-light dark:text-gm-text-secondary">
                {capability.description}
              </p>
            </button>
          ))}
        </div>
      </Modal>
    </div>
  )
}

export default Dashboard
