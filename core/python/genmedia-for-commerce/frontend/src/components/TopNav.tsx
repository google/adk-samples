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
import { Link, useLocation } from 'react-router-dom'
import logo from '../assets/logo.png'

function TopNav() {
  const location = useLocation()
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

  const isActive = (path: string) => location.pathname === path

  return (
    // Z-index hierarchy: TopNav (z-50) > Capability Selector (z-10) > Content (z-0)
    <header className="px-6 lg:px-8 py-4 border-b border-black/[0.06] dark:border-white/[0.08] bg-white/50 dark:bg-black/20 backdrop-blur-glass sticky top-0 z-50">
      <nav className="max-w-[1440px] mx-auto flex items-center justify-between">
        {/* Logo */}
        <Link to="/" className="flex items-center gap-4 lg:gap-6 group">
          <div className="h-8 lg:h-9 rounded-lg overflow-hidden flex items-center justify-center transition-transform duration-300 group-hover:scale-105">
            <img src={logo} alt="GenMedia for Commerce" className="h-full w-auto object-contain" />
          </div>
          <div className="hidden sm:flex flex-col">
            <span className="text-navbar font-semibold tracking-tight">GenMedia</span>
            <span className="text-caption text-gm-text-tertiary-light dark:text-gm-text-tertiary -mt-0.5">for Commerce</span>
          </div>
        </Link>

        {/* Navigation Links */}
        <div className="hidden md:flex items-center">
          <div className="flex items-center gap-1 bg-black/[0.02] dark:bg-white/[0.03] rounded-xl p-1">
            <Link
              to="/"
              className={`px-4 py-2 rounded-lg text-button font-medium transition-all duration-200 ${
                isActive('/')
                  ? 'bg-white dark:bg-white/10 text-gm-text-primary-light dark:text-gm-text-primary shadow-level-1'
                  : 'text-gm-text-secondary-light dark:text-gm-text-secondary hover:text-gm-text-primary-light dark:hover:text-gm-text-primary'
              }`}
            >
              Dashboard
            </Link>
            <Link
              to="/genmedia-tv"
              className={`px-4 py-2 rounded-lg text-button font-medium transition-all duration-200 ${
                isActive('/genmedia-tv')
                  ? 'bg-white dark:bg-white/10 text-gm-text-primary-light dark:text-gm-text-primary shadow-level-1'
                  : 'text-gm-text-secondary-light dark:text-gm-text-secondary hover:text-gm-text-primary-light dark:hover:text-gm-text-primary'
              }`}
            >
              GenMedia TV
            </Link>
            {/* Personal Assistant — hidden (WIP)
            <Link
              to="/personal-assistant"
              className={`px-4 py-2 rounded-lg text-button font-medium transition-all duration-200 ${
                isActive('/personal-assistant')
                  ? 'bg-white dark:bg-white/10 text-gm-text-primary-light dark:text-gm-text-primary shadow-level-1'
                  : 'text-gm-text-secondary-light dark:text-gm-text-secondary hover:text-gm-text-primary-light dark:hover:text-gm-text-primary'
              }`}
            >
              Personal Assistant
            </Link>
            */}
          </div>
        </div>

        {/* Right Side Icons */}
        <div className="flex items-center gap-3">
          {/* Dark/Light Mode Toggle */}
          <button
            onClick={() => setIsDarkMode(!isDarkMode)}
            className="relative w-14 h-7 rounded-full transition-all duration-300 overflow-hidden group"
            style={{
              background: isDarkMode
                ? 'linear-gradient(135deg, #1a1a2e 0%, #16213e 100%)'
                : 'linear-gradient(135deg, #87CEEB 0%, #98D8E8 100%)'
            }}
            aria-label={isDarkMode ? 'Switch to light mode' : 'Switch to dark mode'}
          >
            {/* Track decoration */}
            <div className="absolute inset-0 overflow-hidden">
              {isDarkMode ? (
                // Stars for dark mode
                <>
                  <div className="absolute top-1.5 left-2 w-1 h-1 rounded-full bg-white/40" />
                  <div className="absolute top-3 left-4 w-0.5 h-0.5 rounded-full bg-white/30" />
                  <div className="absolute bottom-2 left-3 w-0.5 h-0.5 rounded-full bg-white/20" />
                </>
              ) : (
                // Clouds for light mode
                <div className="absolute top-1 right-2 w-4 h-2 rounded-full bg-white/40" />
              )}
            </div>

            {/* Toggle knob */}
            <div
              className={`absolute top-1 w-5 h-5 rounded-full bg-white shadow-lg transition-all duration-300 flex items-center justify-center ${
                isDarkMode ? 'left-1' : 'left-8'
              }`}
            >
              {isDarkMode ? (
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none">
                  <path
                    d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"
                    fill="#6366f1"
                  />
                </svg>
              ) : (
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none">
                  <circle cx="12" cy="12" r="5" fill="#f59e0b"/>
                  <g stroke="#f59e0b" strokeWidth="2" strokeLinecap="round">
                    <path d="M12 1v2M12 21v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42M1 12h2M21 12h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42"/>
                  </g>
                </svg>
              )}
            </div>
          </button>

          {/* Mobile menu button */}
          <button className="md:hidden w-9 h-9 flex items-center justify-center rounded-lg hover:bg-black/[0.04] dark:hover:bg-white/[0.06] transition-colors">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
              <path d="M3 12h18M3 6h18M3 18h18" />
            </svg>
          </button>
        </div>
      </nav>
    </header>
  )
}

export default TopNav
