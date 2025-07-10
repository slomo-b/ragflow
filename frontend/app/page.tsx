'use client'

import React, { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { Sidebar } from '@/components/layout/Sidebar'
import { MainContent } from '@/components/layout/MainContent'
import { ChatInterface } from '@/components/chat/ChatInterface'
import { ProjectWorkspace } from '@/components/projects/ProjectWorkspace'
import { DocumentLibrary } from '@/components/documents/DocumentLibrary'
import { SettingsPanel } from '@/components/settings/SettingsPanel'
import { Toaster } from '@/components/ui/toaster'

// Modern Connection Status Component
const ConnectionStatus = () => {
  const [connectionStatus, setConnectionStatus] = useState<'connected' | 'disconnected' | 'checking'>('checking')

  const checkConnection = async () => {
    try {
      const response = await fetch('http://localhost:8000/api/health')
      setConnectionStatus(response.ok ? 'connected' : 'disconnected')
    } catch {
      setConnectionStatus('disconnected')
    }
  }

  // Check connection on mount
  useEffect(() => {
    checkConnection()
    const interval = setInterval(checkConnection, 30000) // Check every 30s
    return () => clearInterval(interval)
  }, [])

  return (
    <div className="fixed bottom-6 right-6 z-50">
      <motion.div 
        initial={{ opacity: 0, scale: 0.8, y: 20 }}
        animate={{ opacity: 1, scale: 1, y: 0 }}
        transition={{ type: "spring", stiffness: 300, damping: 30 }}
        className={`backdrop-blur-xl rounded-2xl px-4 py-3 shadow-2xl border ${
          connectionStatus === 'connected' 
            ? 'bg-emerald-500/20 border-emerald-400/30 text-emerald-300' 
            : connectionStatus === 'disconnected'
            ? 'bg-red-500/20 border-red-400/30 text-red-300'
            : 'bg-amber-500/20 border-amber-400/30 text-amber-300'
        }`}
      >
        <div className="flex items-center space-x-3">
          <div className={`w-2 h-2 rounded-full shadow-lg ${
            connectionStatus === 'connected' 
              ? 'bg-emerald-400 animate-pulse shadow-emerald-400/50'
              : connectionStatus === 'disconnected'
              ? 'bg-red-400 shadow-red-400/50'
              : 'bg-amber-400 animate-bounce shadow-amber-400/50'
          }`} />
          <span className="text-sm font-medium">
            {connectionStatus === 'connected' && 'Backend Online'}
            {connectionStatus === 'disconnected' && 'Backend Offline'}
            {connectionStatus === 'checking' && 'Connecting...'}
          </span>
        </div>
      </motion.div>
    </div>
  )
}

// Modern Background Effects Component
const BackgroundEffects = ({ isDarkMode }: { isDarkMode: boolean }) => (
  <div className="fixed inset-0 overflow-hidden pointer-events-none">
    <div className={`absolute -top-40 -right-40 w-80 h-80 rounded-full blur-3xl opacity-20 ${
      isDarkMode ? 'bg-gradient-to-r from-purple-500 to-cyan-500' : 'bg-gradient-to-r from-blue-400 to-purple-500'
    }`} />
    <div className={`absolute -bottom-40 -left-40 w-96 h-96 rounded-full blur-3xl opacity-20 ${
      isDarkMode ? 'bg-gradient-to-r from-pink-500 to-violet-500' : 'bg-gradient-to-r from-cyan-400 to-blue-500'
    }`} />
    <div className={`absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-72 h-72 rounded-full blur-3xl opacity-10 ${
      isDarkMode ? 'bg-gradient-to-r from-emerald-500 to-teal-500' : 'bg-gradient-to-r from-purple-400 to-pink-500'
    }`} />
  </div>
)

export default function RagFlowApp() {
  const [currentView, setCurrentView] = useState('chat')
  const [isDarkMode, setIsDarkMode] = useState(false)

  // Check system preference for dark mode on mount
  useEffect(() => {
    const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches
    setIsDarkMode(prefersDark)
  }, [])

  const renderContent = () => {
    switch (currentView) {
      case 'chat':
        return <ChatInterface />
      case 'projects':
        return <ProjectWorkspace />
      case 'documents':
        return <DocumentLibrary />
      case 'settings':
        return <SettingsPanel />
      default:
        return <ChatInterface />
    }
  }

  return (
    <div className={`min-h-screen transition-all duration-500 ${
      isDarkMode 
        ? 'dark bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900' 
        : 'bg-gradient-to-br from-blue-50 via-white to-purple-50'
    }`}>
      {/* Modern Background Effects */}
      <BackgroundEffects isDarkMode={isDarkMode} />

      {/* Main App Layout */}
      <div className="flex h-screen relative">
        {/* Sidebar */}
        <Sidebar 
          currentView={currentView} 
          onViewChange={setCurrentView}
          isDarkMode={isDarkMode}
          onThemeToggle={() => setIsDarkMode(!isDarkMode)}
        />

        {/* Main Content */}
        <MainContent>
          {renderContent()}
        </MainContent>
      </div>

      {/* Connection Status */}
      <ConnectionStatus />

      {/* Toast Notifications */}
      <Toaster />
    </div>
  )
}