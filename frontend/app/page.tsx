'use client'

import React, { useState, useEffect, Suspense } from 'react'
import { AnimatePresence } from 'framer-motion'
import dynamic from 'next/dynamic'
import { Sidebar } from '@/components/layout/Sidebar'
import { 
  MainContent, 
  MainContentLoading,
  ChatMainContent,
  ProjectMainContent,
  DocumentMainContent,
  SettingsMainContent
} from '@/components/layout/MainContent'

// Dynamically import components
const ChatInterface = dynamic(
  () => import('@/components/chat/ChatInterface'), 
  { 
    loading: () => <MainContentLoading />,
    ssr: false 
  }
)

const ProjectWorkspace = dynamic(
  () => import('@/components/projects/ProjectWorkspace'), 
  { 
    loading: () => <MainContentLoading />,
    ssr: false 
  }
)

const DocumentLibrary = dynamic(
  () => import('@/components/documents/DocumentLibrary'), 
  { 
    loading: () => <MainContentLoading />,
    ssr: false 
  }
)

const SettingsPanel = dynamic(
  () => import('@/components/settings/SettingsPanel'), 
  { 
    loading: () => <MainContentLoading />,
    ssr: false 
  }
)

export default function HomePage() {
  const [currentView, setCurrentView] = useState('chat')
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false)
  const [isLoading, setIsLoading] = useState(true)

  // Initialize app
  useEffect(() => {
    const timer = setTimeout(() => setIsLoading(false), 1000)
    return () => clearTimeout(timer)
  }, [])

  // Handle view changes with loading state
  const handleViewChange = (newView: string) => {
    if (newView !== currentView) {
      setCurrentView(newView)
    }
  }

  // Render current view content
  const renderCurrentView = () => {
    switch (currentView) {
      case 'chat':
        return (
          <ChatMainContent>
            <Suspense fallback={<MainContentLoading />}>
              <ChatInterface />
            </Suspense>
          </ChatMainContent>
        )
      
      case 'projects':
        return (
          <ProjectMainContent>
            <Suspense fallback={<MainContentLoading />}>
              <ProjectWorkspace />
            </Suspense>
          </ProjectMainContent>
        )
      
      case 'documents':
        return (
          <DocumentMainContent>
            <Suspense fallback={<MainContentLoading />}>
              <DocumentLibrary />
            </Suspense>
          </DocumentMainContent>
        )
      
      case 'settings':
        return (
          <SettingsMainContent>
            <Suspense fallback={<MainContentLoading />}>
              <SettingsPanel />
            </Suspense>
          </SettingsMainContent>
        )
      
      default:
        return (
          <MainContent>
            <div className="flex-1 flex items-center justify-center">
              <div className="text-center">
                <h2 className="text-2xl font-semibold text-gray-900 mb-4">
                  Welcome to RagFlow
                </h2>
                <p className="text-gray-600">
                  Select a section from the sidebar to get started.
                </p>
              </div>
            </div>
          </MainContent>
        )
    }
  }

  // Show loading screen on initial load
  if (isLoading) {
    return (
      <div className="min-h-screen bg-gray-50 flex">
        <div className="w-16 bg-white border-r border-gray-200" />
        <MainContentLoading />
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gray-50 flex overflow-hidden">
      {/* Sidebar */}
      <Sidebar
        currentView={currentView}
        onViewChange={handleViewChange}
        isCollapsed={sidebarCollapsed}
        onToggleCollapse={() => setSidebarCollapsed(!sidebarCollapsed)}
      />

      {/* Main Content Area */}
      <div className="flex-1 flex flex-col overflow-hidden">
        <AnimatePresence mode="wait">
          <div key={currentView} className="flex-1 overflow-hidden">
            {renderCurrentView()}
          </div>
        </AnimatePresence>
      </div>
    </div>
  )
}

// Optional: Add global keyboard shortcuts
export function useKeyboardShortcuts(onViewChange: (view: string) => void) {
  useEffect(() => {
    const handleKeyPress = (event: KeyboardEvent) => {
      if (event.metaKey || event.ctrlKey) {
        switch (event.key) {
          case '1':
            event.preventDefault()
            onViewChange('chat')
            break
          case '2':
            event.preventDefault()
            onViewChange('projects')
            break
          case '3':
            event.preventDefault()
            onViewChange('documents')
            break
          case '4':
            event.preventDefault()
            onViewChange('settings')
            break
        }
      }
    }

    window.addEventListener('keydown', handleKeyPress)
    return () => window.removeEventListener('keydown', handleKeyPress)
  }, [onViewChange])
}