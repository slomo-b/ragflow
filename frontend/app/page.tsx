// frontend/app/page.tsx
'use client'

import React, { useState, useEffect } from 'react'
import { Sidebar } from '@/components/layout/Sidebar'
import { ChatInterface } from '@/components/chat/ChatInterface'
import { ProjectWorkspace } from '@/components/projects/ProjectWorkspace'
import { DocumentLibrary } from '@/components/documents/DocumentLibrary'
import { SettingsPanel } from '@/components/settings/SettingsPanel'

// Simplified Connection Status
const ConnectionStatus = () => {
  const [status, setStatus] = useState<'connected' | 'disconnected' | 'checking'>('checking')

  useEffect(() => {
    const checkConnection = async () => {
      try {
        const response = await fetch('http://localhost:8000/api/health')
        setStatus(response.ok ? 'connected' : 'disconnected')
      } catch {
        setStatus('disconnected')
      }
    }

    checkConnection()
    const interval = setInterval(checkConnection, 30000)
    return () => clearInterval(interval)
  }, [])

  if (status === 'checking') return null

  return (
    <div className="fixed bottom-6 right-6 z-50">
      <div className={`
        px-3 py-2 rounded-lg shadow-lg text-sm font-medium flex items-center space-x-2
        ${status === 'connected' 
          ? 'bg-green-50 text-green-700 border border-green-200' 
          : 'bg-red-50 text-red-700 border border-red-200'
        }
      `}>
        <div className={`w-2 h-2 rounded-full ${
          status === 'connected' ? 'bg-green-500' : 'bg-red-500'
        }`} />
        <span>Backend {status === 'connected' ? 'Online' : 'Offline'}</span>
      </div>
    </div>
  )
}

export default function RagFlowApp() {
  const [currentView, setCurrentView] = useState('chat')
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false)

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
    <div className="h-screen bg-gray-50 flex overflow-hidden">
      {/* Clean Sidebar */}
      <Sidebar
        currentView={currentView}
        onViewChange={setCurrentView}
        isCollapsed={sidebarCollapsed}
        onToggleCollapse={() => setSidebarCollapsed(!sidebarCollapsed)}
      />

      {/* Main Content Area */}
      <div className={`
        flex-1 transition-all duration-300 
        ${sidebarCollapsed ? 'ml-20' : 'ml-80'}
      `}>
        {renderContent()}
      </div>

      {/* Connection Status */}
      <ConnectionStatus />
    </div>
  )
}