// frontend/app/page.tsx
'use client'

import React, { useState, useEffect, Suspense } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { 
  MessageSquare,
  Folder,
  FileText,
  Settings,
  Server,
  AlertCircle,
  CheckCircle,
  Menu,
  Sparkles,
  Loader2
} from 'lucide-react'
import toast from 'react-hot-toast'
import dynamic from 'next/dynamic'

// Dynamically import components to avoid SSR issues
const ChatInterface = dynamic(() => import('@/components/chat/ChatInterface').catch(() => ({ default: DefaultChatInterface })), {
  loading: () => <ComponentLoader name="Chat Interface" />,
  ssr: false
})

const ProjectWorkspace = dynamic(() => import('@/components/projects/ProjectWorkspace').catch(() => ({ default: DefaultProjectWorkspace })), {
  loading: () => <ComponentLoader name="Projects" />,
  ssr: false
})

const DocumentLibrary = dynamic(() => import('@/components/documents/DocumentLibrary').catch(() => ({ default: DefaultDocumentLibrary })), {
  loading: () => <ComponentLoader name="Documents" />,
  ssr: false
})

const SettingsPanel = dynamic(() => import('@/components/settings/SettingsPanel').catch(() => ({ default: DefaultSettingsPanel })), {
  loading: () => <ComponentLoader name="Settings" />,
  ssr: false
})

// Loading Component
const ComponentLoader: React.FC<{ name: string }> = ({ name }) => (
  <div className="h-full flex items-center justify-center bg-gray-50">
    <div className="text-center">
      <Loader2 className="w-8 h-8 text-blue-500 animate-spin mx-auto mb-4" />
      <h2 className="text-lg font-semibold text-gray-900 mb-2">Loading {name}</h2>
      <p className="text-gray-600">Please wait while we load the component...</p>
    </div>
  </div>
)

// Default/Fallback Components
const DefaultChatInterface: React.FC = () => {
  const [messages, setMessages] = useState<Array<{id: string, content: string, role: 'user' | 'assistant', timestamp: string}>>([])
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)

  const sendMessage = async () => {
    if (!input.trim() || isLoading) return

    const userMessage = {
      id: Date.now().toString(),
      content: input.trim(),
      role: 'user' as const,
      timestamp: new Date().toISOString()
    }

    setMessages(prev => [...prev, userMessage])
    setInput('')
    setIsLoading(true)

    try {
      const response = await fetch('http://localhost:8000/api/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: userMessage.content,
          project_id: null
        })
      })

      if (response.ok) {
        const data = await response.json()
        const assistantMessage = {
          id: (Date.now() + 1).toString(),
          content: data.response || 'Sorry, I could not process your request.',
          role: 'assistant' as const,
          timestamp: new Date().toISOString()
        }
        setMessages(prev => [...prev, assistantMessage])
        toast.success('Message sent successfully')
      } else {
        throw new Error('Failed to send message')
      }
    } catch (error) {
      console.error('Chat error:', error)
      const errorMessage = {
        id: (Date.now() + 1).toString(),
        content: 'Sorry, I encountered an error. Please make sure the backend is running.',
        role: 'assistant' as const,
        timestamp: new Date().toISOString()
      }
      setMessages(prev => [...prev, errorMessage])
      toast.error('Failed to send message')
    } finally {
      setIsLoading(false)
    }
  }

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendMessage()
    }
  }

  return (
    <div className="h-full flex flex-col bg-white">
      {/* Header */}
      <div className="border-b border-gray-200 p-4">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 bg-gradient-to-br from-blue-500 to-purple-600 rounded-lg flex items-center justify-center">
            <MessageSquare className="w-5 h-5 text-white" />
          </div>
          <div>
            <h1 className="text-xl font-semibold text-gray-900">AI Chat</h1>
            <p className="text-sm text-gray-600">Chat with your documents using AI</p>
          </div>
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.length === 0 ? (
          <div className="text-center py-12">
            <MessageSquare className="w-16 h-16 text-gray-300 mx-auto mb-4" />
            <h3 className="text-lg font-medium text-gray-900 mb-2">Start a conversation</h3>
            <p className="text-gray-600">Ask questions about your documents or get help with anything.</p>
          </div>
        ) : (
          messages.map((message) => (
            <div key={message.id} className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}>
              <div className={`max-w-xs lg:max-w-md px-4 py-2 rounded-lg ${
                message.role === 'user' 
                  ? 'bg-blue-500 text-white' 
                  : 'bg-gray-100 text-gray-900'
              }`}>
                <p className="text-sm">{message.content}</p>
                <p className="text-xs opacity-75 mt-1">
                  {new Date(message.timestamp).toLocaleTimeString()}
                </p>
              </div>
            </div>
          ))
        )}
        {isLoading && (
          <div className="flex justify-start">
            <div className="bg-gray-100 rounded-lg px-4 py-2">
              <div className="flex items-center space-x-2">
                <Loader2 className="w-4 h-4 animate-spin" />
                <span className="text-sm text-gray-600">AI is thinking...</span>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Input */}
      <div className="border-t border-gray-200 p-4">
        <div className="flex space-x-2">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Ask a question..."
            disabled={isLoading}
            className="flex-1 px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          />
          <button
            onClick={sendMessage}
            disabled={!input.trim() || isLoading}
            className="px-4 py-2 bg-blue-500 text-white rounded-md hover:bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            Send
          </button>
        </div>
      </div>
    </div>
  )
}

const DefaultProjectWorkspace: React.FC = () => {
  const [projects, setProjects] = useState<Array<{id: string, name: string, description: string, created_at: string, document_count: number}>>([])
  const [loading, setLoading] = useState(true)
  const [showCreateForm, setShowCreateForm] = useState(false)
  const [newProject, setNewProject] = useState({ name: '', description: '' })

  useEffect(() => {
    loadProjects()
  }, [])

  const loadProjects = async () => {
    try {
      const response = await fetch('http://localhost:8000/api/projects')
      if (response.ok) {
        const data = await response.json()
        setProjects(data)
      }
    } catch (error) {
      console.error('Failed to load projects:', error)
      toast.error('Failed to load projects')
    } finally {
      setLoading(false)
    }
  }

  const createProject = async () => {
    if (!newProject.name.trim()) return

    try {
      const response = await fetch('http://localhost:8000/api/projects', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(newProject)
      })

      if (response.ok) {
        const project = await response.json()
        setProjects(prev => [project, ...prev])
        setNewProject({ name: '', description: '' })
        setShowCreateForm(false)
        toast.success('Project created successfully')
      }
    } catch (error) {
      console.error('Failed to create project:', error)
      toast.error('Failed to create project')
    }
  }

  return (
    <div className="h-full flex flex-col bg-gray-50">
      {/* Header */}
      <div className="bg-white border-b border-gray-200 p-6">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-gradient-to-br from-green-500 to-teal-600 rounded-lg flex items-center justify-center">
              <Folder className="w-5 h-5 text-white" />
            </div>
            <div>
              <h1 className="text-xl font-semibold text-gray-900">Projects</h1>
              <p className="text-sm text-gray-600">Manage your document collections</p>
            </div>
          </div>
          <button
            onClick={() => setShowCreateForm(true)}
            className="px-4 py-2 bg-green-500 text-white rounded-md hover:bg-green-600"
          >
            New Project
          </button>
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto p-6">
        {loading ? (
          <div className="flex items-center justify-center py-12">
            <Loader2 className="w-8 h-8 animate-spin text-green-500" />
          </div>
        ) : projects.length === 0 ? (
          <div className="text-center py-12">
            <Folder className="w-16 h-16 text-gray-300 mx-auto mb-4" />
            <h3 className="text-lg font-medium text-gray-900 mb-2">No projects yet</h3>
            <p className="text-gray-600 mb-4">Create your first project to get started</p>
            <button
              onClick={() => setShowCreateForm(true)}
              className="px-4 py-2 bg-green-500 text-white rounded-md hover:bg-green-600"
            >
              Create Project
            </button>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {projects.map((project) => (
              <div key={project.id} className="bg-white rounded-lg shadow-sm border border-gray-200 p-4">
                <h3 className="font-medium text-gray-900 mb-2">{project.name}</h3>
                <p className="text-sm text-gray-600 mb-3">{project.description}</p>
                <div className="flex items-center justify-between text-xs text-gray-500">
                  <span>{project.document_count} documents</span>
                  <span>{new Date(project.created_at).toLocaleDateString()}</span>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Create Form Modal */}
      {showCreateForm && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 w-full max-w-md">
            <h2 className="text-lg font-medium mb-4">Create New Project</h2>
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Name</label>
                <input
                  type="text"
                  value={newProject.name}
                  onChange={(e) => setNewProject(prev => ({ ...prev, name: e.target.value }))}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-green-500"
                  placeholder="Project name..."
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Description</label>
                <textarea
                  value={newProject.description}
                  onChange={(e) => setNewProject(prev => ({ ...prev, description: e.target.value }))}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-green-500"
                  rows={3}
                  placeholder="Project description..."
                />
              </div>
            </div>
            <div className="flex justify-end space-x-2 mt-6">
              <button
                onClick={() => setShowCreateForm(false)}
                className="px-4 py-2 text-gray-700 border border-gray-300 rounded-md hover:bg-gray-50"
              >
                Cancel
              </button>
              <button
                onClick={createProject}
                disabled={!newProject.name.trim()}
                className="px-4 py-2 bg-green-500 text-white rounded-md hover:bg-green-600 disabled:opacity-50"
              >
                Create
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

const DefaultDocumentLibrary: React.FC = () => (
  <div className="h-full flex flex-col bg-gray-50">
    <div className="bg-white border-b border-gray-200 p-6">
      <div className="flex items-center gap-3">
        <div className="w-10 h-10 bg-gradient-to-br from-purple-500 to-pink-600 rounded-lg flex items-center justify-center">
          <FileText className="w-5 h-5 text-white" />
        </div>
        <div>
          <h1 className="text-xl font-semibold text-gray-900">Documents</h1>
          <p className="text-sm text-gray-600">Upload and manage your documents</p>
        </div>
      </div>
    </div>
    <div className="flex-1 flex items-center justify-center">
      <div className="text-center">
        <FileText className="w-16 h-16 text-gray-300 mx-auto mb-4" />
        <h3 className="text-lg font-medium text-gray-900 mb-2">Document Library</h3>
        <p className="text-gray-600">Upload documents to get started with AI analysis</p>
      </div>
    </div>
  </div>
)

const DefaultSettingsPanel: React.FC = () => (
  <div className="h-full flex flex-col bg-gray-50">
    <div className="bg-white border-b border-gray-200 p-6">
      <div className="flex items-center gap-3">
        <div className="w-10 h-10 bg-gradient-to-br from-gray-600 to-gray-800 rounded-lg flex items-center justify-center">
          <Settings className="w-5 h-5 text-white" />
        </div>
        <div>
          <h1 className="text-xl font-semibold text-gray-900">Settings</h1>
          <p className="text-sm text-gray-600">Configure your RagFlow experience</p>
        </div>
      </div>
    </div>
    <div className="flex-1 flex items-center justify-center">
      <div className="text-center">
        <Settings className="w-16 h-16 text-gray-300 mx-auto mb-4" />
        <h3 className="text-lg font-medium text-gray-900 mb-2">Settings Panel</h3>
        <p className="text-gray-600">System settings and configuration options</p>
      </div>
    </div>
  </div>
)

// Store Hook
const useStore = () => {
  const [state, setState] = useState({
    currentView: 'chat',
    sidebarCollapsed: false,
  })

  return {
    ...state,
    setCurrentView: (view: string) => setState(prev => ({ ...prev, currentView: view })),
    setSidebarCollapsed: (collapsed: boolean) => setState(prev => ({ ...prev, sidebarCollapsed: collapsed })),
  }
}

// API functions
const createConnectionChecker = (
  onStatusChange: (connected: boolean) => void,
  interval: number = 30000
) => {
  let intervalId: NodeJS.Timeout | null = null

  const check = async () => {
    try {
      const response = await fetch('http://localhost:8000/api/health')
      onStatusChange(response.ok)
    } catch {
      onStatusChange(false)
    }
  }

  const start = () => {
    check()
    intervalId = setInterval(check, interval)
  }

  const stop = () => {
    if (intervalId) {
      clearInterval(intervalId)
      intervalId = null
    }
  }

  return { start, stop, check }
}

// Connection Status Component
const ConnectionStatus: React.FC = () => {
  const [status, setStatus] = useState({
    connected: false,
    checking: true,
    lastCheck: null as Date | null,
  })

  useEffect(() => {
    const checker = createConnectionChecker(
      (connected) => {
        setStatus(prev => ({
          ...prev,
          connected,
          checking: false,
          lastCheck: new Date(),
        }))
      },
      15000
    )

    checker.start()
    return () => checker.stop()
  }, [])

  if (status.checking && !status.lastCheck) {
    return (
      <div className="fixed bottom-6 right-6 z-50">
        <div className="px-4 py-3 bg-blue-50 text-blue-700 border border-blue-200 rounded-lg shadow-lg flex items-center space-x-3">
          <Loader2 className="w-4 h-4 animate-spin" />
          <span className="text-sm font-medium">Checking connection...</span>
        </div>
      </div>
    )
  }

  return (
    <div className="fixed bottom-6 right-6 z-50">
      <div className={`
        px-4 py-3 rounded-lg shadow-lg border flex items-center space-x-3
        ${status.connected 
          ? 'bg-green-50 text-green-700 border-green-200' 
          : 'bg-red-50 text-red-700 border-red-200'
        }
      `}>
        {status.connected ? (
          <CheckCircle className="w-4 h-4 text-green-600" />
        ) : (
          <AlertCircle className="w-4 h-4 text-red-600" />
        )}
        <div>
          <div className="text-sm font-medium">
            Backend {status.connected ? 'Online' : 'Offline'}
          </div>
          {status.lastCheck && (
            <div className="text-xs opacity-75">
              Last check: {status.lastCheck.toLocaleTimeString()}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

// Sidebar Component
const ModernSidebar: React.FC<{
  currentView: string
  onViewChange: (view: string) => void
  isCollapsed: boolean
  onToggleCollapse: () => void
}> = ({ currentView, onViewChange, isCollapsed, onToggleCollapse }) => {
  const navigationItems = [
    {
      id: 'chat',
      label: 'AI Chat',
      icon: MessageSquare,
      description: 'Chat with your documents',
      gradient: 'from-blue-500 to-purple-600'
    },
    {
      id: 'projects',
      label: 'Projects',
      icon: Folder,
      description: 'Manage your workspaces',
      gradient: 'from-green-500 to-teal-600'
    },
    {
      id: 'documents',
      label: 'Documents',
      icon: FileText,
      description: 'Upload and organize files',
      gradient: 'from-purple-500 to-pink-600'
    },
    {
      id: 'settings',
      label: 'Settings',
      icon: Settings,
      description: 'Configure your experience',
      gradient: 'from-gray-600 to-gray-800'
    }
  ]

  return (
    <motion.aside
      initial={false}
      animate={{ width: isCollapsed ? 80 : 320 }}
      transition={{ duration: 0.3, ease: "easeInOut" }}
      className="bg-white border-r border-gray-200 flex flex-col shadow-lg"
    >
      {/* Header */}
      <div className="p-6 border-b border-gray-200">
        <div className="flex items-center justify-between">
          {!isCollapsed && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.1 }}
              className="flex items-center gap-3"
            >
              <div className="w-8 h-8 bg-gradient-to-br from-blue-500 to-purple-600 rounded-lg flex items-center justify-center">
                <Sparkles className="w-4 h-4 text-white" />
              </div>
              <div>
                <h1 className="text-lg font-semibold text-gray-900">RagFlow</h1>
                <p className="text-xs text-gray-600">AI Document Assistant</p>
              </div>
            </motion.div>
          )}
          
          <button
            onClick={onToggleCollapse}
            className="p-2 text-gray-600 hover:text-gray-900 hover:bg-gray-100 rounded-md"
          >
            <Menu className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Navigation */}
      <nav className="flex-1 p-4 space-y-2">
        {navigationItems.map((item) => {
          const isActive = currentView === item.id
          const IconComponent = item.icon
          
          return (
            <button
              key={item.id}
              onClick={() => onViewChange(item.id)}
              className={`
                w-full p-4 rounded-lg transition-all duration-200 text-left
                ${isActive 
                  ? 'bg-gradient-to-r from-blue-50 to-purple-50 shadow-md' 
                  : 'hover:bg-gray-50'
                }
              `}
            >
              <div className="flex items-center gap-3">
                <div className={`
                  w-10 h-10 rounded-lg flex items-center justify-center transition-all
                  ${isActive 
                    ? `bg-gradient-to-r ${item.gradient} text-white shadow-md` 
                    : 'bg-gray-100 text-gray-600'
                  }
                `}>
                  <IconComponent className="w-5 h-5" />
                </div>

                {!isCollapsed && (
                  <div className="flex-1">
                    <div className="font-medium text-gray-900">{item.label}</div>
                    <div className="text-xs text-gray-500 mt-0.5">
                      {item.description}
                    </div>
                  </div>
                )}
              </div>
            </button>
          )
        })}
      </nav>

      {/* Footer */}
      <div className="p-4 border-t border-gray-200">
        {!isCollapsed && (
          <div className="bg-gradient-to-r from-blue-50 to-purple-50 rounded-lg p-3">
            <div className="flex items-center gap-2 mb-2">
              <Sparkles className="w-4 h-4 text-blue-600" />
              <span className="text-sm font-medium text-gray-900">AI Powered</span>
            </div>
            <p className="text-xs text-gray-600">
              Your documents enhanced with artificial intelligence for better insights and conversations.
            </p>
          </div>
        )}
      </div>
    </motion.aside>
  )
}

// Main App Component
export default function RagFlowApp() {
  const store = useStore()

  const renderContent = () => {
    switch (store.currentView) {
      case 'chat':
        return (
          <Suspense fallback={<ComponentLoader name="Chat Interface" />}>
            <ChatInterface />
          </Suspense>
        )
      case 'projects':
        return (
          <Suspense fallback={<ComponentLoader name="Projects" />}>
            <ProjectWorkspace />
          </Suspense>
        )
      case 'documents':
        return (
          <Suspense fallback={<ComponentLoader name="Documents" />}>
            <DocumentLibrary />
          </Suspense>
        )
      case 'settings':
        return (
          <Suspense fallback={<ComponentLoader name="Settings" />}>
            <SettingsPanel />
          </Suspense>
        )
      default:
        return <ChatInterface />
    }
  }

  return (
    <div className="h-screen bg-gray-50 flex overflow-hidden">
      <ModernSidebar
        currentView={store.currentView}
        onViewChange={store.setCurrentView}
        isCollapsed={store.sidebarCollapsed}
        onToggleCollapse={() => store.setSidebarCollapsed(!store.sidebarCollapsed)}
      />

      <motion.div
        initial={false}
        animate={{ marginLeft: store.sidebarCollapsed ? 80 : 320 }}
        transition={{ duration: 0.3, ease: "easeInOut" }}
        className="flex-1 flex flex-col"
      >
        <AnimatePresence mode="wait">
          <motion.div
            key={store.currentView}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            transition={{ duration: 0.3 }}
            className="h-full"
          >
            {renderContent()}
          </motion.div>
        </AnimatePresence>
      </motion.div>

      <ConnectionStatus />
    </div>
  )
}