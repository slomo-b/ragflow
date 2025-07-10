'use client'

import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'

// API Configuration
const API_BASE_URL = 'http://localhost:8000'

// Types
interface Project {
  id: string
  name: string
  description: string
  created_at: string
  updated_at: string
  document_ids: string[]
  document_count: number
  chat_count: number
  status: string
  settings: Record<string, any>
}

interface Document {
  id: string
  filename: string
  file_type: string
  file_size: number
  uploaded_at: string
  project_ids: string[]
  tags: string[]
  processing_status: string
}

interface ChatMessage {
  role: 'user' | 'assistant'
  content: string
  timestamp?: string
}

interface ChatResponse {
  response: string
  chat_id: string
  project_id?: string
  timestamp: string
  model_info: {
    model: string
    temperature: number
  }
  sources: any[]
}

export default function RagFlowApp() {
  // State Management
  const [activeTab, setActiveTab] = useState('dashboard')
  const [isDarkMode, setIsDarkMode] = useState(false)
  const [sidebarOpen, setSidebarOpen] = useState(true)
  
  // Data State
  const [projects, setProjects] = useState<Project[]>([])
  const [documents, setDocuments] = useState<Document[]>([])
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [currentProject, setCurrentProject] = useState<Project | null>(null)
  
  // UI State
  const [isLoading, setIsLoading] = useState(false)
  const [chatInput, setChatInput] = useState('')
  const [connectionStatus, setConnectionStatus] = useState<'connected' | 'disconnected' | 'checking'>('checking')
  const [notifications, setNotifications] = useState<Array<{ id: string; type: 'success' | 'error' | 'info'; message: string }>>([])

  // API Functions
  const checkBackendConnection = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/health`)
      if (response.ok) {
        setConnectionStatus('connected')
        return true
      } else {
        setConnectionStatus('disconnected')
        return false
      }
    } catch (error) {
      setConnectionStatus('disconnected')
      return false
    }
  }

  const loadProjects = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/projects/`)
      if (response.ok) {
        const data = await response.json()
        const projectsArray = Array.isArray(data) ? data : (data.projects || [])
        setProjects(projectsArray)
        if (projectsArray.length > 0 && !currentProject) {
          setCurrentProject(projectsArray[0])
        }
      }
    } catch (error) {
      console.error('Failed to load projects:', error)
      addNotification('error', 'Failed to load projects')
    }
  }

  const loadDocuments = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/documents/`)
      if (response.ok) {
        const data = await response.json()
        setDocuments(data.documents || [])
      }
    } catch (error) {
      console.error('Failed to load documents:', error)
    }
  }

  const createProject = async (name: string, description: string = '') => {
    setIsLoading(true)
    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/projects/`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name, description })
      })
      
      if (response.ok) {
        const newProject = await response.json()
        setProjects(prev => [...prev, newProject])
        setCurrentProject(newProject)
        addNotification('success', `Project "${name}" created successfully!`)
        return newProject
      } else {
        throw new Error('Failed to create project')
      }
    } catch (error) {
      addNotification('error', 'Failed to create project')
      throw error
    } finally {
      setIsLoading(false)
    }
  }

  const uploadDocument = async (files: FileList, projectId?: string) => {
    if (!files.length) return

    const formData = new FormData()
    Array.from(files).forEach(file => {
      formData.append('files', file)
    })
    
    if (projectId) {
      formData.append('project_id', projectId)
    }

    setIsLoading(true)
    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/upload/documents`, {
        method: 'POST',
        body: formData
      })
      
      if (response.ok) {
        const data = await response.json()
        addNotification('success', `Successfully uploaded ${data.documents.length} document(s)`)
        await loadDocuments()
        await loadProjects()
        return data.documents
      } else {
        const errorData = await response.json()
        throw new Error(errorData.detail || 'Upload failed')
      }
    } catch (error) {
      addNotification('error', `Upload failed: ${error instanceof Error ? error.message : 'Unknown error'}`)
      throw error
    } finally {
      setIsLoading(false)
    }
  }

  const sendChatMessage = async (message: string, projectId?: string) => {
    const userMessage: ChatMessage = {
      role: 'user',
      content: message,
      timestamp: new Date().toISOString()
    }
    
    setMessages(prev => [...prev, userMessage])
    setChatInput('')
    setIsLoading(true)

    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          messages: [{ role: 'user', content: message }],
          project_id: projectId
        })
      })
      
      if (response.ok) {
        const data: ChatResponse = await response.json()
        const aiMessage: ChatMessage = {
          role: 'assistant',
          content: data.response,
          timestamp: data.timestamp
        }
        setMessages(prev => [...prev, aiMessage])
      } else {
        const errorData = await response.json()
        throw new Error(errorData.detail || 'Chat failed')
      }
    } catch (error) {
      const errorMessage: ChatMessage = {
        role: 'assistant',
        content: `Sorry, there was an error: ${error instanceof Error ? error.message : 'Unknown error'}`,
        timestamp: new Date().toISOString()
      }
      setMessages(prev => [...prev, errorMessage])
    } finally {
      setIsLoading(false)
    }
  }

  // Utility Functions
  const addNotification = (type: 'success' | 'error' | 'info', message: string) => {
    const id = Date.now().toString()
    setNotifications(prev => [...prev, { id, type, message }])
    setTimeout(() => {
      setNotifications(prev => prev.filter(n => n.id !== id))
    }, 5000)
  }

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 Bytes'
    const k = 1024
    const sizes = ['Bytes', 'KB', 'MB', 'GB']
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
  }

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    })
  }

  // Effects
  useEffect(() => {
    checkBackendConnection()
    loadProjects()
    loadDocuments()
  }, [])

  // Navigation items
  const navigationItems = [
    { id: 'dashboard', name: 'Dashboard', icon: '📊' },
    { id: 'chat', name: 'AI Chat', icon: '💬' },
    { id: 'projects', name: 'Projects', icon: '📁' },
    { id: 'documents', name: 'Documents', icon: '📄' },
    { id: 'settings', name: 'Settings', icon: '⚙️' }
  ]

  return (
    <div className={`min-h-screen transition-all duration-300 ${isDarkMode ? 'dark bg-gray-900' : 'bg-gray-50'}`}>
      {/* Notifications */}
      <div className="fixed top-4 right-4 z-50 space-y-2">
        <AnimatePresence>
          {notifications.map((notification) => (
            <motion.div
              key={notification.id}
              initial={{ opacity: 0, x: 300 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: 300 }}
              className={`px-4 py-3 rounded-lg shadow-lg min-w-[300px] ${
                notification.type === 'success' 
                  ? 'bg-green-500 text-white' 
                  : notification.type === 'error'
                  ? 'bg-red-500 text-white'
                  : 'bg-blue-500 text-white'
              }`}
            >
              {notification.message}
            </motion.div>
          ))}
        </AnimatePresence>
      </div>

      {/* Connection Status */}
      <div className="fixed top-4 left-4 z-40">
        <div className={`px-3 py-1 rounded-full text-xs font-medium ${
          connectionStatus === 'connected' 
            ? 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200' 
            : connectionStatus === 'disconnected'
            ? 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200'
            : 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200'
        }`}>
          {connectionStatus === 'connected' && '🟢 Connected'}
          {connectionStatus === 'disconnected' && '🔴 Offline'}
          {connectionStatus === 'checking' && '🟡 Connecting...'}
        </div>
      </div>

      <div className="flex h-screen">
        {/* Sidebar */}
        <motion.aside
          initial={false}
          animate={{ width: sidebarOpen ? 280 : 80 }}
          className={`${isDarkMode ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-200'} border-r transition-all duration-300 flex flex-col`}
        >
          {/* Header */}
          <div className="p-4 border-b border-gray-200 dark:border-gray-700">
            <div className="flex items-center justify-between">
              {sidebarOpen && (
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  className="flex items-center space-x-3"
                >
                  <div className="w-8 h-8 bg-gradient-to-r from-blue-500 to-purple-600 rounded-lg flex items-center justify-center">
                    <span className="text-white font-bold text-sm">R</span>
                  </div>
                  <h1 className="text-xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
                    RagFlow
                  </h1>
                </motion.div>
              )}
              <button
                onClick={() => setSidebarOpen(!sidebarOpen)}
                className="p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
              >
                {sidebarOpen ? '◀' : '▶'}
              </button>
            </div>
          </div>

          {/* Current Project */}
          {sidebarOpen && currentProject && (
            <div className="p-4 border-b border-gray-200 dark:border-gray-700">
              <div className="text-xs text-gray-500 dark:text-gray-400 mb-1">Current Project</div>
              <div className="font-medium text-sm text-gray-900 dark:text-white truncate">
                {currentProject.name}
              </div>
              <div className="text-xs text-gray-500 dark:text-gray-400">
                {currentProject.document_count} documents
              </div>
            </div>
          )}

          {/* Navigation */}
          <nav className="flex-1 p-4">
            <div className="space-y-2">
              {navigationItems.map((item) => (
                <button
                  key={item.id}
                  onClick={() => setActiveTab(item.id)}
                  className={`w-full flex items-center px-3 py-2 rounded-lg transition-colors text-left ${
                    activeTab === item.id
                      ? 'bg-blue-100 text-blue-700 dark:bg-blue-900 dark:text-blue-200'
                      : 'text-gray-600 hover:bg-gray-100 dark:text-gray-300 dark:hover:bg-gray-700'
                  }`}
                >
                  <span className="text-lg mr-3">{item.icon}</span>
                  {sidebarOpen && (
                    <motion.span
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      className="font-medium"
                    >
                      {item.name}
                    </motion.span>
                  )}
                </button>
              ))}
            </div>
          </nav>

          {/* Theme Toggle */}
          <div className="p-4 border-t border-gray-200 dark:border-gray-700">
            <button
              onClick={() => setIsDarkMode(!isDarkMode)}
              className="w-full flex items-center px-3 py-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
            >
              <span className="text-lg mr-3">{isDarkMode ? '☀️' : '🌙'}</span>
              {sidebarOpen && (
                <span className="font-medium text-gray-600 dark:text-gray-300">
                  {isDarkMode ? 'Light Mode' : 'Dark Mode'}
                </span>
              )}
            </button>
          </div>
        </motion.aside>

        {/* Main Content */}
        <main className="flex-1 overflow-hidden flex flex-col">
          {/* Header */}
          <header className={`${isDarkMode ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-200'} border-b px-6 py-4`}>
            <div className="flex items-center justify-between">
              <div>
                <h2 className="text-2xl font-bold text-gray-900 dark:text-white capitalize">
                  {activeTab === 'dashboard' ? 'Dashboard' : navigationItems.find(i => i.id === activeTab)?.name}
                </h2>
                <p className="text-gray-500 dark:text-gray-400 text-sm">
                  {activeTab === 'dashboard' && 'Overview of your projects and recent activity'}
                  {activeTab === 'chat' && 'Chat with AI about your documents'}
                  {activeTab === 'projects' && 'Manage your document projects'}
                  {activeTab === 'documents' && 'Upload and organize your files'}
                  {activeTab === 'settings' && 'Configure your preferences'}
                </p>
              </div>
              
              {/* Quick Actions */}
              <div className="flex items-center space-x-3">
                {activeTab === 'projects' && (
                  <button
                    onClick={() => {
                      const name = prompt('Project name:')
                      if (name) createProject(name)
                    }}
                    disabled={isLoading}
                    className="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 disabled:opacity-50 transition-colors flex items-center space-x-2"
                  >
                    <span>➕</span>
                    <span>New Project</span>
                  </button>
                )}
                
                {activeTab === 'documents' && (
                  <>
                    <input
                      type="file"
                      multiple
                      accept=".pdf,.doc,.docx,.txt,.md"
                      onChange={(e) => e.target.files && uploadDocument(e.target.files, currentProject?.id)}
                      className="hidden"
                      id="file-upload"
                      disabled={isLoading}
                    />
                    <label
                      htmlFor="file-upload"
                      className={`px-4 py-2 bg-purple-500 text-white rounded-lg hover:bg-purple-600 cursor-pointer transition-colors flex items-center space-x-2 ${
                        isLoading ? 'opacity-50 cursor-not-allowed' : ''
                      }`}
                    >
                      <span>📤</span>
                      <span>Upload Files</span>
                    </label>
                  </>
                )}
              </div>
            </div>
          </header>

          {/* Content Area */}
          <div className="flex-1 overflow-auto p-6">
            <AnimatePresence mode="wait">
              <motion.div
                key={activeTab}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                transition={{ duration: 0.2 }}
                className="h-full"
              >
                {/* Dashboard */}
                {activeTab === 'dashboard' && (
                  <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 h-full">
                    {/* Stats Cards */}
                    <div className="lg:col-span-3 grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
                      <div className={`${isDarkMode ? 'bg-gray-800' : 'bg-white'} rounded-xl p-6 shadow-sm border ${isDarkMode ? 'border-gray-700' : 'border-gray-200'}`}>
                        <div className="flex items-center">
                          <div className="p-3 bg-blue-100 dark:bg-blue-900 rounded-lg">
                            <span className="text-2xl">📁</span>
                          </div>
                          <div className="ml-4">
                            <p className="text-sm text-gray-500 dark:text-gray-400">Projects</p>
                            <p className="text-2xl font-bold text-gray-900 dark:text-white">{projects.length}</p>
                          </div>
                        </div>
                      </div>
                      
                      <div className={`${isDarkMode ? 'bg-gray-800' : 'bg-white'} rounded-xl p-6 shadow-sm border ${isDarkMode ? 'border-gray-700' : 'border-gray-200'}`}>
                        <div className="flex items-center">
                          <div className="p-3 bg-purple-100 dark:bg-purple-900 rounded-lg">
                            <span className="text-2xl">📄</span>
                          </div>
                          <div className="ml-4">
                            <p className="text-sm text-gray-500 dark:text-gray-400">Documents</p>
                            <p className="text-2xl font-bold text-gray-900 dark:text-white">{documents.length}</p>
                          </div>
                        </div>
                      </div>
                      
                      <div className={`${isDarkMode ? 'bg-gray-800' : 'bg-white'} rounded-xl p-6 shadow-sm border ${isDarkMode ? 'border-gray-700' : 'border-gray-200'}`}>
                        <div className="flex items-center">
                          <div className="p-3 bg-green-100 dark:bg-green-900 rounded-lg">
                            <span className="text-2xl">💬</span>
                          </div>
                          <div className="ml-4">
                            <p className="text-sm text-gray-500 dark:text-gray-400">Chats</p>
                            <p className="text-2xl font-bold text-gray-900 dark:text-white">{messages.length}</p>
                          </div>
                        </div>
                      </div>
                    </div>

                    {/* Recent Projects */}
                    <div className="lg:col-span-2">
                      <div className={`${isDarkMode ? 'bg-gray-800' : 'bg-white'} rounded-xl p-6 shadow-sm border ${isDarkMode ? 'border-gray-700' : 'border-gray-200'} h-full`}>
                        <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">Recent Projects</h3>
                        <div className="space-y-3">
                          {projects.slice(0, 5).map((project) => (
                            <div
                              key={project.id}
                              onClick={() => setCurrentProject(project)}
                              className={`p-4 rounded-lg border cursor-pointer transition-all hover:shadow-md ${
                                currentProject?.id === project.id
                                  ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20'
                                  : 'border-gray-200 dark:border-gray-700 hover:border-gray-300 dark:hover:border-gray-600'
                              }`}
                            >
                              <h4 className="font-medium text-gray-900 dark:text-white">{project.name}</h4>
                              <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">{project.description}</p>
                              <div className="flex items-center justify-between mt-3">
                                <span className="text-xs text-gray-500 dark:text-gray-400">
                                  {project.document_count} documents
                                </span>
                                <span className="text-xs text-gray-500 dark:text-gray-400">
                                  {formatDate(project.updated_at)}
                                </span>
                              </div>
                            </div>
                          ))}
                          {projects.length === 0 && (
                            <div className="text-center py-8 text-gray-500 dark:text-gray-400">
                              <p>No projects yet</p>
                              <button
                                onClick={() => setActiveTab('projects')}
                                className="mt-2 text-blue-500 hover:text-blue-600 text-sm"
                              >
                                Create your first project →
                              </button>
                            </div>
                          )}
                        </div>
                      </div>
                    </div>

                    {/* Quick Chat */}
                    <div className="lg:col-span-1">
                      <div className={`${isDarkMode ? 'bg-gray-800' : 'bg-white'} rounded-xl p-6 shadow-sm border ${isDarkMode ? 'border-gray-700' : 'border-gray-200'} h-full flex flex-col`}>
                        <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">Quick Chat</h3>
                        <div className="flex-1 flex flex-col">
                          <div className="flex-1 mb-4">
                            <div className="space-y-3 max-h-60 overflow-y-auto">
                              {messages.slice(-3).map((message, index) => (
                                <div
                                  key={index}
                                  className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
                                >
                                  <div
                                    className={`max-w-xs px-3 py-2 rounded-lg text-sm ${
                                      message.role === 'user'
                                        ? 'bg-blue-500 text-white'
                                        : 'bg-gray-100 dark:bg-gray-700 text-gray-900 dark:text-white'
                                    }`}
                                  >
                                    {message.content}
                                  </div>
                                </div>
                              ))}
                            </div>
                          </div>
                          <div className="flex space-x-2">
                            <input
                              type="text"
                              value={chatInput}
                              onChange={(e) => setChatInput(e.target.value)}
                              onKeyPress={(e) => e.key === 'Enter' && chatInput.trim() && sendChatMessage(chatInput, currentProject?.id)}
                              placeholder="Ask something..."
                              className="flex-1 px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg text-sm bg-transparent focus:outline-none focus:ring-2 focus:ring-blue-500"
                              disabled={isLoading}
                            />
                            <button
                              onClick={() => chatInput.trim() && sendChatMessage(chatInput, currentProject?.id)}
                              disabled={isLoading || !chatInput.trim()}
                              className="px-3 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 disabled:opacity-50 transition-colors text-sm"
                            >
                              →
                            </button>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                )}

                {/* Chat Tab */}
                {activeTab === 'chat' && (
                  <div className={`${isDarkMode ? 'bg-gray-800' : 'bg-white'} rounded-xl shadow-sm border ${isDarkMode ? 'border-gray-700' : 'border-gray-200'} h-full flex flex-col`}>
                    <div className="p-6 border-b border-gray-200 dark:border-gray-700">
                      <div className="flex items-center justify-between">
                        <div>
                          <h3 className="text-lg font-semibold text-gray-900 dark:text-white">AI Chat</h3>
                          <p className="text-sm text-gray-500 dark:text-gray-400">
                            {currentProject ? `Chatting about: ${currentProject.name}` : 'Select a project to start chatting'}
                          </p>
                        </div>
                        {messages.length > 0 && (
                          <button
                            onClick={() => setMessages([])}
                            className="px-3 py-1 text-sm text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200"
                          >
                            Clear Chat
                          </button>
                        )}
                      </div>
                    </div>
                    
                    <div className="flex-1 overflow-y-auto p-6">
                      <div className="space-y-4">
                        {messages.map((message, index) => (
                          <div
                            key={index}
                            className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
                          >
                            <div
                              className={`max-w-md lg:max-w-2xl px-4 py-3 rounded-lg ${
                                message.role === 'user'
                                  ? 'bg-blue-500 text-white'
                                  : 'bg-gray-100 dark:bg-gray-700 text-gray-900 dark:text-white'
                              }`}
                            >
                              <div className="whitespace-pre-wrap">{message.content}</div>
                              {message.timestamp && (
                                <div className={`text-xs mt-2 opacity-75 ${
                                  message.role === 'user' ? 'text-blue-100' : 'text-gray-500 dark:text-gray-400'
                                }`}>
                                  {formatDate(message.timestamp)}
                                </div>
                              )}
                            </div>
                          </div>
                        ))}
                        {isLoading && (
                          <div className="flex justify-start">
                            <div className="bg-gray-100 dark:bg-gray-700 px-4 py-3 rounded-lg">
                              <div className="flex space-x-1">
                                <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></div>
                                <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
                                <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                              </div>
                            </div>
                          </div>
                        )}
                      </div>
                      
                      {messages.length === 0 && (
                        <div className="flex items-center justify-center h-full">
                          <div className="text-center">
                            <div className="text-6xl mb-4">💬</div>
                            <h4 className="text-lg font-medium text-gray-900 dark:text-white mb-2">Start a conversation</h4>
                            <p className="text-gray-500 dark:text-gray-400 mb-6">
                              Ask questions about your documents or get help with your projects
                            </p>
                            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 max-w-md mx-auto">
                              {[
                                "Summarize my documents",
                                "What are the key points?",
                                "Help me understand this",
                                "Find specific information"
                              ].map((suggestion, index) => (
                                <button
                                  key={index}
                                  onClick={() => sendChatMessage(suggestion, currentProject?.id)}
                                  className="p-3 text-left text-sm bg-gray-50 dark:bg-gray-700 hover:bg-gray-100 dark:hover:bg-gray-600 rounded-lg transition-colors"
                                >
                                  {suggestion}
                                </button>
                              ))}
                            </div>
                          </div>
                        </div>
                      )}
                    </div>
                    
                    <div className="p-6 border-t border-gray-200 dark:border-gray-700">
                      <div className="flex space-x-3">
                        <input
                          type="text"
                          value={chatInput}
                          onChange={(e) => setChatInput(e.target.value)}
                          onKeyPress={(e) => e.key === 'Enter' && !e.shiftKey && chatInput.trim() && sendChatMessage(chatInput, currentProject?.id)}
                          placeholder="Type your message... (Enter to send, Shift+Enter for new line)"
                          className="flex-1 px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-lg bg-transparent focus:outline-none focus:ring-2 focus:ring-blue-500 resize-none"
                          disabled={isLoading}
                        />
                        <button
                          onClick={() => chatInput.trim() && sendChatMessage(chatInput, currentProject?.id)}
                          disabled={isLoading || !chatInput.trim()}
                          className="px-6 py-3 bg-blue-500 text-white rounded-lg hover:bg-blue-600 disabled:opacity-50 transition-colors font-medium"
                        >
                          Send
                        </button>
                      </div>
                    </div>
                  </div>
                )}

                {/* Projects Tab */}
                {activeTab === 'projects' && (
                  <div>
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
                      {Array.isArray(projects) && projects.map((project) => (
                        <motion.div
                          key={project.id}
                          whileHover={{ scale: 1.02 }}
                          onClick={() => setCurrentProject(project)}
                          className={`${isDarkMode ? 'bg-gray-800' : 'bg-white'} rounded-xl p-6 shadow-sm border cursor-pointer transition-all hover:shadow-lg ${
                            currentProject?.id === project.id
                              ? 'border-blue-500 ring-2 ring-blue-200 dark:ring-blue-800'
                              : 'border-gray-200 dark:border-gray-700 hover:border-gray-300 dark:hover:border-gray-600'
                          }`}
                        >
                          <div className="flex items-start justify-between mb-4">
                            <div className="p-3 bg-blue-100 dark:bg-blue-900 rounded-lg">
                              <span className="text-2xl">📁</span>
                            </div>
                            <div className={`px-2 py-1 rounded-full text-xs font-medium ${
                              project.status === 'active' 
                                ? 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200'
                                : 'bg-gray-100 text-gray-800 dark:bg-gray-900 dark:text-gray-200'
                            }`}>
                              {project.status}
                            </div>
                          </div>
                          
                          <h4 className="font-semibold text-gray-900 dark:text-white mb-2 truncate">
                            {project.name}
                          </h4>
                          <p className="text-sm text-gray-500 dark:text-gray-400 mb-4 line-clamp-2">
                            {project.description || 'No description'}
                          </p>
                          
                          <div className="flex items-center justify-between text-xs text-gray-500 dark:text-gray-400">
                            <span>{project.document_count} documents</span>
                            <span>{formatDate(project.updated_at)}</span>
                          </div>
                        </motion.div>
                      ))}
                    </div>
                    
                    {Array.isArray(projects) && projects.length === 0 && (
                      <div className="text-center py-16">
                        <div className="text-6xl mb-4">📁</div>
                        <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-2">No projects yet</h3>
                        <p className="text-gray-500 dark:text-gray-400 mb-6 max-w-md mx-auto">
                          Create your first project to start organizing your documents and AI conversations
                        </p>
                        <button
                          onClick={() => {
                            const name = prompt('Project name:')
                            if (name) createProject(name)
                          }}
                          className="px-6 py-3 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors font-medium"
                        >
                          Create First Project
                        </button>
                      </div>
                    )}
                  </div>
                )}

                {/* Documents Tab */}
                {activeTab === 'documents' && (
                  <div>
                    {!currentProject ? (
                      <div className="text-center py-16">
                        <div className="text-6xl mb-4">📄</div>
                        <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-2">Select a project</h3>
                        <p className="text-gray-500 dark:text-gray-400 mb-6">
                          Choose a project to view and manage its documents
                        </p>
                        <button
                          onClick={() => setActiveTab('projects')}
                          className="px-6 py-3 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors font-medium"
                        >
                          Go to Projects
                        </button>
                      </div>
                    ) : (
                      <>
                        {/* Document Grid */}
                        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
                          {Array.isArray(documents) && documents
                            .filter(doc => doc.project_ids.includes(currentProject.id))
                            .map((document) => (
                              <motion.div
                                key={document.id}
                                whileHover={{ scale: 1.02 }}
                                className={`${isDarkMode ? 'bg-gray-800' : 'bg-white'} rounded-xl p-6 shadow-sm border border-gray-200 dark:border-gray-700 hover:shadow-lg transition-all`}
                              >
                                <div className="flex items-start justify-between mb-4">
                                  <div className="p-3 bg-purple-100 dark:bg-purple-900 rounded-lg">
                                    <span className="text-2xl">
                                      {document.file_type === 'pdf' ? '📕' : 
                                       document.file_type === 'docx' ? '📘' : 
                                       document.file_type === 'txt' ? '📄' : '📋'}
                                    </span>
                                  </div>
                                  <div className={`px-2 py-1 rounded-full text-xs font-medium ${
                                    document.processing_status === 'completed'
                                      ? 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200'
                                      : 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200'
                                  }`}>
                                    {document.processing_status}
                                  </div>
                                </div>
                                
                                <h4 className="font-semibold text-gray-900 dark:text-white mb-2 truncate" title={document.filename}>
                                  {document.filename}
                                </h4>
                                
                                <div className="space-y-2 text-sm text-gray-500 dark:text-gray-400">
                                  <div className="flex justify-between">
                                    <span>Type:</span>
                                    <span className="uppercase">{document.file_type}</span>
                                  </div>
                                  <div className="flex justify-between">
                                    <span>Size:</span>
                                    <span>{formatFileSize(document.file_size)}</span>
                                  </div>
                                  <div className="flex justify-between">
                                    <span>Uploaded:</span>
                                    <span>{formatDate(document.uploaded_at)}</span>
                                  </div>
                                </div>
                                
                                {document.tags.length > 0 && (
                                  <div className="mt-4 flex flex-wrap gap-1">
                                    {document.tags.slice(0, 3).map((tag, index) => (
                                      <span
                                        key={index}
                                        className="px-2 py-1 text-xs bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-300 rounded"
                                      >
                                        {tag}
                                      </span>
                                    ))}
                                    {document.tags.length > 3 && (
                                      <span className="px-2 py-1 text-xs bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-300 rounded">
                                        +{document.tags.length - 3}
                                      </span>
                                    )}
                                  </div>
                                )}
                              </motion.div>
                            ))}
                        </div>
                        
                        {/* Empty State */}
                        {Array.isArray(documents) && documents.filter(doc => doc.project_ids.includes(currentProject.id)).length === 0 && (
                          <div className="text-center py-16">
                            <div className="text-6xl mb-4">📄</div>
                            <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-2">No documents yet</h3>
                            <p className="text-gray-500 dark:text-gray-400 mb-6 max-w-md mx-auto">
                              Upload documents to {currentProject.name} to start analyzing them with AI
                            </p>
                            <label
                              htmlFor="file-upload-empty"
                              className="inline-flex items-center px-6 py-3 bg-purple-500 text-white rounded-lg hover:bg-purple-600 cursor-pointer transition-colors font-medium"
                            >
                              <span className="mr-2">📤</span>
                              Upload First Document
                            </label>
                            <input
                              type="file"
                              multiple
                              accept=".pdf,.doc,.docx,.txt,.md"
                              onChange={(e) => e.target.files && uploadDocument(e.target.files, currentProject.id)}
                              className="hidden"
                              id="file-upload-empty"
                              disabled={isLoading}
                            />
                          </div>
                        )}
                      </>
                    )}
                  </div>
                )}

                {/* Settings Tab */}
                {activeTab === 'settings' && (
                  <div className="max-w-4xl mx-auto">
                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                      {/* General Settings */}
                      <div className={`${isDarkMode ? 'bg-gray-800' : 'bg-white'} rounded-xl p-6 shadow-sm border ${isDarkMode ? 'border-gray-700' : 'border-gray-200'}`}>
                        <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">General Settings</h3>
                        <div className="space-y-4">
                          <div className="flex items-center justify-between">
                            <div>
                              <label className="font-medium text-gray-900 dark:text-white">Dark Mode</label>
                              <p className="text-sm text-gray-500 dark:text-gray-400">Toggle dark/light theme</p>
                            </div>
                            <button
                              onClick={() => setIsDarkMode(!isDarkMode)}
                              className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
                                isDarkMode ? 'bg-blue-600' : 'bg-gray-200'
                              }`}
                            >
                              <span
                                className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                                  isDarkMode ? 'translate-x-6' : 'translate-x-1'
                                }`}
                              />
                            </button>
                          </div>
                          
                          <div className="flex items-center justify-between">
                            <div>
                              <label className="font-medium text-gray-900 dark:text-white">Sidebar Collapsed</label>
                              <p className="text-sm text-gray-500 dark:text-gray-400">Minimize sidebar by default</p>
                            </div>
                            <button
                              onClick={() => setSidebarOpen(!sidebarOpen)}
                              className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
                                !sidebarOpen ? 'bg-blue-600' : 'bg-gray-200'
                              }`}
                            >
                              <span
                                className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                                  !sidebarOpen ? 'translate-x-6' : 'translate-x-1'
                                }`}
                              />
                            </button>
                          </div>
                        </div>
                      </div>

                      {/* System Information */}
                      <div className={`${isDarkMode ? 'bg-gray-800' : 'bg-white'} rounded-xl p-6 shadow-sm border ${isDarkMode ? 'border-gray-700' : 'border-gray-200'}`}>
                        <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">System Information</h3>
                        <div className="space-y-3">
                          <div className="flex justify-between">
                            <span className="text-gray-500 dark:text-gray-400">Backend Status</span>
                            <span className={`font-medium ${
                              connectionStatus === 'connected' 
                                ? 'text-green-600 dark:text-green-400' 
                                : 'text-red-600 dark:text-red-400'
                            }`}>
                              {connectionStatus === 'connected' ? 'Connected' : 'Disconnected'}
                            </span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-gray-500 dark:text-gray-400">API URL</span>
                            <span className="text-gray-900 dark:text-white font-mono text-sm">{API_BASE_URL}</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-gray-500 dark:text-gray-400">Total Projects</span>
                            <span className="text-gray-900 dark:text-white">{projects.length}</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-gray-500 dark:text-gray-400">Total Documents</span>
                            <span className="text-gray-900 dark:text-white">{documents.length}</span>
                          </div>
                        </div>
                        
                        <div className="mt-6 pt-4 border-t border-gray-200 dark:border-gray-700">
                          <button
                            onClick={() => {
                              checkBackendConnection()
                              loadProjects()
                              loadDocuments()
                              addNotification('info', 'Data refreshed')
                            }}
                            className="w-full px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors font-medium"
                          >
                            Refresh Data
                          </button>
                        </div>
                      </div>

                      {/* Advanced Settings */}
                      <div className={`lg:col-span-2 ${isDarkMode ? 'bg-gray-800' : 'bg-white'} rounded-xl p-6 shadow-sm border ${isDarkMode ? 'border-gray-700' : 'border-gray-200'}`}>
                        <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">Advanced Settings</h3>
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                          <div>
                            <label className="block font-medium text-gray-900 dark:text-white mb-2">
                              Upload Settings
                            </label>
                            <div className="space-y-2 text-sm text-gray-500 dark:text-gray-400">
                              <div className="flex justify-between">
                                <span>Max file size:</span>
                                <span>10 MB</span>
                              </div>
                              <div className="flex justify-between">
                                <span>Allowed types:</span>
                                <span>PDF, DOCX, TXT, MD</span>
                              </div>
                            </div>
                          </div>
                          
                          <div>
                            <label className="block font-medium text-gray-900 dark:text-white mb-2">
                              AI Settings
                            </label>
                            <div className="space-y-2 text-sm text-gray-500 dark:text-gray-400">
                              <div className="flex justify-between">
                                <span>Model:</span>
                                <span>Gemini 1.5 Flash</span>
                              </div>
                              <div className="flex justify-between">
                                <span>Temperature:</span>
                                <span>0.7</span>
                              </div>
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                )}
              </motion.div>
            </AnimatePresence>
          </div>
        </main>
      </div>
    </div>
  )
}