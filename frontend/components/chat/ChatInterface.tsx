// frontend/components/chat/ChatInterface.tsx - Vollständig korrigiert
'use client'

import React, { useState, useRef, useEffect, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { 
  Send, 
  MessageSquare,
  User,
  Bot,
  AlertCircle,
  RefreshCw,
  Trash2,
  Copy,
  Check,
  Settings,
  Loader2,
  FileText,
  Database,
  Sparkles,
  WifiOff,
  Zap,
  StopCircle,
  RotateCcw,
  Save,
  History
} from 'lucide-react'
import toast from 'react-hot-toast'
import ApiService, { Project } from '@/services/api'
import { useChat } from '@/hooks/useChat'

// ✅ Types korrigiert nach Backend Schema
interface ChatStats {
  userMessages: number
  assistantMessages: number
  errorMessages: number
  totalMessages: number
  totalSources: number
  lastActivity: Date | null
}

export const ChatInterface: React.FC = () => {
  // State Management
  const [input, setInput] = useState('')
  const [projects, setProjects] = useState<Project[]>([])
  const [selectedProjectId, setSelectedProjectId] = useState<string | null>(null)
  const [copiedMessageId, setCopiedMessageId] = useState<string | null>(null)
  const [showSettings, setShowSettings] = useState(false)

  // ✅ Use corrected chat hook
  const {
    messages,
    isLoading,
    connectionStatus,
    sendMessage,
    stopGeneration,
    clearChat,
    retryLastMessage,
    testConnection,
    testAI,
    getChatStats,
    checkConnection
  } = useChat(selectedProjectId || undefined)

  // Refs
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLTextAreaElement>(null)

  // ===== AUTO-SCROLL =====
  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [])

  useEffect(() => {
    scrollToBottom()
  }, [messages, scrollToBottom])

  // ===== INITIALIZATION =====
  useEffect(() => {
    initializeChat()
  }, [])

  const initializeChat = async () => {
    console.log('🚀 Initializing chat interface...')
    console.log('🧪 Starting API connectivity test...')
    
    try {
      // ✅ Test connection using corrected ApiService
      const healthCheck = await ApiService.healthCheck()
      
      if (healthCheck.status === 'healthy') {
        console.log('✅ Backend connection established')
        
        // Load projects
        const projectsResponse = await ApiService.getProjects()
        const projectsData = projectsResponse.projects || []
        
        setProjects(projectsData)
        console.log(`📁 Loaded ${projectsData.length} projects`)
        
        // Auto-select first project if available
        if (projectsData.length > 0) {
          setSelectedProjectId(projectsData[0].id)
          console.log(`🎯 Auto-selected project: ${projectsData[0].name}`)
        }
        
        toast.success('Connected to RAGFlow backend!', { duration: 2000 })
      } else {
        throw new Error('Backend unhealthy')
      }
    } catch (error) {
      console.error('💥 Initialization failed:', error)
      toast.error('Failed to initialize chat. Please refresh the page.')
    }
  }

  // ===== MESSAGE HANDLING =====
  const handleSendMessage = useCallback(async () => {
    if (!input.trim() || isLoading) return

    const message = input.trim()
    setInput('')

    // Focus back to input after sending
    setTimeout(() => {
      inputRef.current?.focus()
    }, 100)

    await sendMessage(message)
  }, [input, isLoading, sendMessage])

  const handleKeyPress = useCallback((e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSendMessage()
    }
  }, [handleSendMessage])

  // ===== UTILITY FUNCTIONS =====
  const copyToClipboard = useCallback(async (text: string, messageId: string) => {
    try {
      await navigator.clipboard.writeText(text)
      setCopiedMessageId(messageId)
      toast.success('Copied to clipboard!')
      
      setTimeout(() => {
        setCopiedMessageId(null)
      }, 2000)
    } catch (error) {
      toast.error('Failed to copy to clipboard')
    }
  }, [])

  const handleProjectChange = useCallback((projectId: string) => {
    setSelectedProjectId(projectId || null)
    console.log(`🎯 Selected project: ${projectId}`)
  }, [])

  const refreshProjects = useCallback(async () => {
    try {
      const response = await ApiService.getProjects()
      setProjects(response.projects || [])
      toast.success('Projects refreshed')
    } catch (error) {
      console.error('Failed to refresh projects:', error)
      toast.error('Failed to refresh projects')
    }
  }, [])

  // ===== CHAT STATS =====
  const stats: ChatStats = getChatStats()

  // ===== RENDER HELPERS =====
  const getConnectionStatusColor = () => {
    switch (connectionStatus) {
      case 'connected': return 'text-green-600 bg-green-100'
      case 'disconnected': return 'text-red-600 bg-red-100'
      default: return 'text-yellow-600 bg-yellow-100'
    }
  }

  const getConnectionStatusIcon = () => {
    switch (connectionStatus) {
      case 'connected': return <Zap className="w-4 h-4" />
      case 'disconnected': return <WifiOff className="w-4 h-4" />
      default: return <Loader2 className="w-4 h-4 animate-spin" />
    }
  }

  const renderMessage = (message: any, index: number) => {
    const isUser = message.role === 'user'
    const isError = message.role === 'error'
    const isAssistant = message.role === 'assistant'

    return (
      <motion.div
        key={message.id}
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3, delay: index * 0.1 }}
        className={`flex gap-4 ${isUser ? 'justify-end' : 'justify-start'}`}
      >
        <div className={`flex gap-3 max-w-3xl ${isUser ? 'flex-row-reverse' : 'flex-row'}`}>
          {/* Avatar */}
          <div className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center ${
            isUser ? 'bg-blue-600' : isError ? 'bg-red-600' : 'bg-green-600'
          }`}>
            {isUser ? (
              <User className="w-4 h-4 text-white" />
            ) : isError ? (
              <AlertCircle className="w-4 h-4 text-white" />
            ) : (
              <Bot className="w-4 h-4 text-white" />
            )}
          </div>

          {/* Message Content */}
          <div className={`flex-1 ${isUser ? 'text-right' : 'text-left'}`}>
            <div className={`inline-block p-4 rounded-2xl ${
              isUser 
                ? 'bg-blue-600 text-white' 
                : isError 
                ? 'bg-red-50 text-red-800 border border-red-200'
                : 'bg-gray-100 text-gray-900'
            }`}>
              {/* Message Text */}
              <div className="whitespace-pre-wrap break-words">
                {message.content}
              </div>

              {/* Sources (for assistant messages) */}
              {isAssistant && message.sources && message.sources.length > 0 && (
                <div className="mt-3 pt-3 border-t border-gray-200">
                  <div className="flex items-center gap-2 mb-2">
                    <FileText className="w-4 h-4 text-gray-600" />
                    <span className="text-sm font-medium text-gray-600">
                      Sources ({message.sources.length})
                    </span>
                  </div>
                  <div className="space-y-2">
                    {message.sources.map((source: any, idx: number) => (
                      <div key={idx} className="text-sm bg-gray-50 rounded-lg p-2">
                        <div className="font-medium text-gray-800">{source.filename}</div>
                        <div className="text-gray-600 text-xs mt-1">
                          Relevance: {Math.round(source.relevance_score * 100)}%
                        </div>
                        {source.excerpt && (
                          <div className="text-gray-700 mt-1 text-xs">
                            "{source.excerpt.substring(0, 100)}..."
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Intelligence Metadata */}
              {isAssistant && message.intelligence_metadata && (
                <div className="mt-3 pt-3 border-t border-gray-200">
                  <div className="flex items-center gap-2 mb-2">
                    <Sparkles className="w-4 h-4 text-purple-600" />
                    <span className="text-sm font-medium text-gray-600">AI Analysis</span>
                  </div>
                  <div className="grid grid-cols-3 gap-2 text-xs">
                    <div className="text-center bg-gray-50 rounded p-2">
                      <div className="font-medium">Complexity</div>
                      <div className="text-gray-600">{message.intelligence_metadata.query_complexity}</div>
                    </div>
                    <div className="text-center bg-gray-50 rounded p-2">
                      <div className="font-medium">Reasoning</div>
                      <div className="text-gray-600">{message.intelligence_metadata.reasoning_depth}</div>
                    </div>
                    <div className="text-center bg-gray-50 rounded p-2">
                      <div className="font-medium">Context</div>
                      <div className="text-gray-600">{message.intelligence_metadata.context_integration}</div>
                    </div>
                  </div>
                </div>
              )}
            </div>

            {/* Message Actions */}
            <div className={`flex items-center gap-2 mt-2 ${isUser ? 'justify-end' : 'justify-start'}`}>
              <span className="text-xs text-gray-500">
                {message.timestamp.toLocaleTimeString()}
              </span>
              
              {!isError && (
                <button
                  onClick={() => copyToClipboard(message.content, message.id)}
                  className="text-gray-400 hover:text-gray-600 transition-colors p-1 rounded"
                  title="Copy message"
                >
                  {copiedMessageId === message.id ? (
                    <Check className="w-3 h-3 text-green-600" />
                  ) : (
                    <Copy className="w-3 h-3" />
                  )}
                </button>
              )}
            </div>
          </div>
        </div>
      </motion.div>
    )
  }

  // ===== MAIN RENDER =====
  return (
    <div className="flex flex-col h-screen bg-gradient-to-br from-blue-50 via-indigo-50 to-purple-50">
      {/* Header */}
      <div className="bg-white border-b border-gray-200 shadow-sm">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <div className="flex items-center gap-2">
                <MessageSquare className="w-6 h-6 text-blue-600" />
                <h1 className="text-xl font-semibold text-gray-900">RAG Chat</h1>
              </div>
              
              {/* Connection Status */}
              <div className={`flex items-center gap-2 px-3 py-1 rounded-full text-sm ${getConnectionStatusColor()}`}>
                {getConnectionStatusIcon()}
                <span className="font-medium">
                  {connectionStatus === 'connected' ? 'Connected' :
                   connectionStatus === 'disconnected' ? 'Disconnected' : 'Checking...'}
                </span>
              </div>
            </div>

            {/* Header Controls */}
            <div className="flex items-center gap-2">
              {/* Project Selector */}
              <select
                value={selectedProjectId || ''}
                onChange={(e) => handleProjectChange(e.target.value)}
                className="px-3 py-2 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
              >
                <option value="">No Project Selected</option>
                {projects.map((project) => (
                  <option key={project.id} value={project.id}>
                    {project.name} ({project.document_count} docs)
                  </option>
                ))}
              </select>

              <button
                onClick={refreshProjects}
                className="p-2 text-gray-400 hover:text-gray-600 transition-colors"
                title="Refresh projects"
              >
                <RefreshCw className="w-4 h-4" />
              </button>

              {/* Chat Actions */}
              <div className="flex items-center gap-1 border-l border-gray-200 pl-2">
                <button
                  onClick={testConnection}
                  className="p-2 text-gray-400 hover:text-gray-600 transition-colors"
                  title="Test connection"
                  disabled={connectionStatus === 'checking'}
                >
                  <Database className="w-4 h-4" />
                </button>

                <button
                  onClick={testAI}
                  className="p-2 text-gray-400 hover:text-gray-600 transition-colors"
                  title="Test AI service"
                  disabled={isLoading}
                >
                  <Sparkles className="w-4 h-4" />
                </button>

                <button
                  onClick={retryLastMessage}
                  className="p-2 text-gray-400 hover:text-gray-600 transition-colors"
                  title="Retry last message"
                  disabled={isLoading || messages.length === 0}
                >
                  <RotateCcw className="w-4 h-4" />
                </button>

                <button
                  onClick={clearChat}
                  className="p-2 text-gray-400 hover:text-red-600 transition-colors"
                  title="Clear chat"
                  disabled={isLoading || messages.length === 0}
                >
                  <Trash2 className="w-4 h-4" />
                </button>
              </div>
            </div>
          </div>

          {/* Stats Bar */}
          {stats.totalMessages > 0 && (
            <div className="mt-3 flex items-center gap-6 text-sm text-gray-600">
              <div className="flex items-center gap-1">
                <MessageSquare className="w-4 h-4" />
                <span>{stats.totalMessages} messages</span>
              </div>
              {stats.totalSources > 0 && (
                <div className="flex items-center gap-1">
                  <FileText className="w-4 h-4" />
                  <span>{stats.totalSources} sources found</span>
                </div>
              )}
              {stats.lastActivity && (
                <div className="flex items-center gap-1">
                  <History className="w-4 h-4" />
                  <span>Last: {stats.lastActivity.toLocaleTimeString()}</span>
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Messages Area */}
      <div className="flex-1 overflow-hidden">
        <div className="h-full overflow-y-auto">
          <div className="container mx-auto px-4 py-6">
            {messages.length === 0 ? (
              /* Empty State */
              <div className="flex flex-col items-center justify-center h-full text-center">
                <div className="bg-white rounded-2xl p-8 shadow-lg max-w-md">
                  <Bot className="w-16 h-16 text-blue-600 mx-auto mb-4" />
                  <h3 className="text-xl font-semibold text-gray-900 mb-2">
                    Welcome to RAG Chat!
                  </h3>
                  <p className="text-gray-600 mb-4">
                    Ask questions about your documents and get AI-powered answers with source citations.
                  </p>
                  <div className="space-y-2 text-sm text-gray-500">
                    <div className="flex items-center gap-2">
                      <FileText className="w-4 h-4" />
                      <span>Upload documents to get started</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <Database className="w-4 h-4" />
                      <span>Select a project for context</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <Sparkles className="w-4 h-4" />
                      <span>Ask natural language questions</span>
                    </div>
                  </div>
                </div>
              </div>
            ) : (
              /* Messages */
              <div className="space-y-6 pb-6">
                <AnimatePresence mode="popLayout">
                  {messages.map((message, index) => renderMessage(message, index))}
                </AnimatePresence>
                
                {/* Loading Indicator */}
                {isLoading && (
                  <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="flex gap-4 justify-start"
                  >
                    <div className="flex gap-3 max-w-3xl">
                      <div className="flex-shrink-0 w-8 h-8 rounded-full bg-green-600 flex items-center justify-center">
                        <Bot className="w-4 h-4 text-white" />
                      </div>
                      <div className="bg-gray-100 rounded-2xl p-4">
                        <div className="flex items-center gap-2">
                          <Loader2 className="w-4 h-4 animate-spin text-gray-600" />
                          <span className="text-gray-600">AI is thinking...</span>
                        </div>
                      </div>
                    </div>
                  </motion.div>
                )}
                
                <div ref={messagesEndRef} />
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Input Area */}
      <div className="bg-white border-t border-gray-200 shadow-lg">
        <div className="container mx-auto px-4 py-4">
          <div className="flex gap-4 items-end">
            <div className="flex-1">
              <textarea
                ref={inputRef}
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder={
                  selectedProjectId 
                    ? "Ask a question about your documents..." 
                    : "Select a project and ask a question..."
                }
                disabled={isLoading || connectionStatus === 'disconnected'}
                className="w-full px-4 py-3 border border-gray-300 rounded-xl resize-none focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 disabled:bg-gray-100 disabled:cursor-not-allowed"
                rows={Math.min(Math.max(input.split('\n').length, 1), 4)}
                maxLength={1000}
              />
              <div className="flex justify-between items-center mt-2">
                <div className="text-xs text-gray-500">
                  {input.length}/1000 characters
                </div>
                <div className="text-xs text-gray-500">
                  Press Enter to send, Shift+Enter for new line
                </div>
              </div>
            </div>
            
            <div className="flex gap-2">
              {isLoading ? (
                <button
                  onClick={stopGeneration}
                  className="px-4 py-3 bg-red-600 text-white rounded-xl hover:bg-red-700 transition-colors flex items-center gap-2"
                >
                  <StopCircle className="w-4 h-4" />
                  Stop
                </button>
              ) : (
                <button
                  onClick={handleSendMessage}
                  disabled={!input.trim() || connectionStatus === 'disconnected'}
                  className="px-4 py-3 bg-blue-600 text-white rounded-xl hover:bg-blue-700 transition-colors disabled:bg-gray-400 disabled:cursor-not-allowed flex items-center gap-2"
                >
                  <Send className="w-4 h-4" />
                  Send
                </button>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default ChatInterface