// frontend/components/chat/ChatInterface.tsx - An neue API angepasst
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
  Zap
} from 'lucide-react'
import toast from 'react-hot-toast'
import { ApiService } from '@/services/api'
import { useChat } from '@/hooks/useChat'

// Types
interface Project {
  id: string
  name: string
  description: string
  document_count: number
}

interface ChatStats {
  totalMessages: number
  responseTime: number
  lastActivity: string
}

export const ChatInterface: React.FC = () => {
  // State Management
  const [input, setInput] = useState('')
  const [projects, setProjects] = useState<Project[]>([])
  const [selectedProjectId, setSelectedProjectId] = useState<string | null>(null)
  const [copiedMessageId, setCopiedMessageId] = useState<string | null>(null)
  const [stats, setStats] = useState<ChatStats>({
    totalMessages: 0,
    responseTime: 0,
    lastActivity: ''
  })

  // Use enhanced chat hook
  const {
    messages,
    isLoading,
    connectionStatus,
    sendMessage,
    clearChat,
    testConnection,
    testAI
  } = useChat(selectedProjectId || undefined)

  // Refs
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLTextAreaElement>(null)

  // Auto-scroll to bottom
  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [])

  useEffect(() => {
    scrollToBottom()
  }, [messages, scrollToBottom])

  // Load projects on mount
  useEffect(() => {
    initializeChat()
  }, [])

  const initializeChat = async () => {
    console.log('🚀 Initializing chat interface...')
    console.log('🧪 Starting API connectivity test...')
    
    try {
      // Test connection using ApiService
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
        
        toast.success('Connected to RagFlow backend!', { duration: 2000 })
      } else {
        throw new Error('Backend unhealthy')
      }
    } catch (error) {
      console.error('💥 Initialization failed:', error)
      toast.error('Failed to initialize chat. Please refresh the page.')
    }
  }

  // Copy message function
  const copyMessage = async (messageId: string, content: string) => {
    try {
      await navigator.clipboard.writeText(content)
      setCopiedMessageId(messageId)
      toast.success('Message copied to clipboard!', { duration: 1500 })
      setTimeout(() => setCopiedMessageId(null), 2000)
    } catch (error) {
      toast.error('Failed to copy message')
    }
  }

  // Handle message sending
  const handleSendMessage = async () => {
    const messageContent = input.trim()
    if (!messageContent || isLoading || connectionStatus === 'disconnected') return

    // Use the chat hook's sendMessage function
    await sendMessage(messageContent)
    
    // Clear input and update stats
    setInput('')
    setStats(prev => ({
      ...prev,
      totalMessages: prev.totalMessages + 1,
      lastActivity: new Date().toISOString()
    }))
  }

  // Handle Enter key in textarea
  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSendMessage()
    }
  }

  // Handle input changes with auto-resize
  const handleInputChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    const value = e.target.value
    if (value.length <= 1000) {
      setInput(value)
      
      // Auto-resize textarea
      if (inputRef.current) {
        inputRef.current.style.height = 'auto'
        inputRef.current.style.height = `${Math.min(inputRef.current.scrollHeight, 120)}px`
      }
    }
  }

  // Project selector
  const ProjectSelector = () => (
    <div className="bg-white border-b border-gray-200 p-4">
      <div className="flex items-center justify-between max-w-4xl mx-auto">
        <div className="flex items-center gap-3">
          <Database className="w-5 h-5 text-blue-600" />
          <div>
            <h3 className="font-medium text-gray-900">Select Project</h3>
            <p className="text-sm text-gray-600">Choose a project to chat with its documents</p>
          </div>
        </div>
        
        <div className="flex items-center gap-3">
          {/* Project Selection */}
          <select
            value={selectedProjectId || ''}
            onChange={(e) => setSelectedProjectId(e.target.value || null)}
            className="px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent bg-white text-sm"
          >
            <option value="">General Chat (No Project)</option>
            {projects.map((project) => (
              <option key={project.id} value={project.id}>
                {project.name} ({project.document_count} docs)
              </option>
            ))}
          </select>

          {/* Connection Actions */}
          <div className="flex items-center gap-2">
            <button
              onClick={testConnection}
              className="p-2 text-gray-600 hover:text-gray-900 rounded-lg hover:bg-gray-100 transition-colors"
              title="Test Connection"
            >
              <RefreshCw className="w-4 h-4" />
            </button>
            
            <button
              onClick={clearChat}
              className="p-2 text-gray-600 hover:text-gray-900 rounded-lg hover:bg-gray-100 transition-colors"
              title="Clear Chat"
            >
              <Trash2 className="w-4 h-4" />
            </button>

            {/* Connection Status */}
            <div className={`flex items-center gap-1 px-2 py-1 rounded text-xs font-medium ${
              connectionStatus === 'connected' 
                ? 'bg-green-100 text-green-700' 
                : connectionStatus === 'disconnected'
                ? 'bg-red-100 text-red-700'
                : 'bg-yellow-100 text-yellow-700'
            }`}>
              <div className={`w-2 h-2 rounded-full ${
                connectionStatus === 'connected' 
                  ? 'bg-green-500' 
                  : connectionStatus === 'disconnected'
                  ? 'bg-red-500'
                  : 'bg-yellow-500'
              }`} />
              {connectionStatus === 'connected' ? 'Connected' : 
               connectionStatus === 'disconnected' ? 'Disconnected' : 'Checking...'}
            </div>
          </div>
        </div>
      </div>
    </div>
  )

  // Message Component
  const MessageComponent: React.FC<{ message: any }> = ({ message }) => {
    const isUser = message.role === 'user'
    const isError = message.role === 'error'

    return (
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3 }}
        className={`flex gap-4 p-4 rounded-lg ${
          isUser ? 'bg-blue-50' : isError ? 'bg-red-50' : 'bg-white'
        }`}
      >
        {/* Avatar */}
        <div className={`w-8 h-8 rounded-full flex items-center justify-center text-white text-sm font-medium flex-shrink-0 ${
          isUser ? 'bg-blue-500' : isError ? 'bg-red-500' : 'bg-gray-700'
        }`}>
          {isUser ? <User className="w-4 h-4" /> : isError ? <AlertCircle className="w-4 h-4" /> : <Bot className="w-4 h-4" />}
        </div>

        {/* Content */}
        <div className="flex-1 space-y-2 min-w-0">
          <div className="flex items-center gap-2">
            <span className="font-medium text-sm text-gray-900">
              {isUser ? 'You' : isError ? 'Error' : 'Assistant'}
            </span>
            <span className="text-xs text-gray-500">
              {new Date(message.timestamp).toLocaleTimeString()}
            </span>
          </div>

          {/* Message Content */}
          <div className="prose prose-sm max-w-none">
            <p className={`whitespace-pre-wrap break-words ${
              isError ? 'text-red-800' : 'text-gray-800'
            }`}>
              {message.content}
            </p>
          </div>

          {/* Sources */}
          {message.sources && message.sources.length > 0 && (
            <div className="mt-3 space-y-2">
              <div className="flex items-center gap-2 text-sm text-gray-600">
                <FileText className="w-4 h-4" />
                <span>Found {message.sources.length} relevant sources</span>
              </div>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                {message.sources.slice(0, 4).map((source: any, idx: number) => (
                  <div key={idx} className="text-xs bg-blue-50 border border-blue-200 p-3 rounded-lg">
                    <div className="flex items-center gap-2 mb-1">
                      <Database className="w-3 h-3 text-blue-600" />
                      <span className="font-medium text-blue-800">
                        {source.filename || source.name || `Document ${idx + 1}`}
                      </span>
                      {source.relevance_score && (
                        <span className="text-blue-600 bg-blue-100 px-1 rounded">
                          {Math.round(source.relevance_score * 100)}%
                        </span>
                      )}
                    </div>
                    <p className="text-gray-700 line-clamp-2">
                      {source.excerpt || source.content || 'Relevant content found'}
                    </p>
                  </div>
                ))}
              </div>
              {message.sources.length > 4 && (
                <p className="text-xs text-gray-600">
                  ... and {message.sources.length - 4} more sources
                </p>
              )}
            </div>
          )}

          {/* Intelligence Metadata */}
          {message.intelligence_metadata && (
            <div className="mt-2 flex items-center gap-2 text-xs text-purple-600 bg-purple-50 px-2 py-1 rounded">
              <Sparkles className="w-3 h-3" />
              <span>
                {message.intelligence_metadata.query_complexity} complexity • {message.intelligence_metadata.reasoning_depth} reasoning
              </span>
            </div>
          )}

          {/* Actions */}
          <div className="flex items-center gap-2 mt-2">
            <button
              onClick={() => copyMessage(message.id, message.content)}
              className="text-xs text-gray-500 hover:text-gray-700 flex items-center gap-1 px-2 py-1 rounded hover:bg-gray-100 transition-colors"
            >
              {copiedMessageId === message.id ? (
                <Check className="w-3 h-3" />
              ) : (
                <Copy className="w-3 h-3" />
              )}
              {copiedMessageId === message.id ? 'Copied!' : 'Copy'}
            </button>
          </div>
        </div>
      </motion.div>
    )
  }

  return (
    <div className="h-full flex flex-col bg-gray-50">
      {/* Project Selector Header */}
      <ProjectSelector />

      {/* Messages Area */}
      <div className="flex-1 overflow-hidden relative">
        <div className="h-full overflow-y-auto pb-32" ref={messagesEndRef}>
          <div className="max-w-4xl mx-auto p-4 space-y-4">
            {/* Welcome Message */}
            {messages.length === 0 && (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className="text-center py-12"
              >
                <div className="w-16 h-16 bg-gradient-to-br from-blue-500 to-purple-600 rounded-full mx-auto mb-4 flex items-center justify-center">
                  <MessageSquare className="w-8 h-8 text-white" />
                </div>
                <h3 className="text-xl font-semibold text-gray-900 mb-2">
                  Welcome to RagFlow Chat
                </h3>
                <p className="text-gray-600 max-w-lg mx-auto mb-6">
                  {selectedProjectId 
                    ? "Ask questions about your documents and get AI-powered answers with source references."
                    : "Select a project above to chat with your documents, or ask general questions."
                  }
                </p>
                <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 text-left max-w-lg mx-auto">
                  <h4 className="font-medium text-blue-900 mb-2">💡 Tips for better results:</h4>
                  <ul className="text-sm text-blue-800 space-y-1">
                    <li>• Be specific in your questions</li>
                    <li>• Ask about document content for source-backed answers</li>
                    <li>• Use follow-up questions for deeper insights</li>
                  </ul>
                </div>
              </motion.div>
            )}

            {/* Messages */}
            <AnimatePresence>
              {messages.map((message) => (
                <MessageComponent key={message.id} message={message} />
              ))}
            </AnimatePresence>
          </div>
        </div>

        {/* Connection Warning */}
        {connectionStatus === 'disconnected' && (
          <div className="absolute top-4 left-4 right-4 bg-red-50 border border-red-200 rounded-lg p-3 flex items-center gap-3">
            <WifiOff className="w-5 h-5 text-red-600" />
            <div className="flex-1">
              <p className="text-red-800 font-medium">Backend Disconnected</p>
              <p className="text-red-600 text-sm">Please check your connection and restart the server.</p>
            </div>
            <button
              onClick={testConnection}
              className="px-3 py-1 bg-red-600 text-white rounded text-sm hover:bg-red-700 transition-colors"
            >
              Retry
            </button>
          </div>
        )}
      </div>

      {/* Input Area */}
      <div className="absolute bottom-0 left-0 right-0 bg-white border-t border-gray-200 p-4 shadow-lg">
        <div className="max-w-4xl mx-auto">
          <div className="flex gap-3 items-end">
            {/* Text Input */}
            <div className="flex-1 relative">
              <textarea
                ref={inputRef}
                value={input}
                onChange={handleInputChange}
                onKeyPress={handleKeyPress}
                placeholder={
                  connectionStatus === 'disconnected' 
                    ? "Backend disconnected - please check connection" 
                    : selectedProjectId 
                      ? "Ask a question about your documents..." 
                      : "Ask a question (select a project for document-specific answers)..."
                }
                disabled={isLoading || connectionStatus === 'disconnected'}
                className="w-full resize-none border border-gray-300 rounded-lg px-4 py-3 pr-16 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:bg-gray-100 disabled:cursor-not-allowed"
                rows={1}
                style={{ minHeight: '44px', maxHeight: '120px' }}
              />
              
              {/* Character count */}
              <div className="absolute bottom-2 right-2 text-xs text-gray-400">
                {input.length}/1000
              </div>
            </div>

            {/* Send Button */}
            <button
              onClick={handleSendMessage}
              disabled={!input.trim() || input.length > 1000 || isLoading || connectionStatus === 'disconnected'}
              className={`px-6 py-3 rounded-lg font-medium transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed flex-shrink-0 shadow-sm ${
                isLoading 
                  ? 'bg-gray-100 text-gray-400' 
                  : 'bg-gradient-to-r from-blue-500 to-purple-600 hover:from-blue-600 hover:to-purple-700 text-white shadow-md hover:shadow-lg'
              }`}
            >
              {isLoading ? (
                <div className="flex items-center gap-2">
                  <Loader2 className="w-4 h-4 animate-spin" />
                  <span>Sending...</span>
                </div>
              ) : (
                <div className="flex items-center gap-2">
                  <Send className="w-4 h-4" />
                  <span>Send</span>
                </div>
              )}
            </button>
          </div>

          {/* Input Hints */}
          <div className="flex items-center justify-between mt-3 text-xs text-gray-500">
            <div className="flex items-center gap-4">
              {connectionStatus === 'connected' && (
                <div className="flex items-center gap-1">
                  <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                  <span>Backend connected</span>
                </div>
              )}
              {selectedProjectId && (
                <div className="flex items-center gap-1">
                  <FileText className="w-3 h-3" />
                  <span>Using project: {projects.find(p => p.id === selectedProjectId)?.name}</span>
                </div>
              )}
              <span>💡 Be specific for better results</span>
            </div>
            <div className="flex items-center gap-2">
              <kbd className="px-1.5 py-0.5 bg-gray-100 border border-gray-300 rounded text-xs">Enter</kbd>
              <span>to send</span>
              <span>•</span>
              <kbd className="px-1.5 py-0.5 bg-gray-100 border border-gray-300 rounded text-xs">Shift+Enter</kbd>
              <span>for new line</span>
            </div>
          </div>

          {/* AI Status Indicator */}
          {connectionStatus === 'connected' && (
            <div className="mt-2 flex items-center justify-center">
              <div className="flex items-center gap-1 text-xs text-green-600 bg-green-50 px-2 py-1 rounded">
                <div className="w-1.5 h-1.5 bg-green-500 rounded-full animate-pulse"></div>
                <span>AI Ready</span>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export default ChatInterface