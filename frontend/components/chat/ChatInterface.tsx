// frontend/components/chat/ChatInterface.tsx
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
import { ChatAPI, ProjectAPI, handleAPIError, withErrorHandling, testAPIConnection } from '@/lib/api'

// Enhanced Types
interface ChatMessage {
  id: string
  content: string
  role: 'user' | 'assistant' | 'error'
  timestamp: string
  sources?: Array<{
    document_id: string
    content: string
    score: number
    metadata: Record<string, any>
  }>
  status?: 'sending' | 'sent' | 'error'
  error?: string
}

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
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [projects, setProjects] = useState<Project[]>([])
  const [selectedProjectId, setSelectedProjectId] = useState<string | null>(null)
  const [copiedMessageId, setCopiedMessageId] = useState<string | null>(null)
  const [connectionStatus, setConnectionStatus] = useState<'checking' | 'connected' | 'disconnected'>('checking')
  const [stats, setStats] = useState<ChatStats>({
    totalMessages: 0,
    responseTime: 0,
    lastActivity: ''
  })

  // Refs
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLTextAreaElement>(null)
  const startTimeRef = useRef<number>(0)
  const messagesContainerRef = useRef<HTMLDivElement>(null)

  // Auto-scroll to bottom
  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [])

  useEffect(() => {
    scrollToBottom()
  }, [messages, scrollToBottom])

  // Load projects and test connection on mount
  useEffect(() => {
    initializeChat()
  }, [])

  const initializeChat = async () => {
    console.log('🚀 Initializing chat interface...')
    
    // Test API connection first
    setConnectionStatus('checking')
    try {
      const testResults = await testAPIConnection()
      
      if (testResults.health) {
        setConnectionStatus('connected')
        console.log('✅ Backend connection established')
        
        // Load projects if available
        if (testResults.projects) {
          const projectsData = testResults.details.projects as Project[]
          setProjects(projectsData)
          console.log(`📁 Loaded ${projectsData.length} projects`)
          
          // Auto-select first project if available
          if (projectsData.length > 0) {
            setSelectedProjectId(projectsData[0].id)
            console.log(`🎯 Auto-selected project: ${projectsData[0].name}`)
          }
        }
        
        // Show connection status
        toast.success('Connected to RagFlow backend!', { duration: 2000 })
      } else {
        setConnectionStatus('disconnected')
        console.warn('❌ Backend connection failed')
        toast.error('Backend connection failed. Please check if the server is running.')
      }
    } catch (error) {
      setConnectionStatus('disconnected')
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

  // Clear all messages
  const clearMessages = () => {
    setMessages([])
    setStats(prev => ({ ...prev, totalMessages: 0 }))
    toast.success('Chat cleared!', { duration: 1500 })
  }

  // Send message function
  const sendMessage = async () => {
    const messageContent = input.trim()
    if (!messageContent || isLoading || connectionStatus === 'disconnected') return

    const userMessage: ChatMessage = {
      id: Date.now().toString(),
      content: messageContent,
      role: 'user',
      timestamp: new Date().toISOString()
    }

    // Add user message and clear input
    setMessages(prev => [...prev, userMessage])
    setInput('')
    setIsLoading(true)
    startTimeRef.current = Date.now()

    try {
      const response = await ChatAPI.sendMessage({
        message: messageContent,
        project_id: selectedProjectId
      })

      if (response.success && response.data) {
        const responseTime = Date.now() - startTimeRef.current
        
        const assistantMessage: ChatMessage = {
          id: (Date.now() + 1).toString(),
          content: response.data.response || 'Sorry, I could not process your request.',
          role: 'assistant',
          timestamp: new Date().toISOString(),
          sources: response.data.sources || [],
          status: 'sent'
        }

        setMessages(prev => [...prev, assistantMessage])
        setStats(prev => ({
          totalMessages: prev.totalMessages + 2,
          responseTime,
          lastActivity: new Date().toISOString()
        }))

        // Show success indicators
        if (response.data.sources && response.data.sources.length > 0) {
          toast.success(`Found ${response.data.sources.length} relevant sources`, { duration: 2000 })
        }

      } else {
        throw new Error(response.error || 'Unknown error occurred')
      }

    } catch (error) {
      console.error('Chat error:', error)
      
      const errorMessage: ChatMessage = {
        id: (Date.now() + 1).toString(),
        content: error instanceof Error ? error.message : 'Failed to send message. Please try again.',
        role: 'error',
        timestamp: new Date().toISOString(),
        status: 'error'
      }

      setMessages(prev => [...prev, errorMessage])
      toast.error('Failed to send message')
    } finally {
      setIsLoading(false)
    }
  }

  // Handle Enter key in textarea
  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendMessage()
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

  // Message Component
  const MessageComponent: React.FC<{ message: ChatMessage }> = ({ message }) => {
    const isUser = message.role === 'user'
    const isError = message.role === 'error'
    const isSending = message.status === 'sending'

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
          {isUser ? <User className="w-4 h-4" /> : <Bot className="w-4 h-4" />}
        </div>

        {/* Content */}
        <div className="flex-1 space-y-2 min-w-0">
          <div className="flex items-center gap-2">
            <span className="font-medium text-sm text-gray-900">
              {isUser ? 'You' : 'Assistant'}
            </span>
            <span className="text-xs text-gray-500">
              {new Date(message.timestamp).toLocaleTimeString()}
            </span>
            {isSending && (
              <div className="flex items-center gap-1 text-xs text-blue-600">
                <Loader2 className="w-3 h-3 animate-spin" />
                Thinking...
              </div>
            )}
            {isError && (
              <div className="flex items-center gap-1 text-xs text-red-600">
                <AlertCircle className="w-3 h-3" />
                Error
              </div>
            )}
          </div>

          {/* Message Content */}
          <div className="prose prose-sm max-w-none">
            <p className="text-gray-800 whitespace-pre-wrap break-words">{message.content}</p>
          </div>

          {/* Sources */}
          {message.sources && message.sources.length > 0 && (
            <div className="mt-3 space-y-2">
              <div className="flex items-center gap-2 text-sm text-gray-600">
                <FileText className="w-4 h-4" />
                <span>Sources ({message.sources.length})</span>
              </div>
              <div className="flex flex-wrap gap-2">
                {message.sources.map((source, idx) => (
                  <div key={idx} className="text-xs bg-gray-100 px-2 py-1 rounded flex items-center gap-1">
                    <Database className="w-3 h-3" />
                    Document {idx + 1}
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Actions */}
          {!isSending && (
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
          )}
        </div>
      </motion.div>
    )
  }

  return (
    <div className="h-full flex flex-col bg-gray-50 relative">
      {/* Header */}
      <div className="flex-shrink-0 bg-white border-b border-gray-200 p-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-gradient-to-br from-blue-500 to-purple-600 rounded-lg flex items-center justify-center">
              <MessageSquare className="w-5 h-5 text-white" />
            </div>
            <div>
              <h1 className="text-lg font-semibold text-gray-900">AI Chat</h1>
              <p className="text-sm text-gray-600">
                {connectionStatus === 'connected' 
                  ? `${messages.length} messages • ${selectedProjectId ? projects.find(p => p.id === selectedProjectId)?.name || 'No project' : 'No project selected'}`
                  : 'Connecting to backend...'
                }
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2">
            {/* Project Selector */}
            {projects.length > 0 && (
              <select
                value={selectedProjectId || ''}
                onChange={(e) => setSelectedProjectId(e.target.value || null)}
                className="text-sm border border-gray-300 rounded-md px-3 py-1 focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value="">No project</option>
                {projects.map(project => (
                  <option key={project.id} value={project.id}>
                    {project.name} ({project.document_count} docs)
                  </option>
                ))}
              </select>
            )}

            {/* Clear Messages */}
            <button
              onClick={clearMessages}
              disabled={messages.length === 0}
              className="p-2 text-gray-500 hover:text-gray-700 hover:bg-gray-100 rounded-md disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              title="Clear messages"
            >
              <Trash2 className="w-4 h-4" />
            </button>

            {/* Connection Status */}
            <div className={`flex items-center gap-1 px-2 py-1 rounded text-xs font-medium ${
              connectionStatus === 'connected' 
                ? 'bg-green-100 text-green-700' 
                : connectionStatus === 'checking'
                  ? 'bg-yellow-100 text-yellow-700'
                  : 'bg-red-100 text-red-700'
            }`}>
              {connectionStatus === 'connected' ? (
                <>
                  <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
                  Online
                </>
              ) : connectionStatus === 'checking' ? (
                <>
                  <Loader2 className="w-3 h-3 animate-spin" />
                  Connecting
                </>
              ) : (
                <>
                  <WifiOff className="w-3 h-3" />
                  Offline
                </>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Messages Area - Mit Platz für fixen Input */}
      <div 
        ref={messagesContainerRef}
        className="flex-1 overflow-y-auto custom-scrollbar p-4 space-y-4"
        style={{ paddingBottom: '200px' }} // Platz für fixen Input
      >
        <AnimatePresence>
          {messages.length === 0 ? (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="flex items-center justify-center text-center min-h-96"
            >
              <div className="max-w-md mx-auto">
                <div className="w-16 h-16 bg-gradient-to-br from-blue-500 to-purple-600 rounded-2xl flex items-center justify-center mx-auto mb-4">
                  <Sparkles className="w-8 h-8 text-white" />
                </div>
                <h3 className="text-xl font-semibold text-gray-900 mb-2">
                  Welcome to AI Chat
                </h3>
                <p className="text-gray-600 mb-6">
                  {selectedProjectId 
                    ? "Ask questions about your documents and get AI-powered answers with source references."
                    : "Select a project above to chat with your documents, or ask general questions."
                  }
                </p>
                <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 text-left">
                  <h4 className="font-medium text-blue-900 mb-2">💡 Tips for better results:</h4>
                  <ul className="text-sm text-blue-800 space-y-1">
                    <li>• Be specific in your questions</li>
                    <li>• Ask about document content for source-backed answers</li>
                    <li>• Use follow-up questions for deeper insights</li>
                  </ul>
                </div>
              </div>
            </motion.div>
          ) : (
            messages.map((message) => (
              <MessageComponent key={message.id} message={message} />
            ))
          )}
        </AnimatePresence>
        <div ref={messagesEndRef} />
      </div>

      {/* Input Area - Fixed nur im Chat-Bereich, nicht über die Sidebar */}
      <div className="absolute bottom-0 left-0 right-0 bg-white border-t border-gray-200 p-4 shadow-lg z-40">
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
                className="w-full resize-none border border-gray-300 rounded-lg px-4 py-3 pr-16 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:bg-gray-100 disabled:cursor-not-allowed custom-scrollbar shadow-sm"
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
              onClick={sendMessage}
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

          {/* API Status Indicator */}
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