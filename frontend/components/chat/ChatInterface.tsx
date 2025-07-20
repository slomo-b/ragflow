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
  role: 'user' | 'assistant'
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

  // Handle message sending
  const sendMessage = useCallback(async () => {
    if (!input.trim() || isLoading) return

    const messageContent = input.trim()
    setInput('')
    
    // Start timing for response time
    startTimeRef.current = Date.now()

    // Create user message
    const userMessage: ChatMessage = {
      id: `user-${Date.now()}`,
      content: messageContent,
      role: 'user',
      timestamp: new Date().toISOString(),
      status: 'sent'
    }

    // Create assistant message placeholder
    const assistantMessage: ChatMessage = {
      id: `assistant-${Date.now()}`,
      content: '',
      role: 'assistant',
      timestamp: new Date().toISOString(),
      status: 'sending'
    }

    setMessages(prev => [...prev, userMessage, assistantMessage])
    setIsLoading(true)

    console.log(`💬 Sending message: "${messageContent}"`)
    console.log(`📁 Selected project: ${selectedProjectId || 'None'}`)

    try {
      const response = await ChatAPI.sendMessage(messageContent, selectedProjectId || undefined)
      
      // Calculate response time
      const responseTime = Date.now() - startTimeRef.current
      
      // Update assistant message with response
      setMessages(prev => prev.map(msg => 
        msg.id === assistantMessage.id 
          ? {
              ...msg,
              content: response.response,
              sources: response.sources?.map(sourceId => ({
                document_id: sourceId,
                content: '',
                score: 0,
                metadata: {}
              })),
              status: 'sent'
            }
          : msg
      ))

      // Update stats
      setStats(prev => ({
        totalMessages: prev.totalMessages + 1,
        responseTime,
        lastActivity: new Date().toISOString()
      }))

      console.log(`✅ Message sent successfully in ${responseTime}ms`)
      
      // Show success feedback
      if (response.sources && response.sources.length > 0) {
        toast.success(`Response generated with ${response.sources.length} source(s)`)
      } else {
        toast.success('Message sent successfully')
      }

    } catch (error) {
      console.error('💥 Chat error:', error)
      
      const errorMessage = handleAPIError(error)
      
      // Update assistant message with error
      setMessages(prev => prev.map(msg => 
        msg.id === assistantMessage.id 
          ? {
              ...msg,
              content: `Sorry, I encountered an error: ${errorMessage}`,
              status: 'error',
              error: errorMessage
            }
          : msg
      ))

      toast.error(errorMessage)
    } finally {
      setIsLoading(false)
    }
  }, [input, isLoading, selectedProjectId])

  // Handle input key press
  const handleKeyPress = useCallback((e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendMessage()
    }
  }, [sendMessage])

  // Copy message content
  const copyMessage = useCallback(async (messageId: string, content: string) => {
    try {
      await navigator.clipboard.writeText(content)
      setCopiedMessageId(messageId)
      toast.success('Message copied to clipboard')
      setTimeout(() => setCopiedMessageId(null), 2000)
    } catch (error) {
      console.error('Failed to copy:', error)
      toast.error('Failed to copy message')
    }
  }, [])

  // Clear chat
  const clearChat = useCallback(() => {
    setMessages([])
    setStats(prev => ({ ...prev, totalMessages: 0 }))
    toast.success('Chat cleared')
  }, [])

  // Refresh connection
  const refreshConnection = useCallback(() => {
    initializeChat()
  }, [])

  // Message Component
  const MessageComponent: React.FC<{ message: ChatMessage; index: number }> = ({ message, index }) => {
    const isUser = message.role === 'user'
    const isError = message.status === 'error'
    const isSending = message.status === 'sending'

    return (
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: index * 0.1 }}
        className={`flex gap-4 p-4 rounded-lg ${
          isUser ? 'bg-blue-50 ml-12' : isError ? 'bg-red-50' : 'bg-white'
        }`}
      >
        {/* Avatar */}
        <div className={`w-8 h-8 rounded-full flex items-center justify-center text-white text-sm font-medium ${
          isUser ? 'bg-blue-500' : isError ? 'bg-red-500' : 'bg-gray-700'
        }`}>
          {isUser ? <User className="w-4 h-4" /> : <Bot className="w-4 h-4" />}
        </div>

        {/* Content */}
        <div className="flex-1 space-y-2">
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
            <p className="text-gray-800 whitespace-pre-wrap">{message.content}</p>
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
                className="text-xs text-gray-500 hover:text-gray-700 flex items-center gap-1 px-2 py-1 rounded hover:bg-gray-100"
              >
                {copiedMessageId === message.id ? (
                  <Check className="w-3 h-3" />
                ) : (
                  <Copy className="w-3 h-3" />
                )}
                {copiedMessageId === message.id ? 'Copied' : 'Copy'}
              </button>
            </div>
          )}
        </div>
      </motion.div>
    )
  }

  return (
    <div className="h-full flex flex-col bg-gray-50">
      {/* Header */}
      <div className="bg-white border-b border-gray-200 p-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-gradient-to-br from-blue-500 to-purple-600 rounded-lg flex items-center justify-center">
              <MessageSquare className="w-5 h-5 text-white" />
            </div>
            <div>
              <h1 className="text-xl font-semibold text-gray-900">AI Chat</h1>
              <div className="flex items-center gap-2 text-sm text-gray-600">
                <div className={`w-2 h-2 rounded-full ${
                  connectionStatus === 'connected' ? 'bg-green-500' : 
                  connectionStatus === 'checking' ? 'bg-yellow-500' : 'bg-red-500'
                }`} />
                <span>
                  {connectionStatus === 'connected' ? 'Connected' : 
                   connectionStatus === 'checking' ? 'Connecting...' : 'Disconnected'}
                </span>
                {selectedProjectId && (
                  <>
                    <span>•</span>
                    <span>{projects.find(p => p.id === selectedProjectId)?.name || 'Selected Project'}</span>
                  </>
                )}
              </div>
            </div>
          </div>

          {/* Header Actions */}
          <div className="flex items-center gap-2">
            {/* Project Selector */}
            <select
              value={selectedProjectId || ''}
              onChange={(e) => setSelectedProjectId(e.target.value || null)}
              className="text-sm border border-gray-300 rounded px-2 py-1 bg-white"
              disabled={projects.length === 0}
            >
              <option value="">No Project</option>
              {projects.map(project => (
                <option key={project.id} value={project.id}>
                  {project.name} ({project.document_count} docs)
                </option>
              ))}
            </select>

            <button
              onClick={refreshConnection}
              disabled={connectionStatus === 'checking'}
              className="p-2 text-gray-600 hover:text-gray-900 hover:bg-gray-100 rounded"
              title="Refresh connection"
            >
              <RefreshCw className={`w-4 h-4 ${connectionStatus === 'checking' ? 'animate-spin' : ''}`} />
            </button>

            <button
              onClick={clearChat}
              disabled={messages.length === 0}
              className="p-2 text-gray-600 hover:text-gray-900 hover:bg-gray-100 rounded disabled:opacity-50"
              title="Clear chat"
            >
              <Trash2 className="w-4 h-4" />
            </button>
          </div>
        </div>

        {/* Stats Bar */}
        {stats.totalMessages > 0 && (
          <div className="mt-3 flex items-center gap-4 text-xs text-gray-500">
            <div className="flex items-center gap-1">
              <MessageSquare className="w-3 h-3" />
              {stats.totalMessages} messages
            </div>
            {stats.responseTime > 0 && (
              <div className="flex items-center gap-1">
                <Zap className="w-3 h-3" />
                {stats.responseTime}ms avg
              </div>
            )}
            <div className="flex items-center gap-1">
              <Sparkles className="w-3 h-3" />
              AI-powered responses
            </div>
          </div>
        )}
      </div>

      {/* Messages Area */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {connectionStatus === 'disconnected' && (
          <div className="bg-red-50 border border-red-200 rounded-lg p-4 mb-4">
            <div className="flex items-center gap-2 text-red-700 mb-2">
              <WifiOff className="w-4 h-4" />
              <span className="font-medium">Backend Disconnected</span>
            </div>
            <p className="text-sm text-red-600 mb-3">
              Unable to connect to the RagFlow backend. Please make sure the server is running on http://localhost:8000
            </p>
            <button
              onClick={refreshConnection}
              className="text-sm bg-red-100 text-red-700 px-3 py-1 rounded hover:bg-red-200"
            >
              Retry Connection
            </button>
          </div>
        )}

        <AnimatePresence>
          {messages.length === 0 ? (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="text-center py-12"
            >
              <div className="w-16 h-16 bg-gradient-to-br from-blue-500 to-purple-600 rounded-full flex items-center justify-center mx-auto mb-6">
                <Sparkles className="w-8 h-8 text-white" />
              </div>
              <h3 className="text-lg font-medium text-gray-900 mb-2">Start a conversation</h3>
              <p className="text-gray-600 max-w-md mx-auto mb-4">
                Ask questions about your documents, get insights, or explore your knowledge base with AI.
              </p>
              
              {/* Quick Start Tips */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 max-w-2xl mx-auto mt-8">
                <div className="bg-white rounded-lg p-4 shadow-sm border border-gray-200">
                  <MessageSquare className="w-6 h-6 text-blue-500 mx-auto mb-2" />
                  <h4 className="font-medium text-gray-900 mb-1">Ask Questions</h4>
                  <p className="text-xs text-gray-600">Ask about your uploaded documents</p>
                </div>
                <div className="bg-white rounded-lg p-4 shadow-sm border border-gray-200">
                  <FileText className="w-6 h-6 text-green-500 mx-auto mb-2" />
                  <h4 className="font-medium text-gray-900 mb-1">Get Insights</h4>
                  <p className="text-xs text-gray-600">Extract key information and summaries</p>
                </div>
                <div className="bg-white rounded-lg p-4 shadow-sm border border-gray-200">
                  <Database className="w-6 h-6 text-purple-500 mx-auto mb-2" />
                  <h4 className="font-medium text-gray-900 mb-1">Search Content</h4>
                  <p className="text-xs text-gray-600">Find relevant information across documents</p>
                </div>
              </div>

              {/* Example Questions */}
              <div className="mt-8">
                <p className="text-sm text-gray-500 mb-3">Try asking:</p>
                <div className="flex flex-wrap justify-center gap-2">
                  {[
                    "What are the main topics in my documents?",
                    "Summarize the key findings",
                    "Find information about..."
                  ].map((question, idx) => (
                    <button
                      key={idx}
                      onClick={() => setInput(question)}
                      className="text-xs bg-blue-50 text-blue-600 px-3 py-1 rounded-full hover:bg-blue-100 transition-colors"
                    >
                      "{question}"
                    </button>
                  ))}
                </div>
              </div>
            </motion.div>
          ) : (
            messages.map((message, index) => (
              <MessageComponent key={message.id} message={message} index={index} />
            ))
          )}
        </AnimatePresence>
        <div ref={messagesEndRef} />
      </div>

      {/* Input Area */}
      <div className="bg-white border-t border-gray-200 p-4">
        <div className="flex gap-3">
          <div className="flex-1 relative">
            <textarea
              ref={inputRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
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
            onClick={sendMessage}
            disabled={!input.trim() || input.length > 1000 || isLoading || connectionStatus === 'disconnected'}
            className={`px-6 py-3 rounded-lg font-medium transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed ${
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
  )
}

export default ChatInterface