import React, { useState, useRef, useEffect, useCallback, useMemo } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { 
  Send, 
  StopCircle, 
  Sparkles, 
  Copy, 
  Check, 
  MessageSquare,
  FileText,
  User,
  Bot,
  AlertCircle,
  RefreshCw,
  Zap,
  CheckCircle,
  Clock,
  WifiOff
} from 'lucide-react'
import { Button } from "@/components/ui/Button"
import { Badge } from "@/components/ui/Badge"
import { cn } from '@/lib/utils'
import { useStore } from '@/stores/useStore'
import toast from 'react-hot-toast'

// ===== ENHANCED TYPES =====
interface Source {
  id: string
  name: string
  excerpt: string
  relevance_score: number
  filename?: string
  document_id?: string
}

interface Message {
  id: string
  content: string
  role: 'user' | 'assistant'
  timestamp: string
  sources?: Source[]
  status?: 'sending' | 'sent' | 'error'
  model_info?: {
    model: string
    features_used?: Record<string, boolean>
  }
  intelligence_metadata?: {
    context_length: number
    sources_found: number
    processing_time: number
  }
}

interface ChatResponse {
  response: string
  chat_id?: string
  timestamp: string
  project_id?: string
  sources?: Source[]
  success: boolean
  model_info?: {
    model: string
    temperature: number
    features_used: Record<string, boolean>
  }
  intelligence_metadata?: {
    context_length: number
    sources_found: number
    processing_time: number
  }
  error?: string
}

// ===== OPTIMIZED CHAT API =====
class ChatAPI {
  private static readonly BASE_URL = 'http://localhost:8000'
  private static controller: AbortController | null = null

  static cancelRequest() {
    if (this.controller) {
      this.controller.abort()
      this.controller = null
    }
  }

  static async sendMessage(content: string, projectId?: string): Promise<ChatResponse> {
    console.log('🚀 Sending chat message:', { content: content.substring(0, 50) + '...', projectId })
    
    // Cancel any existing request
    this.cancelRequest()
    this.controller = new AbortController()
    
    const payload = {
      message: content,
      project_id: projectId || null,
      use_documents: !!projectId
    }
    
    console.log('📦 Chat API payload:', payload)
    
    try {
      const response = await fetch(`${this.BASE_URL}/api/v1/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload),
        signal: this.controller.signal
      })
      
      console.log('📊 Chat response status:', response.status)
      
      if (!response.ok) {
        const errorText = await response.text()
        console.error('❌ Chat API error:', errorText)
        throw new Error(`Chat failed: ${response.status} ${response.statusText}`)
      }
      
      const result = await response.json()
      console.log('✅ Chat API result:', {
        responseLength: result.response?.length || 0,
        sourcesCount: result.sources?.length || 0,
        hasMetadata: !!result.intelligence_metadata
      })
      return result
    } catch (error) {
      if (error instanceof Error && error.name === 'AbortError') {
        throw new Error('Request was cancelled')
      }
      console.error('❌ Chat API error:', error)
      throw error
    }
  }
  
  static async getChatHistory(projectId?: string): Promise<any> {
    const params = new URLSearchParams()
    if (projectId) {
      params.append('project_id', projectId)
    }
    
    const response = await fetch(`${this.BASE_URL}/api/v1/chats/?${params}`)
    
    if (!response.ok) {
      throw new Error(`Failed to fetch chat history: ${response.statusText}`)
    }
    
    return response.json()
  }

  static async healthCheck(): Promise<boolean> {
    try {
      const response = await fetch(`${this.BASE_URL}/api/health`, {
        method: 'GET',
        signal: AbortSignal.timeout(5000)
      })
      return response.ok
    } catch {
      return false
    }
  }
}

// ===== ENHANCED MESSAGE COMPONENT =====
const MessageComponent = React.memo(({ 
  message, 
  onRetry 
}: { 
  message: Message
  onRetry?: (messageId: string) => void 
}) => {
  const [copied, setCopied] = useState(false)
  const [expanded, setExpanded] = useState(false)
  const isUser = message.role === 'user'

  const copyToClipboard = useCallback(async () => {
    try {
      await navigator.clipboard.writeText(message.content)
      setCopied(true)
      toast.success('📋 Copied to clipboard!')
      setTimeout(() => setCopied(false), 2000)
    } catch {
      toast.error('Failed to copy text')
    }
  }, [message.content])

  const getStatusIcon = useMemo(() => {
    switch (message.status) {
      case 'sending':
        return <Clock size={12} className="text-blue-500 animate-pulse" />
      case 'sent':
        return <CheckCircle size={12} className="text-green-500" />
      case 'error':
        return <AlertCircle size={12} className="text-red-500" />
      default:
        return null
    }
  }, [message.status])

  const shouldTruncate = message.content.length > 800
  const displayContent = useMemo(() => {
    if (!shouldTruncate || expanded) return message.content
    return message.content.slice(0, 800) + '...'
  }, [message.content, expanded, shouldTruncate])

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -20 }}
      transition={{ duration: 0.3, ease: "easeOut" }}
      className={cn(
        "flex w-full group",
        isUser ? "justify-end" : "justify-start"
      )}
    >
      <div className={cn(
        "max-w-[80%] rounded-2xl px-4 py-3 shadow-sm relative",
        isUser 
          ? "bg-blue-600 text-white" 
          : "bg-white border border-gray-200 text-gray-900"
      )}>
        {/* Enhanced Message Header */}
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center gap-2">
            <div className={cn(
              "w-6 h-6 rounded-full flex items-center justify-center",
              isUser ? "bg-blue-500" : "bg-gray-100"
            )}>
              {isUser ? (
                <User size={12} className="text-white" />
              ) : (
                <Bot size={12} className="text-gray-600" />
              )}
            </div>
            <span className={cn(
              "text-xs font-medium",
              isUser ? "text-blue-100" : "text-gray-600"
            )}>
              {isUser ? "You" : "AI Assistant"}
            </span>
            <span className={cn(
              "text-xs",
              isUser ? "text-blue-200" : "text-gray-400"
            )}>
              {new Date(message.timestamp).toLocaleTimeString('en-US', {
                hour: '2-digit',
                minute: '2-digit'
              })}
            </span>
            {getStatusIcon}
          </div>
          
          {/* Enhanced Action Buttons */}
          <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
            <button
              onClick={copyToClipboard}
              className={cn(
                "p-1 rounded transition-colors hover:scale-110",
                isUser
                  ? 'text-blue-200 hover:text-white hover:bg-blue-700' 
                  : 'text-gray-400 hover:text-gray-600 hover:bg-gray-100'
              )}
              title="Copy message"
            >
              {copied ? <Check size={14} /> : <Copy size={14} />}
            </button>
            
            {message.status === 'error' && onRetry && (
              <button
                onClick={() => onRetry(message.id)}
                className="p-1 rounded transition-colors hover:bg-red-100 text-red-500 hover:scale-110"
                title="Retry message"
              >
                <RefreshCw size={14} />
              </button>
            )}
          </div>
        </div>

        {/* Enhanced Message Content */}
        <div className={cn(
          "text-sm leading-relaxed whitespace-pre-wrap break-words",
          isUser ? "text-white" : "text-gray-900"
        )}>
          {displayContent}
          
          {shouldTruncate && (
            <button
              onClick={() => setExpanded(!expanded)}
              className={cn(
                "ml-2 text-xs underline hover:no-underline transition-colors",
                isUser ? "text-blue-200 hover:text-white" : "text-blue-600 hover:text-blue-800"
              )}
            >
              {expanded ? 'Show less' : 'Show more'}
            </button>
          )}
        </div>

        {/* Enhanced Intelligence Metadata Display */}
        {message.intelligence_metadata && (
          <motion.div 
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            className="mt-3 p-2 bg-blue-50 rounded-lg border border-blue-200"
          >
            <div className="flex items-center gap-2 text-xs">
              <Zap size={12} className="text-blue-600" />
              <span className="font-medium text-blue-900">Enhanced AI Processing</span>
            </div>
            <div className="mt-1 text-xs text-blue-700 grid grid-cols-3 gap-2">
              <div>Context: {message.intelligence_metadata.context_length} chars</div>
              <div>Sources: {message.intelligence_metadata.sources_found}</div>
              <div>Time: {message.intelligence_metadata.processing_time}s</div>
            </div>
          </motion.div>
        )}

        {/* Enhanced Sources Display */}
        {message.sources && message.sources.length > 0 && (
          <motion.div 
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            className="mt-3 space-y-2"
          >
            <div className="text-xs font-medium text-gray-600 flex items-center gap-1">
              <FileText size={12} />
              Sources ({message.sources.length}):
            </div>
            <div className="space-y-2 max-h-32 overflow-y-auto">
              {message.sources.slice(0, 3).map((source, index) => (
                <motion.div
                  key={source.id || index}
                  initial={{ opacity: 0, x: -10 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: index * 0.1 }}
                  className="p-3 bg-gray-50 rounded-lg border border-gray-200 text-sm hover:bg-gray-100 transition-colors"
                >
                  <div className="flex items-center justify-between mb-1">
                    <span className="font-medium text-gray-900 truncate flex items-center gap-1">
                      <FileText size={10} />
                      {source.filename || source.name}
                    </span>
                    <div className="flex items-center gap-1">
                      <span className="text-xs text-gray-500">
                        {Math.round(source.relevance_score * 100)}%
                      </span>
                      <div className={cn(
                        "w-2 h-2 rounded-full",
                        source.relevance_score > 0.8 ? "bg-green-500" :
                        source.relevance_score > 0.6 ? "bg-yellow-500" : "bg-gray-400"
                      )} />
                    </div>
                  </div>
                  <p className="text-gray-600 text-xs line-clamp-2">
                    {source.excerpt}
                  </p>
                </motion.div>
              ))}
              
              {message.sources.length > 3 && (
                <div className="text-xs text-gray-500 text-center py-1">
                  +{message.sources.length - 3} more sources available
                </div>
              )}
            </div>
          </motion.div>
        )}
      </div>
    </motion.div>
  )
})

MessageComponent.displayName = 'MessageComponent'

// ===== ENHANCED TYPING INDICATOR =====
const TypingIndicator = React.memo(() => (
  <motion.div 
    initial={{ opacity: 0, y: 20 }}
    animate={{ opacity: 1, y: 0 }}
    exit={{ opacity: 0, y: -20 }}
    className="flex justify-start"
  >
    <div className="bg-white border border-gray-200 rounded-2xl px-4 py-3 shadow-sm">
      <div className="flex items-center gap-3">
        <Bot size={16} className="text-blue-600" />
        <div className="flex space-x-1">
          {[0, 1, 2].map((i) => (
            <motion.div
              key={i}
              className="w-2 h-2 bg-blue-500 rounded-full"
              animate={{
                scale: [1, 1.5, 1],
                opacity: [0.5, 1, 0.5]
              }}
              transition={{
                duration: 1.2,
                repeat: Infinity,
                delay: i * 0.2
              }}
            />
          ))}
        </div>
        <span className="text-xs text-gray-500">AI is analyzing your request...</span>
      </div>
    </div>
  </motion.div>
))

TypingIndicator.displayName = 'TypingIndicator'

// ===== ENHANCED WELCOME SCREEN =====
const WelcomeScreen = React.memo(({ 
  onSuggestionClick 
}: { 
  onSuggestionClick: (suggestion: string) => void 
}) => {
  const { currentProject } = useStore()

  const suggestions = useMemo(() => {
    const hasDocuments = currentProject?.document_count && currentProject.document_count > 0
    
    return hasDocuments ? [
      { icon: "📄", text: "Summarize my documents", category: "Analysis" },
      { icon: "🔍", text: "What are the key insights?", category: "Insights" },
      { icon: "💡", text: "Find specific information", category: "Search" },
      { icon: "📊", text: "Compare different sections", category: "Comparison" }
    ] : [
      { icon: "❓", text: "How can you help me?", category: "Help" },
      { icon: "📤", text: "What can I upload?", category: "Upload" },
      { icon: "🔧", text: "Explain how this works", category: "Tutorial" },
      { icon: "🚀", text: "Get started guide", category: "Guide" }
    ]
  }, [currentProject])

  return (
    <motion.div 
      initial={{ opacity: 0, y: 30 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6, ease: "easeOut" }}
      className="flex flex-col items-center justify-center min-h-0 text-center px-8 py-8"
    >
      {/* Enhanced Welcome Icon */}
      <motion.div 
        className="w-20 h-20 bg-gradient-to-br from-blue-600 via-purple-600 to-indigo-600 rounded-full flex items-center justify-center mb-6 shadow-xl"
        whileHover={{ scale: 1.05, rotate: 5 }}
        transition={{ type: "spring", stiffness: 300 }}
      >
        <Sparkles className="h-10 w-10 text-white" />
      </motion.div>

      {/* Enhanced Welcome Text */}
      <h2 className="text-3xl font-bold text-gray-900 mb-3">
        Welcome to RagFlow AI
      </h2>
      <p className="text-lg text-gray-600 max-w-2xl mb-8 leading-relaxed">
        Your intelligent document analysis assistant powered by advanced AI. 
        I can help you analyze documents, answer questions, and provide insights.{' '}
        <br />
        {currentProject ? (
          <span className="text-blue-600 font-medium">
            Currently working with project "{currentProject.name}" 
            ({currentProject.document_count || 0} documents available).
          </span>
        ) : (
          <span className="text-orange-600">
            Create a project and upload documents to get started!
          </span>
        )}
      </p>

      {/* Enhanced Suggestion Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 w-full max-w-2xl">
        {suggestions.map((suggestion, index) => (
          <motion.button
            key={index}
            onClick={() => onSuggestionClick(suggestion.text)}
            className="group p-4 text-left bg-white hover:bg-gradient-to-r hover:from-blue-50 hover:to-purple-50 rounded-2xl border border-gray-200 hover:border-blue-300 hover:shadow-lg transition-all duration-300"
            whileHover={{ scale: 1.02, y: -2 }}
            whileTap={{ scale: 0.98 }}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: index * 0.1 }}
          >
            <div className="flex items-center space-x-4">
              <div className="text-2xl group-hover:scale-110 transition-transform duration-300">
                {suggestion.icon}
              </div>
              <div className="flex-1">
                <span className="font-semibold text-gray-900 group-hover:text-blue-700 transition-colors text-base block">
                  {suggestion.text}
                </span>
                <span className="text-xs text-gray-500 uppercase tracking-wide">
                  {suggestion.category}
                </span>
              </div>
            </div>
          </motion.button>
        ))}
      </div>

      {/* Enhanced Project Stats */}
      {currentProject && (
        <motion.div 
          className="mt-8 grid grid-cols-2 gap-6 text-sm"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.5 }}
        >
          <div className="flex items-center gap-2 text-gray-600">
            <FileText size={16} className="text-blue-600" />
            <span className="font-medium">{currentProject.document_count || 0}</span>
            <span>documents</span>
          </div>
          <div className="flex items-center gap-2 text-gray-600">
            <MessageSquare size={16} className="text-green-600" />
            <span className="font-medium">{currentProject.chat_count || 0}</span>
            <span>conversations</span>
          </div>
        </motion.div>
      )}
    </motion.div>
  )
})

WelcomeScreen.displayName = 'WelcomeScreen'

// ===== MAIN OPTIMIZED CHAT INTERFACE =====
export function ChatInterface() {
  // State Management
  const [message, setMessage] = useState('')
  const [messages, setMessages] = useState<Message[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [connectionStatus, setConnectionStatus] = useState<'connected' | 'disconnected' | 'checking'>('checking')
  const [currentChatId, setCurrentChatId] = useState<string | null>(null)
  
  // Refs
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const textAreaRef = useRef<HTMLTextAreaElement>(null)
  const messageCounterRef = useRef(0)
  
  // Store
  const { currentProject, addChat, updateChat } = useStore()

  // ===== CONNECTION MONITORING =====
  useEffect(() => {
    let intervalId: NodeJS.Timeout

    const checkConnection = async () => {
      try {
        const isHealthy = await ChatAPI.healthCheck()
        setConnectionStatus(isHealthy ? 'connected' : 'disconnected')
      } catch {
        setConnectionStatus('disconnected')
      }
    }

    // Initial check
    checkConnection()
    
    // Periodic health checks every 30 seconds
    intervalId = setInterval(checkConnection, 30000)

    return () => {
      clearInterval(intervalId)
      ChatAPI.cancelRequest()
    }
  }, [])

  // ===== OPTIMIZED SCROLL BEHAVIOR =====
  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ 
      behavior: 'smooth',
      block: 'end'
    })
  }, [])

  useEffect(() => {
    const timeoutId = setTimeout(scrollToBottom, 100)
    return () => clearTimeout(timeoutId)
  }, [messages, scrollToBottom])

  // ===== MESSAGE HANDLING =====
  const handleSuggestionClick = useCallback((suggestion: string) => {
    setMessage(suggestion)
    textAreaRef.current?.focus()
  }, [])

  const adjustTextareaHeight = useCallback(() => {
    if (textAreaRef.current) {
      textAreaRef.current.style.height = 'auto'
      textAreaRef.current.style.height = `${Math.min(textAreaRef.current.scrollHeight, 120)}px`
    }
  }, [])

  useEffect(() => {
    adjustTextareaHeight()
  }, [message, adjustTextareaHeight])

  const handleKeyPress = useCallback((e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSendMessage()
    }
  }, [])

  const handleRetryMessage = useCallback((messageId: string) => {
    const messageIndex = messages.findIndex(m => m.id === messageId)
    if (messageIndex > 0) {
      const previousUserMessage = messages[messageIndex - 1]
      if (previousUserMessage.role === 'user') {
        // Remove error message and retry
        setMessages(prev => prev.filter(m => m.id !== messageId))
        setMessage(previousUserMessage.content)
        textAreaRef.current?.focus()
      }
    }
  }, [messages])

  const stopGeneration = useCallback(() => {
    ChatAPI.cancelRequest()
    setIsLoading(false)
    toast.info('🛑 Generation stopped')
  }, [])

  const handleSendMessage = useCallback(async () => {
    if (!message.trim() || isLoading) return

    if (connectionStatus === 'disconnected') {
      toast.error('🔌 Backend is offline. Please check your connection.')
      return
    }

    const messageContent = message.trim()
    setMessage('')
    
    // Auto-resize textarea
    if (textAreaRef.current) {
      textAreaRef.current.style.height = 'auto'
    }

    console.log('📤 Sending message with project:', currentProject)

    const userMessage: Message = {
      id: `user_${++messageCounterRef.current}_${Date.now()}`,
      content: messageContent,
      role: 'user',
      timestamp: new Date().toISOString(),
      status: 'sending'
    }

    setMessages(prev => [...prev, userMessage])
    setIsLoading(true)

    try {
      // Update message status to sent
      setMessages(prev => prev.map(msg => 
        msg.id === userMessage.id ? { ...msg, status: 'sent' } : msg
      ))

      // Send to backend with correct project_id
      const response = await ChatAPI.sendMessage(
        messageContent, 
        currentProject?.id
      )

      const aiMessage: Message = {
        id: response.chat_id || `ai_${messageCounterRef.current}_${Date.now()}`,
        content: response.response || 'Sorry, I could not generate a response.',
        role: 'assistant',
        timestamp: response.timestamp || new Date().toISOString(),
        sources: response.sources || [],
        model_info: response.model_info,
        intelligence_metadata: response.intelligence_metadata
      }

      setMessages(prev => [...prev, aiMessage])

      // Enhanced success notifications
      if (response.sources && response.sources.length > 0) {
        const uniqueFiles = [...new Set(response.sources.map(s => s.filename || s.name))]
        toast.success(
          `🔍 Found relevant content in ${uniqueFiles.length} document${uniqueFiles.length > 1 ? 's' : ''}`,
          { duration: 3000 }
        )
      }

      if (response.intelligence_metadata) {
        toast.success(
          `⚡ Enhanced AI: ${response.intelligence_metadata.context_length} chars analyzed`,
          { duration: 2000 }
        )
      }

      // Create or update chat in store
      if (currentProject) {
        if (!currentChatId) {
          const newChatData = {
            title: messageContent.slice(0, 50) + (messageContent.length > 50 ? '...' : ''),
            project_id: currentProject.id,
            messages: [userMessage, aiMessage],
            created_at: new Date().toISOString(),
            updated_at: new Date().toISOString()
          }
          
          const chatId = addChat(newChatData)
          setCurrentChatId(chatId)
        } else {
          updateChat(currentChatId, {
            messages: [...messages, userMessage, aiMessage],
            updated_at: new Date().toISOString()
          })
        }
      }

    } catch (error) {
      console.error('❌ Send message error:', error)
      
      let errorMessage = 'Sorry, there was an error processing your message.'
      
      if (error instanceof Error) {
        if (error.message.includes('Failed to fetch') || error.message.includes('Network connection failed')) {
          errorMessage = `🔗 Backend connection failed. Please ensure:
          
• Backend server is running (http://localhost:8000)
• GOOGLE_API_KEY is configured in .env file  
• No firewall is blocking the connection

Try again in a few moments.`
        } else if (error.message.includes('Request was cancelled')) {
          errorMessage = '🛑 Request was cancelled. Please try again.'
        } else if (error.message.includes('HTTP 400')) {
          errorMessage = '⚠️ Invalid request. Please try rephrasing your question.'
        } else if (error.message.includes('HTTP 500')) {
          errorMessage = '🔧 Server error. The backend encountered an issue processing your request.'
        } else {
          errorMessage = `❌ Error: ${error.message}`
        }
      }

      const errorMessage_obj: Message = {
        id: `error_${Date.now()}`,
        content: errorMessage,
        role: 'assistant',
        timestamp: new Date().toISOString(),
        status: 'error'
      }

      setMessages(prev => [...prev, errorMessage_obj])
      
      // Show error notification
      toast.error('Failed to send message. Please try again.', { duration: 4000 })
      
    } finally {
      setIsLoading(false)
    }
  }, [message, isLoading, connectionStatus, currentProject, currentChatId, messages, addChat, updateChat])

  // ===== RENDER =====
  return (
    <div className="flex flex-col h-full bg-gradient-to-br from-gray-50 to-blue-50">
      {/* Enhanced Header */}
      <div className="flex-shrink-0 bg-white border-b border-gray-200 shadow-sm">
        <div className="max-w-4xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <div className="p-2 bg-blue-100 rounded-xl">
                <MessageSquare className="h-6 w-6 text-blue-600" />
              </div>
              <div>
                <h1 className="text-xl font-bold text-gray-900">Intelligent Chat</h1>
                <p className="text-sm text-gray-600">
                  {currentProject 
                    ? `Project: ${currentProject.name} • ${currentProject.document_count || 0} documents`
                    : 'No project selected'
                  }
                </p>
              </div>
            </div>
            
            <div className="flex items-center space-x-3">
              {/* Enhanced Connection Status */}
              <div className={cn(
                "flex items-center space-x-2 px-3 py-1 rounded-full text-xs font-medium transition-all",
                connectionStatus === 'connected' 
                  ? "bg-green-100 text-green-700 border border-green-200"
                  : connectionStatus === 'disconnected'
                  ? "bg-red-100 text-red-700 border border-red-200"
                  : "bg-yellow-100 text-yellow-700 border border-yellow-200"
              )}>
                <div className={cn(
                  "w-2 h-2 rounded-full transition-all",
                  connectionStatus === 'connected' ? "bg-green-500" :
                  connectionStatus === 'disconnected' ? "bg-red-500" : "bg-yellow-500"
                )} />
                <span>
                  {connectionStatus === 'connected' ? 'Connected' :
                   connectionStatus === 'disconnected' ? 'Offline' : 'Connecting...'}
                </span>
                {connectionStatus === 'disconnected' && (
                  <WifiOff size={12} className="ml-1" />
                )}
              </div>

              {/* Enhanced Project Info */}
              {currentProject && (
                <Badge variant="secondary" className="text-xs">
                  {currentProject.document_count || 0} docs • {messages.filter(m => m.role === 'user').length} chats
                </Badge>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Messages Area */}
      <div className="flex-1 overflow-hidden">
        <div className="h-full overflow-y-auto">
          {messages.length === 0 ? (
            <WelcomeScreen onSuggestionClick={handleSuggestionClick} />
          ) : (
            <div className="max-w-4xl mx-auto p-6 space-y-6">
              <AnimatePresence mode="popLayout">
                {messages.map((msg) => (
                  <MessageComponent 
                    key={msg.id} 
                    message={msg} 
                    onRetry={handleRetryMessage}
                  />
                ))}
              </AnimatePresence>
              
              {isLoading && <TypingIndicator />}
              
              <div ref={messagesEndRef} />
            </div>
          )}
        </div>
      </div>

      {/* Enhanced Input Area */}
      <div className="flex-shrink-0 bg-white border-t border-gray-200 shadow-lg">
        <div className="max-w-4xl mx-auto p-6">
          <div className="relative">
            {/* Enhanced Input Container */}
            <div className="relative bg-gray-50 rounded-2xl border border-gray-200 focus-within:border-blue-500 focus-within:ring-2 focus-within:ring-blue-500/20 transition-all duration-200 shadow-sm hover:shadow-md">
              <textarea
                ref={textAreaRef}
                value={message}
                onChange={(e) => setMessage(e.target.value)}
                onKeyDown={handleKeyPress}
                placeholder={
                  connectionStatus === 'disconnected'
                    ? "Backend offline - please check connection..."
                    : currentProject?.document_count && currentProject.document_count > 0
                    ? "Ask a question about your documents..."
                    : "Ask me anything..."
                }
                className="min-h-[60px] max-h-[120px] py-4 pl-6 pr-20 text-base resize-none border-0 bg-transparent focus:ring-0 focus:outline-none placeholder:text-gray-500 w-full leading-relaxed"
                disabled={isLoading || connectionStatus === 'disconnected'}
                rows={1}
              />
              
              {/* Enhanced Character Counter & Status */}
              <div className="absolute bottom-2 left-3 flex items-center gap-2">
                {message.length > 0 && (
                  <span className="text-xs text-gray-400">
                    {message.length} chars
                  </span>
                )}
                {currentProject && (
                  <span className="text-xs text-blue-500 font-medium">
                    RAG enabled
                  </span>
                )}
              </div>
              
              {/* Enhanced Action Buttons */}
              <div className="absolute right-3 bottom-3 flex items-center space-x-2">
                {/* Clear Button */}
                {message.length > 0 && !isLoading && (
                  <Button
                    size="sm"
                    variant="outline"
                    onClick={() => setMessage('')}
                    className="w-8 h-8 p-0 rounded-xl border-gray-300 hover:border-gray-400 hover:scale-105 transition-all"
                    title="Clear input"
                  >
                    ×
                  </Button>
                )}
                
                {/* Send/Stop Button */}
                {isLoading ? (
                  <Button
                    size="sm"
                    onClick={stopGeneration}
                    className="w-10 h-10 p-0 rounded-xl bg-red-600 hover:bg-red-700 shadow-md hover:shadow-lg transition-all"
                    title="Stop generation"
                  >
                    <StopCircle size={16} />
                  </Button>
                ) : (
                  <Button
                    size="sm"
                    onClick={handleSendMessage}
                    disabled={!message.trim() || connectionStatus === 'disconnected'}
                    className={cn(
                      "w-10 h-10 p-0 rounded-xl transition-all duration-200 shadow-md",
                      message.trim() && connectionStatus === 'connected'
                        ? "bg-blue-600 hover:bg-blue-700 hover:shadow-lg hover:scale-105" 
                        : "bg-gray-300 cursor-not-allowed"
                    )}
                    title={
                      connectionStatus === 'disconnected' 
                        ? "Backend offline" 
                        : !message.trim() 
                        ? "Enter a message" 
                        : "Send message"
                    }
                  >
                    <Send size={16} />
                  </Button>
                )}
              </div>
            </div>
            
            {/* Enhanced Quick Actions Bar */}
            <div className="flex items-center justify-between mt-4">
              <div className="flex items-center space-x-4 text-xs text-gray-500">
                <div className="flex items-center gap-1">
                  <span>Press</span>
                  <kbd className="px-1 py-0.5 bg-gray-100 rounded text-xs">Enter</kbd>
                  <span>to send</span>
                </div>
                <div className="flex items-center gap-1">
                  <kbd className="px-1 py-0.5 bg-gray-100 rounded text-xs">Shift+Enter</kbd>
                  <span>for new line</span>
                </div>
                {currentProject && (
                  <div className="flex items-center gap-1 text-blue-600">
                    <Zap size={12} />
                    <span>Document search enabled</span>
                  </div>
                )}
              </div>
              
              <div className="flex items-center space-x-3">
                {/* Enhanced Message Stats */}
                {messages.length > 0 && (
                  <div className="text-xs text-gray-500 flex items-center gap-3">
                    <span>{messages.filter(m => m.role === 'user').length} messages</span>
                    <span>{messages.filter(m => m.sources?.length).length} with sources</span>
                    {messages.some(m => m.intelligence_metadata) && (
                      <span className="text-blue-600 flex items-center gap-1">
                        <Zap size={10} />
                        Enhanced AI
                      </span>
                    )}
                  </div>
                )}
                
                {/* Enhanced Clear Chat Button */}
                {messages.length > 0 && (
                  <Button
                    size="sm"
                    variant="outline"
                    onClick={() => {
                      setMessages([])
                      setCurrentChatId(null)
                      toast.success('🗑️ Chat cleared')
                    }}
                    className="text-xs px-3 py-1 h-7 border-gray-300 hover:border-red-300 hover:text-red-600 transition-colors"
                  >
                    Clear Chat
                  </Button>
                )}
                
                {/* Export Chat Button */}
                {messages.length > 0 && (
                  <Button
                    size="sm"
                    variant="outline"
                    onClick={() => {
                      const chatData = {
                        project: currentProject?.name || 'No Project',
                        timestamp: new Date().toISOString(),
                        messages: messages.map(m => ({
                          role: m.role,
                          content: m.content,
                          timestamp: m.timestamp,
                          sources: m.sources?.length || 0
                        }))
                      }
                      
                      const blob = new Blob([JSON.stringify(chatData, null, 2)], { 
                        type: 'application/json' 
                      })
                      const url = URL.createObjectURL(blob)
                      const a = document.createElement('a')
                      a.href = url
                      a.download = `ragflow-chat-${new Date().toISOString().split('T')[0]}.json`
                      a.click()
                      URL.revokeObjectURL(url)
                      
                      toast.success('💾 Chat exported successfully!')
                    }}
                    className="text-xs px-3 py-1 h-7 border-gray-300 hover:border-blue-300 hover:text-blue-600 transition-colors"
                  >
                    Export
                  </Button>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

// ===== ADDITIONAL PERFORMANCE HOOKS =====

// Hook for chat performance monitoring
export const useChatPerformance = () => {
  const [metrics, setMetrics] = useState({
    totalMessages: 0,
    averageResponseTime: 0,
    successRate: 0,
    sourcesFound: 0
  })

  const updateMetrics = useCallback((messages: Message[]) => {
    const userMessages = messages.filter(m => m.role === 'user').length
    const aiMessages = messages.filter(m => m.role === 'assistant').length
    const errorMessages = messages.filter(m => m.status === 'error').length
    const sourcesTotal = messages.reduce((sum, m) => sum + (m.sources?.length || 0), 0)
    
    const successRate = aiMessages > 0 ? ((aiMessages - errorMessages) / aiMessages) * 100 : 0
    
    setMetrics({
      totalMessages: messages.length,
      averageResponseTime: messages
        .filter(m => m.intelligence_metadata?.processing_time)
        .reduce((sum, m) => sum + (m.intelligence_metadata?.processing_time || 0), 0) / aiMessages || 0,
      successRate,
      sourcesFound: sourcesTotal
    })
  }, [])

  return { metrics, updateMetrics }
}

// Hook for connection monitoring
export const useConnectionMonitor = () => {
  const [status, setStatus] = useState<'connected' | 'disconnected' | 'checking'>('checking')
  const [lastCheck, setLastCheck] = useState<Date | null>(null)

  const checkConnection = useCallback(async () => {
    setStatus('checking')
    try {
      const isHealthy = await ChatAPI.healthCheck()
      setStatus(isHealthy ? 'connected' : 'disconnected')
      setLastCheck(new Date())
      return isHealthy
    } catch {
      setStatus('disconnected')
      setLastCheck(new Date())
      return false
    }
  }, [])

  useEffect(() => {
    checkConnection()
    const interval = setInterval(checkConnection, 30000)
    return () => clearInterval(interval)
  }, [checkConnection])

  return { status, lastCheck, checkConnection }
}

export default ChatInterface