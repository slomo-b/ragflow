// frontend/components/chat/ChatInterface.tsx
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
  WifiOff,
  Search,
  Database,
  Brain
} from 'lucide-react'
import { Button } from "@/components/ui/Button"
import { Badge } from "@/components/ui/Badge"
import { cn } from '@/lib/utils'
import { useStore } from '@/stores/useStore'
import toast from 'react-hot-toast'

// ===== ENHANCED TYPES =====
interface Source {
  id: string
  filename: string
  excerpt: string
  relevance_score: number
  search_methods?: string[]
  type: string
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
    context_used: boolean
    context_length: number
    features_used?: Record<string, boolean>
  }
  intelligence_metadata?: {
    sources_found: number
    context_enhanced: boolean
    processing_method: string
    search_methods_used?: string[]
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
    context_used: boolean
    context_length: number
    features_used: Record<string, boolean>
  }
  intelligence_metadata?: {
    sources_found: number
    context_enhanced: boolean
    processing_method: string
    search_methods_used: string[]
  }
  error?: string
}

// ===== OPTIMIZED CHAT API =====
class OptimizedChatAPI {
  private static readonly BASE_URL = 'http://localhost:8000'
  private static controller: AbortController | null = null

  static cancelRequest() {
    if (this.controller) {
      this.controller.abort()
      this.controller = null
    }
  }

  static async sendMessage(content: string, projectId?: string): Promise<ChatResponse> {
    console.log('🚀 Sending optimized chat message:', { 
      content: content.substring(0, 50) + '...', 
      projectId 
    })
    
    // Cancel any existing request
    this.cancelRequest()
    this.controller = new AbortController()
    
    const payload = {
      message: content,
      project_id: projectId || null,
      use_documents: true,
      model: 'gemini-1.5-flash'
    }

    try {
      const response = await fetch(`${this.BASE_URL}/api/v1/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload),
        signal: this.controller.signal,
      })

      const data = await response.json()

      if (!response.ok) {
        throw new Error(data.error || data.detail || 'Chat request failed')
      }

      console.log('✅ Chat response received:', {
        success: data.success,
        sourcesCount: data.sources?.length || 0,
        contextUsed: data.model_info?.context_used,
        processingMethod: data.intelligence_metadata?.processing_method
      })

      return {
        ...data,
        success: true
      }

    } catch (error: any) {
      if (error.name === 'AbortError') {
        throw new Error('Request cancelled')
      }
      
      console.error('❌ Chat request failed:', error)
      throw new Error(error.message || 'Failed to send message')
    } finally {
      this.controller = null
    }
  }

  static async testConnection(): Promise<boolean> {
    try {
      const response = await fetch(`${this.BASE_URL}/api/health`)
      return response.ok
    } catch {
      return false
    }
  }

  static async getSystemStatus(): Promise<any> {
    try {
      const response = await fetch(`${this.BASE_URL}/api/v1/system/status`)
      if (response.ok) {
        return await response.json()
      }
      return null
    } catch {
      return null
    }
  }
}

// ===== SOURCE DISPLAY COMPONENT =====
const SourceCard: React.FC<{ source: Source }> = ({ source }) => {
  const [isExpanded, setIsExpanded] = useState(false)
  
  const getMethodIcon = (method: string) => {
    switch (method) {
      case 'tfidf': return <Search className="w-3 h-3" />
      case 'semantic': return <Brain className="w-3 h-3" />
      default: return <Database className="w-3 h-3" />
    }
  }
  
  const getMethodColor = (method: string) => {
    switch (method) {
      case 'tfidf': return 'bg-blue-100 text-blue-800'
      case 'semantic': return 'bg-purple-100 text-purple-800'
      default: return 'bg-gray-100 text-gray-800'
    }
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className="bg-white border border-gray-200 rounded-lg p-3 hover:shadow-md transition-shadow"
    >
      <div className="flex items-start justify-between mb-2">
        <div className="flex items-center gap-2">
          <FileText className="w-4 h-4 text-gray-500" />
          <span className="font-medium text-sm truncate">{source.filename}</span>
        </div>
        <div className="flex items-center gap-1">
          <Badge variant="secondary" className="text-xs">
            {(source.relevance_score * 100).toFixed(0)}%
          </Badge>
        </div>
      </div>
      
      {source.search_methods && source.search_methods.length > 0 && (
        <div className="flex gap-1 mb-2">
          {source.search_methods.map((method, idx) => (
            <span
              key={idx}
              className={cn(
                "inline-flex items-center gap-1 px-2 py-1 rounded-full text-xs font-medium",
                getMethodColor(method)
              )}
            >
              {getMethodIcon(method)}
              {method}
            </span>
          ))}
        </div>
      )}
      
      <div className="text-sm text-gray-600">
        {isExpanded ? source.excerpt : `${source.excerpt.substring(0, 100)}...`}
        {source.excerpt.length > 100 && (
          <button
            onClick={() => setIsExpanded(!isExpanded)}
            className="ml-2 text-blue-600 hover:text-blue-800 font-medium"
          >
            {isExpanded ? 'Weniger' : 'Mehr'}
          </button>
        )}
      </div>
    </motion.div>
  )
}

// ===== MESSAGE COMPONENT =====
const MessageBubble: React.FC<{ message: Message }> = ({ message }) => {
  const [copied, setCopied] = useState(false)
  
  const copyToClipboard = async () => {
    try {
      await navigator.clipboard.writeText(message.content)
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
      toast.success('Nachricht kopiert!')
    } catch {
      toast.error('Kopieren fehlgeschlagen')
    }
  }

  const isUser = message.role === 'user'
  const isError = message.status === 'error'

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -20 }}
      className={cn(
        "flex gap-3 mb-6",
        isUser ? "justify-end" : "justify-start"
      )}
    >
      {!isUser && (
        <div className="flex-shrink-0">
          <div className={cn(
            "w-8 h-8 rounded-full flex items-center justify-center",
            isError ? "bg-red-100" : "bg-blue-100"
          )}>
            {isError ? (
              <AlertCircle className="w-4 h-4 text-red-600" />
            ) : (
              <Bot className="w-4 h-4 text-blue-600" />
            )}
          </div>
        </div>
      )}

      <div className={cn(
        "max-w-[80%] space-y-2",
        isUser ? "order-1" : "order-2"
      )}>
        {/* Message Bubble */}
        <div className={cn(
          "rounded-2xl px-4 py-3 shadow-sm",
          isUser 
            ? "bg-blue-600 text-white ml-auto" 
            : isError 
              ? "bg-red-50 border border-red-200 text-red-800"
              : "bg-gray-50 border border-gray-200"
        )}>
          <div className="whitespace-pre-wrap text-sm leading-relaxed">
            {message.content}
          </div>
          
          {/* Message Actions */}
          <div className="flex items-center justify-between mt-2 pt-2 border-t border-opacity-20">
            <span className="text-xs opacity-70">
              {new Date(message.timestamp).toLocaleTimeString('de-DE', {
                hour: '2-digit',
                minute: '2-digit'
              })}
            </span>
            
            <div className="flex items-center gap-1">
              {message.status === 'sending' && (
                <RefreshCw className="w-3 h-3 animate-spin" />
              )}
              {message.status === 'sent' && (
                <CheckCircle className="w-3 h-3" />
              )}
              <button
                onClick={copyToClipboard}
                className="p-1 rounded hover:bg-black hover:bg-opacity-10 transition-colors"
              >
                {copied ? (
                  <Check className="w-3 h-3" />
                ) : (
                  <Copy className="w-3 h-3" />
                )}
              </button>
            </div>
          </div>
        </div>

        {/* Intelligence Metadata */}
        {!isUser && message.intelligence_metadata && (
          <div className="flex items-center gap-2 text-xs text-gray-500">
            <Zap className="w-3 h-3" />
            <span>
              {message.intelligence_metadata.processing_method} • 
              {message.intelligence_metadata.sources_found} Quellen • 
              {message.intelligence_metadata.context_enhanced ? 'Kontext erweitert' : 'Basis-Antwort'}
            </span>
            {message.intelligence_metadata.search_methods_used && 
             message.intelligence_metadata.search_methods_used.length > 0 && (
              <span>• {message.intelligence_metadata.search_methods_used.join(', ')}</span>
            )}
          </div>
        )}

        {/* Sources */}
        {!isUser && message.sources && message.sources.length > 0 && (
          <div className="space-y-2">
            <div className="flex items-center gap-2 text-sm font-medium text-gray-700">
              <FileText className="w-4 h-4" />
              Quellen ({message.sources.length})
            </div>
            <div className="space-y-2 max-h-60 overflow-y-auto">
              {message.sources.map((source, idx) => (
                <SourceCard key={idx} source={source} />
              ))}
            </div>
          </div>
        )}
      </div>

      {isUser && (
        <div className="flex-shrink-0 order-2">
          <div className="w-8 h-8 rounded-full bg-green-100 flex items-center justify-center">
            <User className="w-4 h-4 text-green-600" />
          </div>
        </div>
      )}
    </motion.div>
  )
}

// ===== MAIN CHAT INTERFACE COMPONENT =====
export const ChatInterface: React.FC = () => {
  const { currentProject } = useStore()
  const [messages, setMessages] = useState<Message[]>([])
  const [inputValue, setInputValue] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [connectionStatus, setConnectionStatus] = useState<'connected' | 'disconnected' | 'checking'>('checking')
  const [systemStatus, setSystemStatus] = useState<any>(null)
  
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLTextAreaElement>(null)

  // Check connection status
  const checkConnection = useCallback(async () => {
    setConnectionStatus('checking')
    try {
      const isConnected = await OptimizedChatAPI.testConnection()
      const status = await OptimizedChatAPI.getSystemStatus()
      
      setConnectionStatus(isConnected ? 'connected' : 'disconnected')
      setSystemStatus(status)
      
      if (!isConnected) {
        toast.error('Backend-Verbindung fehlgeschlagen')
      }
    } catch {
      setConnectionStatus('disconnected')
      setSystemStatus(null)
    }
  }, [])

  // Auto-scroll to bottom
  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [])

  // Send message
  const sendMessage = useCallback(async () => {
    if (!inputValue.trim() || isLoading) return

    const userMessage: Message = {
      id: `user-${Date.now()}`,
      content: inputValue.trim(),
      role: 'user',
      timestamp: new Date().toISOString(),
      status: 'sent'
    }

    setMessages(prev => [...prev, userMessage])
    setInputValue('')
    setIsLoading(true)

    // Add loading indicator
    const loadingMessage: Message = {
      id: `assistant-${Date.now()}`,
      content: 'Analysiere Dokumente und erstelle Antwort...',
      role: 'assistant',
      timestamp: new Date().toISOString(),
      status: 'sending'
    }
    setMessages(prev => [...prev, loadingMessage])

    try {
      const response = await OptimizedChatAPI.sendMessage(
        userMessage.content,
        currentProject?.id
      )

      if (response.success) {
        const assistantMessage: Message = {
          id: response.chat_id || `assistant-${Date.now()}`,
          content: response.response,
          role: 'assistant',
          timestamp: response.timestamp,
          sources: response.sources,
          status: 'sent',
          model_info: response.model_info,
          intelligence_metadata: response.intelligence_metadata
        }

        setMessages(prev => prev.slice(0, -1).concat(assistantMessage))
        
        // Show success toast with details
        if (response.sources && response.sources.length > 0) {
          toast.success(`Antwort mit ${response.sources.length} Quellen generiert`)
        } else {
          toast.success('Antwort generiert')
        }
      } else {
        throw new Error(response.error || 'Unbekannter Fehler')
      }

    } catch (error: any) {
      console.error('Chat error:', error)
      
      const errorMessage: Message = {
        id: `error-${Date.now()}`,
        content: `Fehler: ${error.message}`,
        role: 'assistant',
        timestamp: new Date().toISOString(),
        status: 'error'
      }

      setMessages(prev => prev.slice(0, -1).concat(errorMessage))
      toast.error('Fehler beim Senden der Nachricht')
    } finally {
      setIsLoading(false)
    }
  }, [inputValue, isLoading, currentProject])

  // Handle key press
  const handleKeyPress = useCallback((e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendMessage()
    }
  }, [sendMessage])

  // Effects
  useEffect(() => {
    checkConnection()
    const interval = setInterval(checkConnection, 30000) // Check every 30s
    return () => clearInterval(interval)
  }, [checkConnection])

  useEffect(() => {
    scrollToBottom()
  }, [messages, scrollToBottom])

  useEffect(() => {
    inputRef.current?.focus()
  }, [])

  // Welcome message
  const welcomeMessage = useMemo(() => {
    if (!currentProject) {
      return "Hallo! Bitte wählen Sie ein Projekt aus, um mit Ihren Dokumenten zu chatten."
    }
    
    const docCount = systemStatus?.data?.documents || 0
    const ragDocs = systemStatus?.components?.rag_system?.documents || 0
    
    return `Hallo! Ich bin bereit, Ihre Fragen zu den Dokumenten in "${currentProject.name}" zu beantworten. ${docCount > 0 ? `${ragDocs} Dokumente sind durchsuchbar.` : 'Bitte laden Sie Dokumente hoch, um zu beginnen.'}`
  }, [currentProject, systemStatus])

  return (
    <div className="flex flex-col h-full bg-gray-50">
      {/* Header */}
      <div className="bg-white border-b border-gray-200 px-6 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <MessageSquare className="w-6 h-6 text-blue-600" />
            <div>
              <h2 className="text-lg font-semibold text-gray-900">
                Chat mit Dokumenten
              </h2>
              {currentProject && (
                <p className="text-sm text-gray-500">
                  Projekt: {currentProject.name}
                </p>
              )}
            </div>
          </div>
          
          <div className="flex items-center gap-3">
            {/* Connection Status */}
            <div className="flex items-center gap-2">
              <div className={cn(
                "w-2 h-2 rounded-full",
                connectionStatus === 'connected' ? 'bg-green-500' :
                connectionStatus === 'disconnected' ? 'bg-red-500' :
                'bg-yellow-500'
              )} />
              <span className="text-sm text-gray-600">
                {connectionStatus === 'connected' ? 'Verbunden' :
                 connectionStatus === 'disconnected' ? 'Getrennt' :
                 'Prüfend...'}
              </span>
            </div>

            {/* System Status */}
            {systemStatus && (
              <div className="flex items-center gap-2 text-sm text-gray-600">
                <Database className="w-4 h-4" />
                <span>
                  {systemStatus.data?.documents || 0} Docs, 
                  {systemStatus.components?.rag_system?.chunks || 0} Chunks
                </span>
              </div>
            )}

            <Button
              variant="outline"
              size="sm"
              onClick={checkConnection}
              disabled={connectionStatus === 'checking'}
            >
              <RefreshCw className={cn(
                "w-4 h-4 mr-2",
                connectionStatus === 'checking' && "animate-spin"
              )} />
              Aktualisieren
            </Button>
          </div>
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-6 py-6">
        {messages.length === 0 && (
          <div className="flex items-center justify-center h-full">
            <div className="text-center max-w-md">
              <Bot className="w-16 h-16 text-gray-400 mx-auto mb-4" />
              <h3 className="text-lg font-medium text-gray-900 mb-2">
                Bereit für Ihre Fragen
              </h3>
              <p className="text-gray-600">
                {welcomeMessage}
              </p>
            </div>
          </div>
        )}

        <AnimatePresence>
          {messages.map((message) => (
            <MessageBubble key={message.id} message={message} />
          ))}
        </AnimatePresence>
        
        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <div className="bg-white border-t border-gray-200 px-6 py-4">
        <div className="flex gap-3">
          <div className="flex-1 relative">
            <textarea
              ref={inputRef}
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyDown={handleKeyPress}
              placeholder={
                !currentProject 
                  ? "Wählen Sie zuerst ein Projekt aus..."
                  : connectionStatus !== 'connected'
                    ? "Keine Verbindung zum Backend..."
                    : "Stellen Sie Fragen zu Ihren Dokumenten..."
              }
              disabled={!currentProject || connectionStatus !== 'connected' || isLoading}
              className="w-full px-4 py-3 border border-gray-300 rounded-xl resize-none focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:bg-gray-100 disabled:cursor-not-allowed"
              rows={1}
              style={{ 
                minHeight: '48px',
                maxHeight: '120px',
                height: 'auto'
              }}
            />
          </div>
          
          <Button
            onClick={sendMessage}
            disabled={!inputValue.trim() || !currentProject || connectionStatus !== 'connected' || isLoading}
            className="px-6 py-3 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-400"
          >
            {isLoading ? (
              <RefreshCw className="w-4 h-4 animate-spin" />
            ) : (
              <Send className="w-4 h-4" />
            )}
          </Button>
        </div>
        
        {!currentProject && (
          <p className="text-xs text-amber-600 mt-2 flex items-center gap-1">
            <AlertCircle className="w-3 h-3" />
            Bitte wählen Sie ein Projekt aus, um mit Dokumenten zu chatten.
          </p>
        )}
      </div>
    </div>
  )
}

export default ChatInterface