// frontend/hooks/useChat.ts - Vollständig korrigiert für RAGFlow Backend
import { useState, useCallback, useEffect, useRef } from 'react'
import toast from 'react-hot-toast'
import ApiService, { ChatResponse } from '@/services/api'

export interface ChatMessage {
  id: string
  content: string
  role: 'user' | 'assistant' | 'error'
  timestamp: Date
  sources?: Array<{
    id: string
    name: string
    filename: string
    excerpt: string
    relevance_score: number
  }>
  intelligence_metadata?: {
    query_complexity: string
    reasoning_depth: string
    context_integration: string
  }
}

export type ConnectionStatus = 'connected' | 'disconnected' | 'checking'

export function useChat(selectedProjectId?: string) {
  // State
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [connectionStatus, setConnectionStatus] = useState<ConnectionStatus>('checking')
  
  // Refs
  const abortControllerRef = useRef<AbortController | null>(null)

  // ===== CONNECTION MANAGEMENT =====
  const checkConnection = useCallback(async (): Promise<boolean> => {
    try {
      const response = await ApiService.healthCheck()
      const isConnected = response.status === 'healthy'
      setConnectionStatus(isConnected ? 'connected' : 'disconnected')
      return isConnected
    } catch (error) {
      console.error('Connection check failed:', error)
      setConnectionStatus('disconnected')
      return false
    }
  }, [])

  // Initial connection check
  useEffect(() => {
    checkConnection()
    
    // Periodic connection check
    const interval = setInterval(checkConnection, 30000) // Check every 30 seconds
    return () => clearInterval(interval)
  }, [checkConnection])

  // ===== SEND MESSAGE - KORRIGIERT =====
  const sendMessage = useCallback(async (content: string) => {
    if (!content.trim() || isLoading) return
    if (content.length > 1000) {
      toast.error('Message too long (max 1000 characters)')
      return
    }

    // Create user message
    const userMessage: ChatMessage = {
      id: Math.random().toString(36),
      content: content.trim(),
      role: 'user',
      timestamp: new Date()
    }

    setMessages(prev => [...prev, userMessage])
    setIsLoading(true)

    // Abort previous request if any
    if (abortControllerRef.current) {
      abortControllerRef.current.abort()
    }
    
    // Create new abort controller
    abortControllerRef.current = new AbortController()

    try {
      // Check connection first
      const healthCheck = await checkConnection()
      if (!healthCheck) {
        throw new Error('Backend connection failed')
      }

      console.log('🚀 Sending chat request:', {
        message: content.substring(0, 50) + '...',
        project_id: selectedProjectId,
        timestamp: new Date().toISOString()
      })

      // ✅ Send to backend using corrected ApiService
      const response: ChatResponse = await ApiService.sendChatMessage({
        message: content.trim(),
        project_id: selectedProjectId,
        // Optional parameters can be added:
        // model: 'gemini-pro',
        // temperature: 0.7,
        // max_tokens: 1000
      })

      console.log('✅ Chat response received:', {
        sources_count: response.sources?.length || 0,
        model_info: response.model_info,
        intelligence_metadata: response.intelligence_metadata
      })

      // Create AI message
      const aiMessage: ChatMessage = {
        id: response.chat_id || Math.random().toString(36),
        content: response.response,
        role: 'assistant',
        timestamp: new Date(response.timestamp),
        sources: response.sources,
        intelligence_metadata: response.intelligence_metadata
      }

      setMessages(prev => [...prev, aiMessage])

      // ===== ENHANCED NOTIFICATIONS =====
      
      // Show enhanced features notification
      if (response.model_info?.features_used) {
        const features = response.model_info.features_used
        const activeFeatures = Object.entries(features)
          .filter(([_, active]) => active)
          .map(([feature, _]) => feature.replace(/_/g, ' '))

        if (activeFeatures.length > 0) {
          toast.success(
            `🧠 Enhanced AI features used: ${activeFeatures.slice(0, 2).join(', ')}${activeFeatures.length > 2 ? ` +${activeFeatures.length - 2} more` : ''}`,
            { 
              duration: 3000,
              icon: '🚀'
            }
          )
        }
      }

      // Show sources found notification
      if (response.sources && response.sources.length > 0) {
        const sourceFiles = [...new Set(response.sources.map(s => s.filename || s.name))].slice(0, 3)
        toast.success(
          `📄 Found relevant content in: ${sourceFiles.join(', ')}${response.sources.length > sourceFiles.length ? ` +${response.sources.length - sourceFiles.length} more` : ''}`,
          { 
            duration: 4000,
            icon: '📚'
          }
        )
      }

      // Intelligence metadata notification
      if (response.intelligence_metadata) {
        const metadata = response.intelligence_metadata
        toast(
          `🤖 Analysis: ${metadata.query_complexity} complexity, ${metadata.reasoning_depth} reasoning`,
          { 
            duration: 2000,
            icon: '🧠'
          }
        )
      }

    } catch (error) {
      console.error('Chat request failed:', error)
      
      let errorMessage = 'Sorry, something went wrong. Please try again.'
      
      if (error instanceof Error) {
        if (error.name === 'AbortError') {
          // Request was aborted, don't show error message
          return
        } else if (error.message.includes('Backend connection failed')) {
          errorMessage = '🔌 Backend is offline. Please check the server connection.'
          setConnectionStatus('disconnected')
        } else if (error.message.includes('Google AI API key not configured')) {
          errorMessage = '🔑 AI service not configured. Please set the Google API key in backend settings.'
        } else if (error.message.includes('timeout')) {
          errorMessage = '⏰ Request timed out. The AI service might be overloaded. Please try again.'
        } else if (error.message.includes('429')) {
          errorMessage = '🚦 Rate limit exceeded. Please wait a moment before trying again.'
        } else if (error.message.includes('500')) {
          errorMessage = '🔧 Server error. Please try again or contact support.'
        } else if (error.message.includes('503')) {
          errorMessage = '🔑 AI service not available. Please check the API configuration.'
        } else {
          errorMessage = `❌ Error: ${error.message}`
        }
      }

      const errorChatMessage: ChatMessage = {
        id: Math.random().toString(36),
        content: errorMessage,
        role: 'error',
        timestamp: new Date()
      }

      setMessages(prev => [...prev, errorChatMessage])
      
      // Show error toast
      if (connectionStatus === 'disconnected') {
        toast.error('Backend offline. Please start the server.', { duration: 5000 })
      } else {
        toast.error('Chat request failed. Please try again.', { duration: 4000 })
      }
      
    } finally {
      setIsLoading(false)
      abortControllerRef.current = null
    }
  }, [selectedProjectId, isLoading, checkConnection, connectionStatus])

  // ===== UTILITY FUNCTIONS =====
  
  const stopGeneration = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort()
    }
    setIsLoading(false)
    toast('Generation stopped', { 
      icon: 'ℹ️',
      duration: 2000 
    })
  }, [])

  const clearChat = useCallback(() => {
    setMessages([])
    toast.success('Chat cleared', { duration: 2000 })
  }, [])

  const retryLastMessage = useCallback(() => {
    if (messages.length === 0) return

    // Find the last user message
    const lastUserMessage = [...messages].reverse().find(msg => msg.role === 'user')
    if (lastUserMessage) {
      // Remove messages after the last user message
      const lastUserIndex = messages.findIndex(msg => msg.id === lastUserMessage.id)
      setMessages(prev => prev.slice(0, lastUserIndex + 1))
      
      // Resend the message
      sendMessage(lastUserMessage.content)
    }
  }, [messages, sendMessage])

  // ===== TEST FUNCTIONS =====
  
  // Test backend connection
  const testConnection = useCallback(async () => {
    const loadingToast = toast.loading('Testing backend connection...')
    
    try {
      const result = await ApiService.testConnection()
      
      if (result.status === 'success') {
        toast.success('✅ Backend connection successful!', { id: loadingToast })
        setConnectionStatus('connected')
        return true
      } else {
        toast.error(`❌ Backend test failed: ${result.message}`, { id: loadingToast })
        setConnectionStatus('disconnected')
        return false
      }
    } catch (error) {
      toast.error('❌ Cannot connect to backend. Please start the server.', { id: loadingToast })
      setConnectionStatus('disconnected')
      return false
    }
  }, [])

  // Test AI service
  const testAI = useCallback(async () => {
    const loadingToast = toast.loading('Testing AI service...')
    
    try {
      // Get AI info first
      const aiInfo = await ApiService.getAIInfo()
      
      if (!aiInfo.api_configured) {
        toast.error('❌ Google AI API key not configured', { id: loadingToast })
        return false
      }

      toast.success('✅ AI service ready!', { id: loadingToast })
      
      // Send test message
      await sendMessage('Hello, this is a connection test.')
      return true

    } catch (error) {
      console.error('AI test failed:', error)
      
      if (error instanceof Error) {
        if (error.message.includes('503')) {
          toast.error('❌ Google AI API key not configured', { id: loadingToast })
        } else {
          toast.error(`❌ AI test failed: ${error.message}`, { id: loadingToast })
        }
      } else {
        toast.error('❌ AI service unavailable', { id: loadingToast })
      }
      return false
    }
  }, [sendMessage])

  // ===== CHAT HISTORY FUNCTIONS =====
  
  const loadChatHistory = useCallback(async (project_id?: string) => {
    try {
      const chats = await ApiService.getChatHistory(project_id)
      console.log('📜 Chat history loaded:', chats)
      // TODO: Implement chat history loading logic
      toast('Chat history feature coming soon', { 
        icon: 'ℹ️',
        duration: 2000 
      })
    } catch (error) {
      console.error('Failed to load chat history:', error)
      toast.error('Failed to load chat history')
    }
  }, [])

  const saveCurrentChat = useCallback(async () => {
    if (messages.length === 0) {
      toast('No messages to save', { 
        icon: 'ℹ️',
        duration: 2000 
      })
      return
    }

    try {
      // TODO: Implement chat saving when backend supports it
      toast('Chat saving feature coming soon', { 
        icon: 'ℹ️',
        duration: 2000 
      })
    } catch (error) {
      console.error('Failed to save chat:', error)
      toast.error('Failed to save chat')
    }
  }, [messages])

  // ===== CHAT STATISTICS =====
  
  const getChatStats = useCallback(() => {
    const userMessages = messages.filter(m => m.role === 'user').length
    const assistantMessages = messages.filter(m => m.role === 'assistant').length
    const errorMessages = messages.filter(m => m.role === 'error').length
    const totalSources = messages.reduce((acc, m) => acc + (m.sources?.length || 0), 0)

    return {
      userMessages,
      assistantMessages,
      errorMessages,
      totalMessages: messages.length,
      totalSources,
      lastActivity: messages.length > 0 ? messages[messages.length - 1].timestamp : null
    }
  }, [messages])

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (abortControllerRef.current) {
        abortControllerRef.current.abort()
      }
    }
  }, [])

    // Return hook interface
  return {
    // State
    messages,
    isLoading,
    connectionStatus,
    
    // Core functions
    sendMessage,
    stopGeneration,
    clearChat,
    retryLastMessage,
    
    // Test functions
    testConnection,
    testAI,
    
    // History functions
    loadChatHistory,
    saveCurrentChat,
    
    // Utility functions
    getChatStats,
    checkConnection
  }
}

export default useChat