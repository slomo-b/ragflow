// === hooks/useChat.ts - Enhanced Version ===
import { useState, useCallback } from 'react'
import toast from 'react-hot-toast'

interface ChatMessage {
  id: string
  content: string
  role: 'user' | 'assistant' | 'error'
  timestamp: Date
  sources?: Array<{
    type: string
    filename: string
    excerpt: string
    relevance_score: number
    match_type?: string
  }>
  intelligence_metadata?: {
    sources_analyzed: number
    context_enhanced: boolean
    prompt_length: number
  }
}

interface ChatResponse {
  response: string
  chat_id?: string
  project_id?: string
  timestamp: string
  model_info?: {
    model: string
    version: string
    features_used: {
      intelligent_document_search: boolean
      proactive_analysis: boolean
      intent_recognition: boolean
      fuzzy_matching: boolean
      context_enhancement: boolean
    }
  }
  sources?: Array<{
    type: string
    filename: string
    excerpt: string
    relevance_score: number
    match_type?: string
  }>
  intelligence_metadata?: {
    sources_analyzed: number
    context_enhanced: boolean
    prompt_length: number
  }
}

export function useChat(projectId?: string) {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [connectionStatus, setConnectionStatus] = useState<'connected' | 'disconnected' | 'unknown'>('unknown')

  // Check backend connection
  const checkConnection = useCallback(async () => {
    try {
      const response = await fetch('http://localhost:8000/api/health', {
        method: 'GET',
        headers: { 'Content-Type': 'application/json' }
      })
      
      if (response.ok) {
        const data = await response.json()
        setConnectionStatus('connected')
        return data
      } else {
        setConnectionStatus('disconnected')
        return null
      }
    } catch (error) {
      console.error('Connection check failed:', error)
      setConnectionStatus('disconnected')
      return null
    }
  }, [])

  const sendMessage = useCallback(async (content: string) => {
    if (!content.trim() || isLoading) return

    const userMessage: ChatMessage = {
      id: Math.random().toString(36),
      content: content.trim(),
      role: 'user',
      timestamp: new Date()
    }

    setMessages(prev => [...prev, userMessage])
    setIsLoading(true)

    try {
      // Check connection first
      const healthCheck = await checkConnection()
      if (!healthCheck) {
        throw new Error('Backend connection failed')
      }

      // Prepare message payload
      const messagePayload = {
        messages: [...messages, userMessage].map(msg => ({
          role: msg.role,
          content: msg.content
        })),
        project_id: projectId
      }

      console.log('🚀 Sending enhanced chat request:', {
        project_id: projectId,
        message_count: messagePayload.messages.length,
        user_query: content.substring(0, 50) + '...'
      })

      // Send to enhanced backend
      const response = await fetch('http://localhost:8000/api/v1/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(messagePayload),
      })

      if (!response.ok) {
        const errorText = await response.text()
        throw new Error(`HTTP ${response.status}: ${errorText}`)
      }

      const data: ChatResponse = await response.json()

      console.log('✅ Enhanced chat response received:', {
        sources_count: data.sources?.length || 0,
        features_used: data.model_info?.features_used,
        intelligence_metadata: data.intelligence_metadata
      })

      const aiMessage: ChatMessage = {
        id: data.chat_id || Math.random().toString(36),
        content: data.response,
        role: 'assistant',
        timestamp: new Date(data.timestamp),
        sources: data.sources,
        intelligence_metadata: data.intelligence_metadata
      }

      setMessages(prev => [...prev, aiMessage])

      // Show enhanced features notification
      if (data.model_info?.features_used) {
        const features = data.model_info.features_used
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
      if (data.sources && data.sources.length > 0) {
        const sourceFiles = [...new Set(data.sources.map(s => s.filename))].slice(0, 3)
        toast.success(
          `📄 Found relevant content in: ${sourceFiles.join(', ')}${data.sources.length > sourceFiles.length ? ` +${data.sources.length - sourceFiles.length} more` : ''}`,
          { 
            duration: 4000,
            icon: '🔍'
          }
        )
      }

    } catch (error) {
      console.error('Enhanced chat error:', error)
      
      let errorMessage = 'Es gab einen unerwarteten Fehler.'
      
      if (error instanceof Error) {
        if (error.message.includes('Failed to fetch') || error.message.includes('connection failed')) {
          errorMessage = `🔗 Verbindung zum Backend fehlgeschlagen. 
          
Stelle sicher, dass:
• Der Backend-Server läuft (http://localhost:8000)
• Die GOOGLE_API_KEY in der .env Datei konfiguriert ist
• Keine Firewall den Zugriff blockiert

Versuche es in ein paar Sekunden erneut.`
        } else if (error.message.includes('HTTP 400')) {
          errorMessage = '⚠️ Ungültige Anfrage. Bitte versuche es mit einer anderen Formulierung.'
        } else if (error.message.includes('HTTP 500')) {
          errorMessage = '🔧 Server-Fehler. Das Backend hat ein Problem bei der Verarbeitung.'
        } else {
          errorMessage = `❌ Fehler: ${error.message}`
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
    }
  }, [messages, projectId, isLoading, checkConnection, connectionStatus])

  const stopGeneration = useCallback(() => {
    setIsLoading(false)
    toast.info('Generation stopped', { duration: 2000 })
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

  // Test backend connection
  const testConnection = useCallback(async () => {
    const loadingToast = toast.loading('Testing backend connection...')
    
    try {
      const response = await fetch('http://localhost:8000/api/v1/chat/test', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      })

      const data = await response.json()
      
      if (data.status === 'success') {
        toast.success('✅ Backend connection successful!', { id: loadingToast })
        setConnectionStatus('connected')
        return true
      } else {
        toast.error(`❌ Backend test failed: ${data.message}`, { id: loadingToast })
        setConnectionStatus('disconnected')
        return false
      }
    } catch (error) {
      toast.error('❌ Cannot connect to backend. Please start the server.', { id: loadingToast })
      setConnectionStatus('disconnected')
      return false
    }
  }, [])

  // Get chat statistics
  const getChatStats = useCallback(() => {
    const userMessages = messages.filter(msg => msg.role === 'user').length
    const aiMessages = messages.filter(msg => msg.role === 'assistant').length
    const errorMessages = messages.filter(msg => msg.role === 'error').length
    const totalSources = messages.reduce((sum, msg) => sum + (msg.sources?.length || 0), 0)
    
    return {
      total_messages: messages.length,
      user_messages: userMessages,
      ai_messages: aiMessages,
      error_messages: errorMessages,
      sources_found: totalSources,
      has_intelligence_data: messages.some(msg => msg.intelligence_metadata)
    }
  }, [messages])

  return {
    messages,
    isLoading,
    connectionStatus,
    sendMessage,
    stopGeneration,
    clearChat,
    retryLastMessage,
    testConnection,
    checkConnection,
    getChatStats
  }
}