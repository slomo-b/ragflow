// === hooks/useChat.ts ===
import { useState } from 'react'

interface Message {
  id: string
  content: string
  role: 'user' | 'assistant' | 'error'
  timestamp: Date
  sources?: Array<{
    name: string
    relevance_score: number
    excerpt: string
  }>
}

export function useChat(projectId?: string) {
  const [messages, setMessages] = useState<Message[]>([])
  const [isLoading, setIsLoading] = useState(false)

  const sendMessage = async (content: string) => {
    const userMessage: Message = {
      id: Math.random().toString(36),
      content,
      role: 'user',
      timestamp: new Date()
    }

    setMessages(prev => [...prev, userMessage])
    setIsLoading(true)

    try {
      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 1000))
      
      const aiMessage: Message = {
        id: Math.random().toString(36),
        content: `I received your message: "${content}". This is a demo response.`,
        role: 'assistant',
        timestamp: new Date()
      }

      setMessages(prev => [...prev, aiMessage])
    } catch (error) {
      const errorMessage: Message = {
        id: Math.random().toString(36),
        content: 'Sorry, there was an error processing your request.',
        role: 'error',
        timestamp: new Date()
      }
      setMessages(prev => [...prev, errorMessage])
    } finally {
      setIsLoading(false)
    }
  }

  const stopGeneration = () => {
    setIsLoading(false)
  }

  const clearChat = () => {
    setMessages([])
  }

  return {
    messages,
    isLoading,
    sendMessage,
    stopGeneration,
    clearChat
  }
}