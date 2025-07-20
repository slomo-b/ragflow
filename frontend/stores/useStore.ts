// frontend/stores/useStore.ts
import { create } from 'zustand'
import { persist } from 'zustand/middleware'

interface AppState {
  // UI State
  currentView: string
  sidebarCollapsed: boolean
  
  // User Preferences
  theme: 'light' | 'dark' | 'system'
  
  // Project State
  selectedProjectId: string | null
  
  // Chat State
  chatHistory: any[]
  
  // Actions
  setCurrentView: (view: string) => void
  setSidebarCollapsed: (collapsed: boolean) => void
  setTheme: (theme: 'light' | 'dark' | 'system') => void
  setSelectedProjectId: (id: string | null) => void
  addChatMessage: (message: any) => void
  clearChatHistory: () => void
  reset: () => void
}

export const useStore = create<AppState>()(
  persist(
    (set, get) => ({
      // Initial state
      currentView: 'chat',
      sidebarCollapsed: false,
      theme: 'light',
      selectedProjectId: null,
      chatHistory: [],
      
      // Actions
      setCurrentView: (view) => set({ currentView: view }),
      setSidebarCollapsed: (collapsed) => set({ sidebarCollapsed: collapsed }),
      setTheme: (theme) => set({ theme }),
      setSelectedProjectId: (id) => set({ selectedProjectId: id }),
      addChatMessage: (message) => set((state) => ({
        chatHistory: [...state.chatHistory, message]
      })),
      clearChatHistory: () => set({ chatHistory: [] }),
      reset: () => set({
        currentView: 'chat',
        sidebarCollapsed: false,
        selectedProjectId: null,
        chatHistory: []
      })
    }),
    {
      name: 'ragflow-storage',
      partialize: (state) => ({
        theme: state.theme,
        sidebarCollapsed: state.sidebarCollapsed,
        selectedProjectId: state.selectedProjectId
      })
    }
  )
)

// Einfache Fallback-Version falls Zustand nicht verfügbar ist
export const useSimpleStore = () => {
  const [state, setState] = React.useState({
    currentView: 'chat',
    sidebarCollapsed: false,
    theme: 'light' as const,
    selectedProjectId: null as string | null,
    chatHistory: [] as any[]
  })

  return {
    ...state,
    setCurrentView: (view: string) => setState(prev => ({ ...prev, currentView: view })),
    setSidebarCollapsed: (collapsed: boolean) => setState(prev => ({ ...prev, sidebarCollapsed: collapsed })),
    setTheme: (theme: 'light' | 'dark' | 'system') => setState(prev => ({ ...prev, theme })),
    setSelectedProjectId: (id: string | null) => setState(prev => ({ ...prev, selectedProjectId: id })),
    addChatMessage: (message: any) => setState(prev => ({ 
      ...prev, 
      chatHistory: [...prev.chatHistory, message] 
    })),
    clearChatHistory: () => setState(prev => ({ ...prev, chatHistory: [] })),
    reset: () => setState({
      currentView: 'chat',
      sidebarCollapsed: false,
      theme: 'light',
      selectedProjectId: null,
      chatHistory: []
    })
  }
}