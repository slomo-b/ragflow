// frontend/stores/useStore.ts
import { create } from 'zustand'
import { persist } from 'zustand/middleware'

interface Project {
  id: string
  name: string
  description?: string
  createdAt: string // ISO String für bessere Serialization
  updatedAt: string
  document_count?: number
  chat_count?: number
}

interface Document {
  id: string
  name: string
  type: string
  size: number
  projectId: string
  uploadedAt: string // ISO String
  processing_status?: 'pending' | 'processing' | 'completed' | 'failed'
}

interface Chat {
  id: string
  title: string
  projectId: string
  messages: any[]
  createdAt: string // ISO String
  updatedAt: string
}

interface AppState {
  // Projects
  projects: Project[]
  currentProject: Project | null
  setCurrentProject: (project: Project | null) => void
  addProject: (project: Omit<Project, 'id' | 'createdAt' | 'updatedAt'>) => void
  updateProject: (id: string, project: Partial<Project>) => void
  deleteProject: (id: string) => void

  // Documents
  documents: Document[]
  addDocument: (document: Omit<Document, 'id' | 'uploadedAt'>) => void
  deleteDocument: (id: string) => void

  // Chats
  chats: Chat[]
  currentChat: Chat | null
  setCurrentChat: (chat: Chat | null) => void
  addChat: (chat: Omit<Chat, 'id' | 'createdAt' | 'updatedAt'>) => void
  updateChat: (id: string, chat: Partial<Chat>) => void
  deleteChat: (id: string) => void

  // UI State
  sidebarCollapsed: boolean
  setSidebarCollapsed: (collapsed: boolean) => void
  
  // Notifications
  notifications: any[]
  addNotification: (notification: any) => void
  clearNotifications: () => void
}

export const useStore = create<AppState>()(
  persist(
    (set, get) => ({
      // Projects
      projects: [],
      currentProject: null,
      setCurrentProject: (project) => set({ currentProject: project }),
      
      addProject: (projectData) => {
        const now = new Date().toISOString()
        const project: Project = {
          ...projectData,
          id: Math.random().toString(36).substr(2, 9),
          createdAt: now,
          updatedAt: now,
          document_count: 0,
          chat_count: 0,
        }
        set((state) => ({ 
          projects: [...state.projects, project],
          currentProject: project // Set as current project after creation
        }))
      },
      
      updateProject: (id, updates) => {
        set((state) => ({
          projects: state.projects.map((p) =>
            p.id === id ? { ...p, ...updates, updatedAt: new Date().toISOString() } : p
          ),
          // Update currentProject if it's the one being updated
          currentProject: state.currentProject?.id === id 
            ? { ...state.currentProject, ...updates, updatedAt: new Date().toISOString() }
            : state.currentProject
        }))
      },
      
      deleteProject: (id) => {
        set((state) => ({
          projects: state.projects.filter((p) => p.id !== id),
          currentProject: state.currentProject?.id === id ? null : state.currentProject,
          // Also remove related documents and chats
          documents: state.documents.filter((d) => d.projectId !== id),
          chats: state.chats.filter((c) => c.projectId !== id),
        }))
      },

      // Documents
      documents: [],
      addDocument: (documentData) => {
        const document: Document = {
          ...documentData,
          id: Math.random().toString(36).substr(2, 9),
          uploadedAt: new Date().toISOString(),
          processing_status: 'pending',
        }
        set((state) => {
          // Update project document count
          const updatedProjects = state.projects.map(p => 
            p.id === document.projectId 
              ? { ...p, document_count: (p.document_count || 0) + 1, updatedAt: new Date().toISOString() }
              : p
          )
          
          return {
            documents: [...state.documents, document],
            projects: updatedProjects,
            currentProject: state.currentProject?.id === document.projectId
              ? { ...state.currentProject, document_count: (state.currentProject.document_count || 0) + 1 }
              : state.currentProject
          }
        })
      },
      
      deleteDocument: (id) => {
        set((state) => {
          const document = state.documents.find(d => d.id === id)
          if (!document) return state
          
          // Update project document count
          const updatedProjects = state.projects.map(p => 
            p.id === document.projectId 
              ? { ...p, document_count: Math.max(0, (p.document_count || 0) - 1), updatedAt: new Date().toISOString() }
              : p
          )
          
          return {
            documents: state.documents.filter((d) => d.id !== id),
            projects: updatedProjects,
            currentProject: state.currentProject?.id === document.projectId
              ? { ...state.currentProject, document_count: Math.max(0, (state.currentProject.document_count || 0) - 1) }
              : state.currentProject
          }
        })
      },

      // Chats
      chats: [],
      currentChat: null,
      setCurrentChat: (chat) => set({ currentChat: chat }),
      
      addChat: (chatData) => {
        const now = new Date().toISOString()
        const chat: Chat = {
          ...chatData,
          id: Math.random().toString(36).substr(2, 9),
          createdAt: now,
          updatedAt: now,
        }
        set((state) => {
          // Update project chat count
          const updatedProjects = state.projects.map(p => 
            p.id === chat.projectId 
              ? { ...p, chat_count: (p.chat_count || 0) + 1, updatedAt: new Date().toISOString() }
              : p
          )
          
          return {
            chats: [...state.chats, chat],
            projects: updatedProjects,
            currentProject: state.currentProject?.id === chat.projectId
              ? { ...state.currentProject, chat_count: (state.currentProject.chat_count || 0) + 1 }
              : state.currentProject
          }
        })
      },
      
      updateChat: (id, updates) => {
        set((state) => ({
          chats: state.chats.map((c) =>
            c.id === id ? { ...c, ...updates, updatedAt: new Date().toISOString() } : c
          ),
          currentChat: state.currentChat?.id === id 
            ? { ...state.currentChat, ...updates, updatedAt: new Date().toISOString() }
            : state.currentChat
        }))
      },
      
      deleteChat: (id) => {
        set((state) => {
          const chat = state.chats.find(c => c.id === id)
          if (!chat) return state
          
          // Update project chat count
          const updatedProjects = state.projects.map(p => 
            p.id === chat.projectId 
              ? { ...p, chat_count: Math.max(0, (p.chat_count || 0) - 1), updatedAt: new Date().toISOString() }
              : p
          )
          
          return {
            chats: state.chats.filter((c) => c.id !== id),
            currentChat: state.currentChat?.id === id ? null : state.currentChat,
            projects: updatedProjects,
            currentProject: state.currentProject?.id === chat.projectId
              ? { ...state.currentProject, chat_count: Math.max(0, (state.currentProject.chat_count || 0) - 1) }
              : state.currentProject
          }
        })
      },

      // UI State
      sidebarCollapsed: false,
      setSidebarCollapsed: (collapsed) => set({ sidebarCollapsed: collapsed }),

      // Notifications
      notifications: [],
      addNotification: (notification) => {
        set((state) => ({
          notifications: [...state.notifications, { ...notification, id: Date.now() }],
        }))
      },
      clearNotifications: () => set({ notifications: [] }),
    }),
    {
      name: 'ragflow-store',
      // Only persist essential data
      partialize: (state) => ({
        projects: state.projects,
        documents: state.documents,
        chats: state.chats,
        currentProject: state.currentProject,
        sidebarCollapsed: state.sidebarCollapsed,
      }),
    }
  )
)