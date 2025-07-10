import { create } from 'zustand'
import { persist } from 'zustand/middleware'

interface Project {
  id: string
  name: string
  description?: string
  createdAt: Date
  updatedAt: Date
}

interface Document {
  id: string
  name: string
  type: string
  size: number
  projectId: string
  uploadedAt: Date
}

interface Chat {
  id: string
  title: string
  projectId: string
  messages: any[]
  createdAt: Date
  updatedAt: Date
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
        const project: Project = {
          ...projectData,
          id: Math.random().toString(36).substr(2, 9),
          createdAt: new Date(),
          updatedAt: new Date(),
        }
        set((state) => ({ projects: [...state.projects, project] }))
      },
      updateProject: (id, updates) => {
        set((state) => ({
          projects: state.projects.map((p) =>
            p.id === id ? { ...p, ...updates, updatedAt: new Date() } : p
          ),
        }))
      },
      deleteProject: (id) => {
        set((state) => ({
          projects: state.projects.filter((p) => p.id !== id),
          currentProject: state.currentProject?.id === id ? null : state.currentProject,
        }))
      },

      // Documents
      documents: [],
      addDocument: (documentData) => {
        const document: Document = {
          ...documentData,
          id: Math.random().toString(36).substr(2, 9),
          uploadedAt: new Date(),
        }
        set((state) => ({ documents: [...state.documents, document] }))
      },
      deleteDocument: (id) => {
        set((state) => ({
          documents: state.documents.filter((d) => d.id !== id),
        }))
      },

      // Chats
      chats: [],
      currentChat: null,
      setCurrentChat: (chat) => set({ currentChat: chat }),
      addChat: (chatData) => {
        const chat: Chat = {
          ...chatData,
          id: Math.random().toString(36).substr(2, 9),
          createdAt: new Date(),
          updatedAt: new Date(),
        }
        set((state) => ({ chats: [...state.chats, chat] }))
      },
      updateChat: (id, updates) => {
        set((state) => ({
          chats: state.chats.map((c) =>
            c.id === id ? { ...c, ...updates, updatedAt: new Date() } : c
          ),
        }))
      },
      deleteChat: (id) => {
        set((state) => ({
          chats: state.chats.filter((c) => c.id !== id),
          currentChat: state.currentChat?.id === id ? null : state.currentChat,
        }))
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