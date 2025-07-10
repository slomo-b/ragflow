import axios, { AxiosInstance, AxiosResponse } from 'axios'
import toast from 'react-hot-toast'

// API Configuration
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

// Create axios instance
const api: AxiosInstance = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
})

// Request interceptor
api.interceptors.request.use(
  (config) => {
    // Add auth token if available
    const token = localStorage.getItem('auth_token')
    if (token) {
      config.headers.Authorization = `Bearer ${token}`
    }
    
    console.log(`🚀 API Request: ${config.method?.toUpperCase()} ${config.url}`)
    return config
  },
  (error) => {
    console.error('❌ API Request Error:', error)
    return Promise.reject(error)
  }
)

// Response interceptor
api.interceptors.response.use(
  (response: AxiosResponse) => {
    console.log(`✅ API Response: ${response.status} ${response.config.url}`)
    return response
  },
  (error) => {
    console.error('❌ API Response Error:', error)
    
    // Handle different error types
    if (error.response) {
      const { status, data } = error.response
      
      switch (status) {
        case 400:
          toast.error(data.detail || 'Invalid request')
          break
        case 401:
          toast.error('Authentication required')
          // Redirect to login if needed
          break
        case 403:
          toast.error('Access denied')
          break
        case 404:
          toast.error('Resource not found')
          break
        case 422:
          toast.error(data.detail || 'Validation error')
          break
        case 429:
          toast.error('Too many requests. Please wait.')
          break
        case 500:
          toast.error('Server error. Please try again.')
          break
        default:
          toast.error('An unexpected error occurred')
      }
    } else if (error.request) {
      toast.error('Network error. Please check your connection.')
    } else {
      toast.error('An unexpected error occurred')
    }
    
    return Promise.reject(error)
  }
)

// API Types
export interface ApiResponse<T = any> {
  success?: boolean
  message?: string
  timestamp?: string
  data?: T
}

export interface PaginatedResponse<T> {
  items?: T[]
  total: number
  skip: number
  limit: number
  has_more?: boolean
}

export interface ChatRequest {
  messages: Array<{
    role: 'user' | 'assistant' | 'system'
    content: string
  }>
  project_id?: string
  model?: string
  temperature?: number
  max_tokens?: number
}

export interface ChatResponse {
  response: string
  chat_id?: string
  project_id?: string
  timestamp: string
  model_info?: {
    model: string
    temperature: number
  }
  sources?: Array<{
    id: string
    name: string
    excerpt: string
    relevance_score: number
  }>
}

export interface Project {
  id: string
  name: string
  description?: string
  created_at: string
  updated_at: string
  document_ids: string[]
  document_count?: number
  chat_count?: number
  status?: string
  settings: Record<string, any>
}

export interface Document {
  id: string
  filename: string
  file_type: string
  file_size: number
  uploaded_at: string
  processing_status: 'pending' | 'processing' | 'completed' | 'failed'
  project_ids: string[]
  tags: string[]
  summary?: string
}

// API Service Class
export class ApiService {
  // Health Check
  static async healthCheck(): Promise<any> {
    const response = await api.get('/api/health')
    return response.data
  }

  static async detailedHealthCheck(): Promise<any> {
    const response = await api.get('/api/health/detailed')
    return response.data
  }

  // Chat APIs
  static async sendChatMessage(request: ChatRequest): Promise<ChatResponse> {
    const response = await api.post('/api/v1/chat', request)
    return response.data
  }

  static async testGeminiConnection(): Promise<any> {
    const response = await api.post('/api/v1/chat/test')
    return response.data
  }

  static async getAvailableModels(): Promise<any> {
    const response = await api.get('/api/v1/chat/models')
    return response.data
  }

  // Project APIs
  static async getProjects(params?: {
    skip?: number
    limit?: number
    search?: string
  }): Promise<{ projects: Project[]; total: number; skip: number; limit: number }> {
    const response = await api.get('/api/v1/projects/', { params })
    return response.data
  }

  static async createProject(data: {
    name: string
    description?: string
  }): Promise<Project> {
    const response = await api.post('/api/v1/projects/', data)
    return response.data
  }

  static async getProject(id: string): Promise<Project & { documents: Document[] }> {
    const response = await api.get(`/api/v1/projects/${id}`)
    return response.data
  }

  static async updateProject(id: string, data: {
    name?: string
    description?: string
  }): Promise<Project> {
    const response = await api.put(`/api/v1/projects/${id}`, data)
    return response.data
  }

  static async deleteProject(id: string): Promise<{ message: string }> {
    const response = await api.delete(`/api/v1/projects/${id}`)
    return response.data
  }

  static async associateDocument(projectId: string, documentId: string): Promise<{ message: string }> {
    const response = await api.post(`/api/v1/projects/${projectId}/documents/${documentId}`)
    return response.data
  }

  static async disassociateDocument(projectId: string, documentId: string): Promise<{ message: string }> {
    const response = await api.delete(`/api/v1/projects/${projectId}/documents/${documentId}`)
    return response.data
  }

  // Document APIs
  static async getDocuments(params?: {
    skip?: number
    limit?: number
    project_id?: string
  }): Promise<{ documents: Document[]; total: number; skip: number; limit: number }> {
    const response = await api.get('/api/v1/documents/', { params })
    return response.data
  }

  // Upload APIs
  static async uploadDocuments(data: {
    files: File[]
    project_id?: string
    tags?: string
  }): Promise<{
    message: string
    documents: Array<{
      id: string
      filename: string
      file_type: string
      file_size: number
      status: string
    }>
  }> {
    const formData = new FormData()
    
    data.files.forEach(file => {
      formData.append('files', file)
    })
    
    if (data.project_id) {
      formData.append('project_id', data.project_id)
    }
    
    if (data.tags) {
      formData.append('tags', data.tags)
    }

    const response = await api.post('/api/v1/upload/documents', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      // Progress tracking
      onUploadProgress: (progressEvent) => {
        if (progressEvent.total) {
          const percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total)
          console.log(`Upload Progress: ${percentCompleted}%`)
        }
      },
    })
    
    return response.data
  }

  static async getUploadInfo(): Promise<{
    max_file_size: number
    max_file_size_mb: number
    allowed_extensions: string[]
    supported_formats: Record<string, string>
  }> {
    const response = await api.get('/api/v1/upload/info')
    return response.data
  }

  static async validateFiles(files: File[]): Promise<{
    total_files: number
    valid_files: number
    invalid_files: number
    results: Array<{
      filename: string
      valid: boolean
      errors: string[]
      file_size?: number
      file_type?: string
    }>
  }> {
    const formData = new FormData()
    files.forEach(file => {
      formData.append('files', file)
    })

    const response = await api.post('/api/v1/upload/validate', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    })
    
    return response.data
  }
}

// Export the configured axios instance for custom requests
export { api }
export default api