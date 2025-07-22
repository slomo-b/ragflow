// frontend/services/api.ts - An tatsächliche Backend-Routen angepasst
import axios from 'axios'

// Base URL
const API_BASE_URL = 'http://localhost:8000'

// Axios instance
export const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
})

// Request/Response interceptors
api.interceptors.request.use(
  (config) => {
    console.log(`🚀 API Request: ${config.method?.toUpperCase()} ${config.url}`)
    return config
  },
  (error) => {
    console.error('❌ API Request Error:', error)
    return Promise.reject(error)
  }
)

api.interceptors.response.use(
  (response) => {
    console.log(`✅ API Response: ${response.status} - ${response.config.url}`)
    return response
  },
  (error) => {
    console.error(`❌ API Response Error: ${error.response?.status} - ${error.config?.url}`)
    return Promise.reject(error)
  }
)

// Type Definitions
export interface ApiResponse<T = any> {
  data: T
  status: number
  statusText: string
}

export interface ChatRequest {
  message: string
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
    features_used?: Record<string, boolean>
  }
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

// API Service Class - Angepasst an tatsächliche Backend-Routen
export class ApiService {
  // ===== HEALTH & SYSTEM =====
  static async healthCheck(): Promise<any> {
    const response = await api.get('/api/health')
    return response.data
  }

  static async getConfiguration(): Promise<any> {
    const response = await api.get('/api/config')
    return response.data
  }

  static async getSystemInfo(): Promise<any> {
    const response = await api.get('/api/system/info')
    return response.data
  }

  static async getAIInfo(): Promise<any> {
    const response = await api.get('/api/ai/info')
    return response.data
  }

  // ===== CHAT APIs =====
  static async sendChatMessage(request: ChatRequest): Promise<ChatResponse> {
    // Backend erwartet diese Struktur für /api/chat
    const response = await api.post('/api/chat', request)
    return response.data
  }

  // Connection Test - Verwendet Health Check da kein separater Test-Endpoint
  static async testConnection(): Promise<any> {
    const response = await api.get('/api/health')
    return {
      status: response.data.status === 'healthy' ? 'success' : 'error',
      message: `Backend is ${response.data.status}`,
      details: response.data
    }
  }

  // Available Models - Nutzt AI Info Endpoint
  static async getAvailableModels(): Promise<any> {
    const response = await api.get('/api/ai/info')
    return {
      models: [{
        id: response.data.model,
        name: response.data.model,
        provider: response.data.provider
      }],
      default_model: response.data.model
    }
  }

  // ===== PROJECT APIs =====
  static async getProjects(params?: {
    skip?: number
    limit?: number
    search?: string
  }): Promise<{ projects: Project[]; total: number; skip: number; limit: number }> {
    const response = await api.get('/api/projects', { params })
    
    // Backend gibt direkte Array zurück, Frontend erwartet wrapped response
    const projects = Array.isArray(response.data) ? response.data : []
    return {
      projects,
      total: projects.length,
      skip: params?.skip || 0,
      limit: params?.limit || 100
    }
  }

  static async createProject(data: {
    name: string
    description?: string
  }): Promise<Project> {
    const response = await api.post('/api/projects', data)
    return response.data
  }

  static async getProject(id: string): Promise<Project & { documents: Document[] }> {
    const response = await api.get(`/api/projects/${id}`)
    return response.data
  }

  static async updateProject(id: string, data: {
    name?: string
    description?: string
  }): Promise<Project> {
    const response = await api.put(`/api/projects/${id}`, data)
    return response.data
  }

  static async deleteProject(id: string): Promise<{ message: string }> {
    const response = await api.delete(`/api/projects/${id}`)
    return response.data
  }

  static async associateDocument(projectId: string, documentId: string): Promise<{ message: string }> {
    const response = await api.post(`/api/projects/${projectId}/documents/${documentId}`)
    return response.data
  }

  static async disassociateDocument(projectId: string, documentId: string): Promise<{ message: string }> {
    const response = await api.delete(`/api/projects/${projectId}/documents/${documentId}`)
    return response.data
  }

  // ===== DOCUMENT APIs =====
  static async getDocuments(params?: {
    skip?: number
    limit?: number
    project_id?: string
  }): Promise<{ documents: Document[]; total: number; skip: number; limit: number }> {
    const response = await api.get('/api/documents', { params })
    
    // Backend gibt direkte Array zurück
    const documents = Array.isArray(response.data) ? response.data : []
    return {
      documents,
      total: documents.length,
      skip: params?.skip || 0,
      limit: params?.limit || 100
    }
  }

  static async deleteDocument(id: string): Promise<{ message: string }> {
    const response = await api.delete(`/api/documents/${id}`)
    return response.data
  }

  // ===== UPLOAD APIs =====
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

    const response = await api.post('/api/upload', formData, {
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
    // Nutzt Configuration Endpoint da kein separater Upload Info Endpoint
    const config = await this.getConfiguration()
    return {
      max_file_size: config.max_file_size,
      max_file_size_mb: Math.round(config.max_file_size / (1024 * 1024)),
      allowed_extensions: ['.pdf', '.docx', '.doc', '.txt', '.md'],
      supported_formats: {
        '.pdf': 'PDF Documents',
        '.docx': 'Word Documents',
        '.doc': 'Word Documents (Legacy)',
        '.txt': 'Text Files',
        '.md': 'Markdown Files'
      }
    }
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
    // Client-seitige Validierung da Backend keinen separaten Validation Endpoint hat
    const config = await this.getConfiguration()
    const maxSize = config.max_file_size
    const allowedTypes = ['.pdf', '.docx', '.doc', '.txt', '.md']
    
    const results = files.map(file => {
      const errors: string[] = []
      const extension = '.' + file.name.split('.').pop()?.toLowerCase()
      
      if (!allowedTypes.includes(extension)) {
        errors.push(`File type ${extension} not allowed`)
      }
      
      if (file.size > maxSize) {
        errors.push(`File size exceeds ${Math.round(maxSize / (1024 * 1024))}MB limit`)
      }
      
      return {
        filename: file.name,
        valid: errors.length === 0,
        errors,
        file_size: file.size,
        file_type: extension
      }
    })
    
    return {
      total_files: files.length,
      valid_files: results.filter(r => r.valid).length,
      invalid_files: results.filter(r => !r.valid).length,
      results
    }
  }

  // ===== SEARCH API =====
  static async searchDocuments(query: string, project_id?: string): Promise<any> {
    const response = await api.post('/api/search', {
      query,
      project_id,
      top_k: 5
    })
    return response.data
  }

  // ===== ADMIN APIs =====
  static async getAdminStats(): Promise<any> {
    const response = await api.get('/api/admin/stats')
    return response.data
  }

  static async reindexDocuments(): Promise<any> {
    const response = await api.post('/api/admin/reindex')
    return response.data
  }
}

// Export only the ApiService class as default
export default ApiService