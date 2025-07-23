// frontend/services/api.ts - Vollständig aktualisiert für ChromaDB Backend
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
    console.error('📄 Error Details:', error.response?.data)
    return Promise.reject(error)
  }
)

// === TYPE DEFINITIONS ===

export interface ApiResponse<T = any> {
  data: T
  status: number
  statusText: string
}

// Updated Project interface for ChromaDB
export interface Project {
  id: string
  name: string
  description: string
  created_at: string
  document_count: number
  chat_count: number
}

// Updated Document interface for ChromaDB
export interface Document {
  id: string
  filename: string
  file_size: number
  file_type: string
  processing_status: string
  created_at: string
  project_ids: string[]
  processing_method?: string
  text_length?: number
}

// Chat interfaces
export interface ChatRequest {
  message: string
  project_id?: string
  model?: string
  temperature?: number
  max_tokens?: number
}

export interface ChatResponse {
  response: string
  chat_id: string
  project_id?: string
  timestamp: string
  model_info: {
    model: string
    temperature: number
    features_used: Record<string, boolean>
    context_documents: number
  }
  sources: Array<{
    id: string
    name: string
    filename: string
    excerpt: string
    relevance_score: number
  }>
  intelligence_metadata: {
    query_complexity: string
    reasoning_depth: string
    context_integration: string
  }
}

// Search interfaces
export interface SearchRequest {
  query: string
  project_id?: string
  top_k?: number
}

export interface SearchResponse {
  query: string
  project_id?: string
  top_k: number
  results: Array<{
    id: string
    filename: string
    excerpt: string
    full_text: string
    relevance_score: number
    metadata: Record<string, any>
  }>
  total_results: number
  search_metadata: {
    embedding_model: string
    search_timestamp: string
  }
}

// System info interface
export interface SystemInfo {
  app: {
    name: string
    version: string
    environment: string
  }
  database: {
    type: string
    location: string
    stats: {
      projects: { total: number }
      documents: { total: number, chunks_total: number }
      chats: { total: number }
    }
  }
  capabilities: {
    document_formats: string[]
    ocr_engines: string[]
    ai_providers: string[]
  }
  statistics: {
    projects: number
    documents: number
    document_chunks: number
    chats: number
  }
}

// Upload interfaces
export interface UploadResponse {
  message: string
  document: Document
  processing_info: {
    method: string
    [key: string]: any
  }
}

// === API SERVICE CLASS ===

export class ApiService {
  
  // === HEALTH & SYSTEM ===
  
  static async healthCheck(): Promise<any> {
    try {
      const response = await api.get('/api/health')
      return response.data
    } catch (error) {
      console.error('Health check failed:', error)
      throw error
    }
  }

  static async getConfiguration(): Promise<any> {
    try {
      const response = await api.get('/api/config')
      return response.data
    } catch (error) {
      console.error('Configuration fetch failed:', error)
      throw error
    }
  }

  static async getSystemInfo(): Promise<SystemInfo> {
    try {
      const response = await api.get('/api/system/info')
      return response.data
    } catch (error) {
      console.error('System info fetch failed:', error)
      throw error
    }
  }

  static async getAIInfo(): Promise<any> {
    try {
      const response = await api.get('/api/ai/info')
      return response.data
    } catch (error) {
      console.error('AI info fetch failed:', error)
      throw error
    }
  }

  // === PROJECT APIs ===

  static async getProjects(): Promise<Project[]> {
    try {
      const response = await api.get('/api/projects')
      // ChromaDB backend returns array directly
      return Array.isArray(response.data) ? response.data : []
    } catch (error) {
      console.error('Projects fetch failed:', error)
      throw error
    }
  }

  static async createProject(data: {
    name: string
    description?: string
  }): Promise<Project> {
    try {
      const response = await api.post('/api/projects', data)
      return response.data
    } catch (error) {
      console.error('Project creation failed:', error)
      throw error
    }
  }

  static async getProject(id: string): Promise<any> {
    try {
      const response = await api.get(`/api/projects/${id}`)
      return response.data
    } catch (error) {
      console.error(`Project fetch failed for ID ${id}:`, error)
      throw error
    }
  }

  static async updateProject(id: string, data: {
    name?: string
    description?: string
  }): Promise<Project> {
    try {
      const response = await api.put(`/api/projects/${id}`, data)
      return response.data
    } catch (error) {
      console.error(`Project update failed for ID ${id}:`, error)
      throw error
    }
  }

  static async deleteProject(id: string): Promise<{ 
    message: string
    details?: {
      documents_affected: number
      chats_deleted: number
      total_file_size_freed?: number
    }
  }> {
    try {
      const response = await api.delete(`/api/projects/${id}`)
      return response.data
    } catch (error) {
      console.error(`Project deletion failed for ID ${id}:`, error)
      throw error
    }
  }

  // === DOCUMENT APIs ===

  static async getDocuments(project_id?: string): Promise<Document[]> {
    try {
      const params = project_id ? { project_id } : {}
      const response = await api.get('/api/documents', { params })
      return Array.isArray(response.data) ? response.data : []
    } catch (error) {
      console.error('Documents fetch failed:', error)
      throw error
    }
  }

  static async uploadDocument(data: {
    file: File
    project_id: string
    use_ocr?: boolean
    ocr_language?: string
    ocr_engine?: string
  }): Promise<UploadResponse> {
    try {
      const formData = new FormData()
      formData.append('file', data.file)
      formData.append('project_id', data.project_id)
      
      if (data.use_ocr !== undefined) {
        formData.append('use_ocr', data.use_ocr.toString())
      }
      if (data.ocr_language) {
        formData.append('ocr_language', data.ocr_language)
      }
      if (data.ocr_engine) {
        formData.append('ocr_engine', data.ocr_engine)
      }

      const response = await api.post('/api/documents/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        timeout: 120000, // 2 minutes for file processing
      })
      
      return response.data
    } catch (error) {
      console.error('Document upload failed:', error)
      throw error
    }
  }

  static async uploadDocumentsBatch(data: {
    files: File[]
    project_id: string
    use_ocr?: boolean
    ocr_language?: string
    ocr_engine?: string
  }): Promise<any> {
    try {
      const formData = new FormData()
      
      data.files.forEach(file => {
        formData.append('files', file)
      })
      
      formData.append('project_id', data.project_id)
      
      if (data.use_ocr !== undefined) {
        formData.append('use_ocr', data.use_ocr.toString())
      }
      if (data.ocr_language) {
        formData.append('ocr_language', data.ocr_language)
      }
      if (data.ocr_engine) {
        formData.append('ocr_engine', data.ocr_engine)
      }

      const response = await api.post('/api/documents/upload-batch', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        timeout: 300000, // 5 minutes for batch processing
      })
      
      return response.data
    } catch (error) {
      console.error('Batch document upload failed:', error)
      throw error
    }
  }

  static async deleteDocument(id: string): Promise<{ message: string; document_id: string }> {
    try {
      const response = await api.delete(`/api/documents/${id}`)
      return response.data
    } catch (error) {
      console.error(`Document deletion failed for ID ${id}:`, error)
      throw error
    }
  }

  // === SEARCH APIs ===

  static async searchDocuments(request: SearchRequest): Promise<SearchResponse> {
    try {
      const response = await api.post('/api/search', request)
      return response.data
    } catch (error) {
      console.error('Document search failed:', error)
      throw error
    }
  }

  // === CHAT APIs ===

  static async sendChatMessage(request: ChatRequest): Promise<ChatResponse> {
    try {
      const response = await api.post('/api/chat', request)
      return response.data
    } catch (error) {
      console.error('Chat message failed:', error)
      throw error
    }
  }

  static async getChats(project_id?: string): Promise<Array<{
    id: string
    project_id?: string
    created_at: string
    updated_at: string
    message_count: number
    last_message: string
    last_message_timestamp: string
  }>> {
    try {
      const params = project_id ? { project_id } : {}
      const response = await api.get('/api/chats', { params })
      return Array.isArray(response.data) ? response.data : []
    } catch (error) {
      console.error('Chats fetch failed:', error)
      throw error
    }
  }

  static async getChat(chat_id: string): Promise<any> {
    try {
      const response = await api.get(`/api/chats/${chat_id}`)
      return response.data
    } catch (error) {
      console.error(`Chat fetch failed for ID ${chat_id}:`, error)
      throw error
    }
  }

  static async deleteChat(chat_id: string): Promise<{ message: string; chat_id: string }> {
    try {
      const response = await api.delete(`/api/chats/${chat_id}`)
      return response.data
    } catch (error) {
      console.error(`Chat deletion failed for ID ${chat_id}:`, error)
      throw error
    }
  }

  // === OCR APIs ===

  static async processImageOCR(data: {
    file: File
    project_id: string
    language?: string
    engine?: string
  }): Promise<any> {
    try {
      const formData = new FormData()
      formData.append('file', data.file)
      formData.append('project_id', data.project_id)
      
      if (data.language) {
        formData.append('language', data.language)
      }
      if (data.engine) {
        formData.append('engine', data.engine)
      }

      const response = await api.post('/api/ocr/process-image', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        timeout: 120000,
      })
      
      return response.data
    } catch (error) {
      console.error('OCR image processing failed:', error)
      throw error
    }
  }

  static async processScannedPDFOCR(data: {
    file: File
    project_id: string
    language?: string
    engine?: string
  }): Promise<any> {
    try {
      const formData = new FormData()
      formData.append('file', data.file)
      formData.append('project_id', data.project_id)
      
      if (data.language) {
        formData.append('language', data.language)
      }
      if (data.engine) {
        formData.append('engine', data.engine)
      }

      const response = await api.post('/api/ocr/process-pdf', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        timeout: 300000, // 5 minutes for PDF OCR
      })
      
      return response.data
    } catch (error) {
      console.error('OCR PDF processing failed:', error)
      throw error
    }
  }

  static async getOCRStatus(): Promise<any> {
    try {
      const response = await api.get('/api/ocr/status')
      return response.data
    } catch (error) {
      console.error('OCR status fetch failed:', error)
      throw error
    }
  }

  // === BATCH PROCESSING APIs ===

  static async batchProcessDirectory(data: {
    directory_path: string
    project_id: string
    recursive?: boolean
    use_ocr?: boolean
    ocr_language?: string
    file_pattern?: string
  }): Promise<any> {
    try {
      const response = await api.post('/api/batch/process-directory', data, {
        timeout: 600000, // 10 minutes for directory processing
      })
      return response.data
    } catch (error) {
      console.error('Batch directory processing failed:', error)
      throw error
    }
  }

  // === ADMIN APIs ===

  static async getAdminStats(): Promise<any> {
    try {
      const response = await api.get('/api/admin/stats')
      return response.data
    } catch (error) {
      console.error('Admin stats fetch failed:', error)
      throw error
    }
  }

  static async reindexDocuments(): Promise<any> {
    try {
      const response = await api.post('/api/admin/reindex')
      return response.data
    } catch (error) {
      console.error('Document reindexing failed:', error)
      throw error
    }
  }

  static async getSystemDiagnostics(): Promise<any> {
    try {
      const response = await api.get('/api/admin/system-diagnostics')
      return response.data
    } catch (error) {
      console.error('System diagnostics failed:', error)
      throw error
    }
  }

  // === UTILITY METHODS ===

  static async testConnection(): Promise<{ status: string; message: string; details: any }> {
    try {
      const health = await this.healthCheck()
      return {
        status: health.status === 'healthy' ? 'success' : 'warning',
        message: `Backend is ${health.status}`,
        details: health
      }
    } catch (error) {
      return {
        status: 'error',
        message: 'Backend connection failed',
        details: error
      }
    }
  }

  static async getAvailableModels(): Promise<any> {
    try {
      const aiInfo = await this.getAIInfo()
      const models = []
      
      if (aiInfo.providers?.google_ai?.available) {
        models.push({
          id: 'gemini-1.5-flash',
          name: 'Gemini 1.5 Flash',
          provider: 'Google AI'
        })
      }
      
      if (aiInfo.providers?.openai?.available) {
        models.push({
          id: 'gpt-3.5-turbo',
          name: 'GPT-3.5 Turbo',
          provider: 'OpenAI'
        })
      }
      
      return {
        models,
        default_model: models[0]?.id || 'gemini-1.5-flash'
      }
    } catch (error) {
      console.error('Available models fetch failed:', error)
      throw error
    }
  }

  // === VALIDATION HELPERS ===

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
    try {
      const config = await this.getConfiguration()
      const maxSize = config.app?.max_file_size || (50 * 1024 * 1024) // 50MB default
      const allowedTypes = ['.pdf', '.docx', '.doc', '.txt', '.md', '.png', '.jpg', '.jpeg']
      
      const results = files.map(file => {
        const errors: string[] = []
        const extension = '.' + file.name.split('.').pop()?.toLowerCase()
        
        if (!allowedTypes.includes(extension)) {
          errors.push(`File type ${extension} not supported`)
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
    } catch (error) {
      console.error('File validation failed:', error)
      throw error
    }
  }
}

// Export the ApiService as default
export default ApiService

// Export commonly used functions for convenience
export const {
  healthCheck,
  getProjects,
  createProject,
  updateProject,
  deleteProject,
  getDocuments,
  uploadDocument,
  deleteDocument,
  searchDocuments,
  sendChatMessage,
  getChats,
  testConnection
} = ApiService