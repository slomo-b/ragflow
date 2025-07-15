import React, { useState, useRef, useCallback, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  DocumentTextIcon,
  CloudArrowUpIcon,
  EyeIcon,
  TrashIcon,
  ArrowDownTrayIcon,
  MagnifyingGlassIcon,
  FunnelIcon,
  CheckCircleIcon,
  ClockIcon,
  ExclamationTriangleIcon,
  PlusIcon,
  SparklesIcon,
  FolderIcon,
  TagIcon,
  CalendarIcon,
  ArrowPathIcon,
  XMarkIcon
} from '@heroicons/react/24/outline'
import {
  DocumentTextIcon as DocumentTextSolidIcon,
  CheckCircleIcon as CheckCircleSolidIcon,
  ClockIcon as ClockSolidIcon,
  ExclamationTriangleIcon as ExclamationTriangleSolidIcon,
} from '@heroicons/react/24/solid'
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/Card"
import { Button } from "@/components/ui/Button"
import { Input } from "@/components/ui/Input"
import { Badge } from "@/components/ui/Badge"
import { 
  DropdownMenu, 
  DropdownMenuContent, 
  DropdownMenuItem, 
  DropdownMenuTrigger,
  DropdownMenuSeparator,
  DropdownMenuLabel
} from "@/components/ui/dropdown-menu"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import { cn } from '@/lib/utils'
import toast from 'react-hot-toast'

// API Service
const API_BASE = 'http://localhost:8000'

class DocumentAPI {
  static async uploadDocuments(files: FileList | File[], projectId: string) {
    console.log('Starting upload...', { fileCount: files.length, projectId })
    
    const formData = new FormData()
    
    Array.from(files).forEach((file, index) => {
      console.log(`Adding file ${index + 1}:`, file.name, file.type, file.size)
      formData.append('files', file)
    })
    
    formData.append('project_id', projectId)
    
    // Debug FormData contents
    console.log('FormData contents:')
    for (let [key, value] of formData.entries()) {
      console.log(key, value)
    }
    
    try {
      const response = await fetch(`${API_BASE}/api/v1/upload/documents`, {
        method: 'POST',
        body: formData,
      })
      
      console.log('Response status:', response.status)
      console.log('Response headers:', Object.fromEntries(response.headers.entries()))
      
      if (!response.ok) {
        const errorText = await response.text()
        console.error('Upload error response:', errorText)
        throw new Error(`Upload failed: ${response.status} ${response.statusText} - ${errorText}`)
      }
      
      const result = await response.json()
      console.log('Upload success:', result)
      return result
    } catch (error) {
      console.error('Upload fetch error:', error)
      throw error
    }
  }
  
  static async getDocuments(projectId?: string, skip = 0, limit = 100) {
    const params = new URLSearchParams({
      skip: skip.toString(),
      limit: limit.toString(),
    })
    
    if (projectId) {
      params.append('project_id', projectId)
    }
    
    console.log('Fetching documents with params:', Object.fromEntries(params))
    
    const response = await fetch(`${API_BASE}/api/v1/documents/?${params}`)
    
    console.log('Documents response status:', response.status)
    
    if (!response.ok) {
      const errorText = await response.text()
      console.error('Documents fetch error:', errorText)
      throw new Error(`Failed to fetch documents: ${response.status} ${response.statusText}`)
    }
    
    const result = await response.json()
    console.log('Documents result:', result)
    return result
  }
  
  static async deleteDocument(documentId: string) {
    console.log('Deleting document:', documentId)
    
    const response = await fetch(`${API_BASE}/api/v1/documents/${documentId}`, {
      method: 'DELETE',
    })
    
    console.log('Delete response status:', response.status)
    
    if (!response.ok) {
      const errorText = await response.text()
      console.error('Delete error:', errorText)
      throw new Error(`Delete failed: ${response.status} ${response.statusText}`)
    }
    
    const result = await response.json()
    console.log('Delete result:', result)
    return result
  }
  
  static async getProjects() {
    console.log('Fetching projects...')
    
    const response = await fetch(`${API_BASE}/api/v1/projects/`)
    
    console.log('Projects response status:', response.status)
    
    if (!response.ok) {
      const errorText = await response.text()
      console.error('Projects fetch error:', errorText)
      throw new Error(`Failed to fetch projects: ${response.status} ${response.statusText}`)
    }
    
    const result = await response.json()
    console.log('Projects result:', result)
    return result
  }
  
  static async createProject(name: string, description = '') {
    const response = await fetch(`${API_BASE}/api/v1/projects/`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ name, description }),
    })
    
    if (!response.ok) {
      throw new Error(`Project creation failed: ${response.statusText}`)
    }
    
    return response.json()
  }
}

export function DocumentLibrary() {
  const [documents, setDocuments] = useState([])
  const [projects, setProjects] = useState([])
  const [currentProject, setCurrentProject] = useState(null)
  const [loading, setLoading] = useState(true)
  const [searchTerm, setSearchTerm] = useState('')
  const [selectedStatus, setSelectedStatus] = useState('all')
  const [sortBy, setSortBy] = useState('uploadedAt')
  const [isUploading, setIsUploading] = useState(false)
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false)
  const [documentToDelete, setDocumentToDelete] = useState(null)
  const [isDragOver, setIsDragOver] = useState(false)
  const fileInputRef = useRef(null)

  // Load initial data
  useEffect(() => {
    loadInitialData()
  }, [])

  // Load documents when project changes
  useEffect(() => {
    if (currentProject) {
      loadDocuments()
    }
  }, [currentProject])

  const loadInitialData = async () => {
    console.log('Loading initial data...')
    
    try {
      setLoading(true)
      const projectsData = await DocumentAPI.getProjects()
      console.log('Loaded projects:', projectsData)
      
      setProjects(projectsData.projects || [])
      
      if (projectsData.projects && projectsData.projects.length > 0) {
        console.log('Setting current project to:', projectsData.projects[0])
        setCurrentProject(projectsData.projects[0])
      } else {
        console.log('No projects found, creating default project...')
        // Create a default project if none exist
        try {
          const defaultProject = await DocumentAPI.createProject('My Documents', 'Default project for documents')
          console.log('Created default project:', defaultProject)
          setProjects([defaultProject])
          setCurrentProject(defaultProject)
        } catch (createError) {
          console.error('Failed to create default project:', createError)
        }
      }
    } catch (error) {
      console.error('Failed to load initial data:', error)
      toast.error(`Failed to load projects: ${error.message}`)
    } finally {
      setLoading(false)
    }
  }

  const loadDocuments = async () => {
    if (!currentProject) {
      console.log('No current project, skipping document load')
      return
    }
    
    console.log('Loading documents for project:', currentProject.id)
    
    try {
      const documentsData = await DocumentAPI.getDocuments(currentProject.id)
      console.log('Loaded documents:', documentsData)
      setDocuments(documentsData.documents || [])
    } catch (error) {
      console.error('Failed to load documents:', error)
      toast.error(`Failed to load documents: ${error.message}`)
    }
  }

  // Status-spezifische Konfiguration
  const getStatusConfig = (status) => {
    switch (status) {
      case 'completed':
        return {
          icon: CheckCircleSolidIcon,
          color: 'text-emerald-500',
          bgColor: 'bg-emerald-50 dark:bg-emerald-900/20',
          borderColor: 'border-emerald-200 dark:border-emerald-800',
          label: 'Completed',
          description: 'Ready for analysis'
        }
      case 'processing':
        return {
          icon: ClockSolidIcon,
          color: 'text-blue-500',
          bgColor: 'bg-blue-50 dark:bg-blue-900/20',
          borderColor: 'border-blue-200 dark:border-blue-800',
          label: 'Processing',
          description: 'AI is analyzing...'
        }
      case 'failed':
        return {
          icon: ExclamationTriangleSolidIcon,
          color: 'text-red-500',
          bgColor: 'bg-red-50 dark:bg-red-900/20',
          borderColor: 'border-red-200 dark:border-red-800',
          label: 'Failed',
          description: 'Processing failed'
        }
      default:
        return {
          icon: DocumentTextSolidIcon,
          color: 'text-slate-500',
          bgColor: 'bg-slate-50 dark:bg-slate-900/20',
          borderColor: 'border-slate-200 dark:border-slate-800',
          label: 'Uploaded',
          description: 'Waiting for processing'
        }
    }
  }

  // File Upload Handler
  const handleFileUpload = async (files) => {
    console.log('handleFileUpload called with:', files)
    
    if (!files || files.length === 0) {
      console.log('No files provided')
      return
    }
    
    if (!currentProject) {
      console.log('No current project selected')
      toast.error('Please select a project first')
      return
    }

    console.log('Current project:', currentProject)
    console.log('Files to upload:', Array.from(files).map(f => ({ name: f.name, type: f.type, size: f.size })))

    setIsUploading(true)
    
    try {
      const result = await DocumentAPI.uploadDocuments(files, currentProject.id)
      console.log('Upload result:', result)
      
      if (result && result.success !== false) {
        toast.success(`${files.length} document(s) uploaded successfully!`)
        await loadDocuments() // Reload documents
      } else {
        console.error('Upload result indicates failure:', result)
        toast.error(result?.error || 'Upload failed')
      }
    } catch (error) {
      console.error('Upload failed with error:', error)
      toast.error(`Upload failed: ${error.message}`)
    } finally {
      setIsUploading(false)
    }
  }

  // Drag & Drop Handler
  const handleDragOver = useCallback((e) => {
    e.preventDefault()
    setIsDragOver(true)
  }, [])

  const handleDragLeave = useCallback((e) => {
    e.preventDefault()
    setIsDragOver(false)
  }, [])

  const handleDrop = useCallback((e) => {
    e.preventDefault()
    setIsDragOver(false)
    const files = e.dataTransfer.files
    handleFileUpload(files)
  }, [currentProject])

  // Delete Document Handler
  const handleDeleteDocument = (document) => {
    setDocumentToDelete(document)
    setDeleteDialogOpen(true)
  }

  const confirmDelete = async () => {
    if (!documentToDelete) return

    try {
      await DocumentAPI.deleteDocument(documentToDelete.id)
      toast.success('Document deleted successfully')
      await loadDocuments() // Reload documents
    } catch (error) {
      console.error('Delete failed:', error)
      toast.error('Failed to delete document')
    } finally {
      setDeleteDialogOpen(false)
      setDocumentToDelete(null)
    }
  }

  // Filter and sort documents
  const filteredDocuments = documents
    .filter(doc => {
      const matchesSearch = doc.filename?.toLowerCase().includes(searchTerm.toLowerCase())
      const matchesStatus = selectedStatus === 'all' || doc.processing_status === selectedStatus
      return matchesSearch && matchesStatus
    })
    .sort((a, b) => {
      switch (sortBy) {
        case 'name':
          return (a.filename || '').localeCompare(b.filename || '')
        case 'size':
          return (b.file_size || 0) - (a.file_size || 0)
        case 'uploadedAt':
        default:
          return new Date(b.uploaded_at || 0).getTime() - new Date(a.uploaded_at || 0).getTime()
      }
    })

  // Statistics
  const stats = {
    total: documents.length,
    completed: documents.filter(doc => doc.processing_status === 'completed').length,
    processing: documents.filter(doc => doc.processing_status === 'processing').length,
    failed: documents.filter(doc => doc.processing_status === 'failed').length,
    totalSize: documents.reduce((sum, doc) => sum + (doc.file_size || 0), 0)
  }

  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 Bytes'
    const k = 1024
    const sizes = ['Bytes', 'KB', 'MB', 'GB']
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
  }

  const formatDate = (dateString) => {
    return new Date(dateString).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    })
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="animate-spin rounded-full h-32 w-32 border-b-2 border-blue-500"></div>
      </div>
    )
  }

  return (
    <div 
      className={cn(
        "h-full transition-colors duration-300",
        isDragOver 
          ? "bg-gradient-to-br from-blue-50 to-violet-50 dark:from-blue-900/20 dark:to-violet-900/20" 
          : "bg-gradient-to-br from-slate-50 to-white dark:from-slate-900 dark:to-slate-800"
      )}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
    >
      {/* Header */}
      <div className="sticky top-0 z-10 bg-white/80 dark:bg-slate-900/80 backdrop-blur-xl border-b border-slate-200/50 dark:border-slate-700/50 p-6">
        <div className="flex flex-col lg:flex-row gap-4 justify-between items-start lg:items-center">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-gradient-to-r from-blue-500 to-violet-500 rounded-xl">
              <FolderIcon className="h-6 w-6 text-white" />
            </div>
            <div>
              <h1 className="text-2xl font-bold text-slate-900 dark:text-slate-100">
                Document Library
              </h1>
              <p className="text-sm text-slate-600 dark:text-slate-400">
                {currentProject ? `Project: ${currentProject.name}` : 'No project selected'}
              </p>
            </div>
          </div>

          <div className="flex items-center gap-3">
            <Button
              onClick={() => fileInputRef.current?.click()}
              disabled={isUploading || !currentProject}
              className="bg-gradient-to-r from-blue-500 to-violet-500 hover:from-blue-600 hover:to-violet-600 text-white"
            >
              {isUploading ? (
                <>
                  <ArrowPathIcon className="h-4 w-4 mr-2 animate-spin" />
                  Uploading...
                </>
              ) : (
                <>
                  <CloudArrowUpIcon className="h-4 w-4 mr-2" />
                  Upload Documents
                </>
              )}
            </Button>
            
            <input
              ref={fileInputRef}
              type="file"
              multiple
              accept=".txt,.pdf,.docx,.md"
              onChange={(e) => handleFileUpload(e.target.files)}
              className="hidden"
            />
          </div>
        </div>

        {/* Statistics */}
        <div className="grid grid-cols-2 md:grid-cols-5 gap-4 mt-6">
          <Card className="bg-white/50 dark:bg-slate-800/50 backdrop-blur-sm">
            <CardContent className="p-4">
              <div className="flex items-center gap-2">
                <DocumentTextIcon className="h-5 w-5 text-slate-500" />
                <div>
                  <p className="text-sm text-slate-600 dark:text-slate-400">Total</p>
                  <p className="text-xl font-semibold text-slate-900 dark:text-slate-100">
                    {stats.total}
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card className="bg-white/50 dark:bg-slate-800/50 backdrop-blur-sm">
            <CardContent className="p-4">
              <div className="flex items-center gap-2">
                <CheckCircleIcon className="h-5 w-5 text-emerald-500" />
                <div>
                  <p className="text-sm text-slate-600 dark:text-slate-400">Completed</p>
                  <p className="text-xl font-semibold text-emerald-600">
                    {stats.completed}
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card className="bg-white/50 dark:bg-slate-800/50 backdrop-blur-sm">
            <CardContent className="p-4">
              <div className="flex items-center gap-2">
                <ClockIcon className="h-5 w-5 text-blue-500" />
                <div>
                  <p className="text-sm text-slate-600 dark:text-slate-400">Processing</p>
                  <p className="text-xl font-semibold text-blue-600">
                    {stats.processing}
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card className="bg-white/50 dark:bg-slate-800/50 backdrop-blur-sm">
            <CardContent className="p-4">
              <div className="flex items-center gap-2">
                <ExclamationTriangleIcon className="h-5 w-5 text-red-500" />
                <div>
                  <p className="text-sm text-slate-600 dark:text-slate-400">Failed</p>
                  <p className="text-xl font-semibold text-red-600">
                    {stats.failed}
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card className="bg-white/50 dark:bg-slate-800/50 backdrop-blur-sm">
            <CardContent className="p-4">
              <div className="flex items-center gap-2">
                <ArrowDownTrayIcon className="h-5 w-5 text-slate-500" />
                <div>
                  <p className="text-sm text-slate-600 dark:text-slate-400">Total Size</p>
                  <p className="text-xl font-semibold text-slate-900 dark:text-slate-100">
                    {formatFileSize(stats.totalSize)}
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Filters */}
        <div className="flex flex-col sm:flex-row gap-4 mt-6">
          <div className="relative flex-1">
            <MagnifyingGlassIcon className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-slate-400" />
            <Input
              placeholder="Search documents..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="pl-10 bg-white/50 dark:bg-slate-800/50 backdrop-blur-sm"
            />
          </div>

          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="outline" className="bg-white/50 dark:bg-slate-800/50 backdrop-blur-sm">
                <FunnelIcon className="h-4 w-4 mr-2" />
                Status: {selectedStatus === 'all' ? 'All' : selectedStatus}
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent>
              <DropdownMenuLabel>Filter by Status</DropdownMenuLabel>
              <DropdownMenuSeparator />
              <DropdownMenuItem onClick={() => setSelectedStatus('all')}>
                All Documents
              </DropdownMenuItem>
              <DropdownMenuItem onClick={() => setSelectedStatus('completed')}>
                Completed
              </DropdownMenuItem>
              <DropdownMenuItem onClick={() => setSelectedStatus('processing')}>
                Processing
              </DropdownMenuItem>
              <DropdownMenuItem onClick={() => setSelectedStatus('failed')}>
                Failed
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>

          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="outline" className="bg-white/50 dark:bg-slate-800/50 backdrop-blur-sm">
                Sort by: {sortBy === 'uploadedAt' ? 'Upload Date' : sortBy === 'name' ? 'Name' : 'Size'}
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent>
              <DropdownMenuLabel>Sort by</DropdownMenuLabel>
              <DropdownMenuSeparator />
              <DropdownMenuItem onClick={() => setSortBy('uploadedAt')}>
                Upload Date
              </DropdownMenuItem>
              <DropdownMenuItem onClick={() => setSortBy('name')}>
                Name
              </DropdownMenuItem>
              <DropdownMenuItem onClick={() => setSortBy('size')}>
                Size
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        </div>
      </div>

      {/* Content */}
      <div className="p-6">
        {isDragOver && (
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            className="fixed inset-0 z-50 flex items-center justify-center bg-blue-500/20 backdrop-blur-sm"
          >
            <div className="text-center">
              <CloudArrowUpIcon className="h-16 w-16 text-blue-500 mx-auto mb-4" />
              <h3 className="text-2xl font-bold text-blue-600 mb-2">Drop files here</h3>
              <p className="text-blue-500">Release to upload to {currentProject?.name}</p>
            </div>
          </motion.div>
        )}

        {!currentProject ? (
          <div className="text-center py-12">
            <FolderIcon className="h-16 w-16 text-slate-400 mx-auto mb-4" />
            <h3 className="text-xl font-semibold text-slate-900 dark:text-slate-100 mb-2">
              No Project Selected
            </h3>
            <p className="text-slate-600 dark:text-slate-400 mb-6">
              Please select or create a project to manage documents.
            </p>
          </div>
        ) : filteredDocuments.length === 0 ? (
          <div className="text-center py-12">
            <DocumentTextIcon className="h-16 w-16 text-slate-400 mx-auto mb-4" />
            <h3 className="text-xl font-semibold text-slate-900 dark:text-slate-100 mb-2">
              {searchTerm ? 'No documents found' : `No documents in ${currentProject.name}`}
            </h3>
            <p className="text-slate-600 dark:text-slate-400 mb-6 max-w-lg mx-auto">
              {searchTerm 
                ? `No documents match your search "${searchTerm}". Try adjusting your filters.`
                : `Drag and drop files here or click the upload button to add documents to ${currentProject.name}.`
              }
            </p>
            
            {!searchTerm && (
              <Button
                onClick={() => fileInputRef.current?.click()}
                className="bg-gradient-to-r from-violet-500 to-blue-500 hover:from-violet-600 hover:to-blue-600 text-white px-8 py-3 rounded-2xl shadow-lg"
              >
                <PlusIcon className="h-5 w-5 mr-2" />
                Upload Your First Document
              </Button>
            )}
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {filteredDocuments.map((doc, index) => {
              const statusConfig = getStatusConfig(doc.processing_status || 'uploaded')
              
              return (
                <motion.div
                  key={doc.id}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.3, delay: index * 0.1 }}
                  whileHover={{ y: -4, scale: 1.02 }}
                  className="group"
                >
                  <Card className="h-full bg-white/70 dark:bg-slate-800/70 backdrop-blur-xl border-slate-200/50 dark:border-slate-700/50 hover:shadow-xl transition-all duration-300 overflow-hidden">
                    <CardHeader className="pb-4">
                      <div className="flex items-start justify-between">
                        <div className="flex items-center gap-3">
                          <div className={cn("p-2 rounded-lg", statusConfig.bgColor)}>
                            <statusConfig.icon className={cn("h-5 w-5", statusConfig.color)} />
                          </div>
                          <div className="flex-1 min-w-0">
                            <h3 className="font-semibold text-slate-900 dark:text-slate-100 truncate">
                              {doc.filename || 'Untitled Document'}
                            </h3>
                            <p className="text-sm text-slate-600 dark:text-slate-400">
                              {formatFileSize(doc.file_size || 0)}
                            </p>
                          </div>
                        </div>
                        
                        <DropdownMenu>
                          <DropdownMenuTrigger asChild>
                            <Button 
                              variant="ghost" 
                              size="sm"
                              className="opacity-0 group-hover:opacity-100 transition-opacity"
                            >
                              <svg className="h-4 w-4" fill="currentColor" viewBox="0 0 20 20">
                                <path d="M10 6a2 2 0 110-4 2 2 0 010 4zM10 12a2 2 0 110-4 2 2 0 010 4zM10 18a2 2 0 110-4 2 2 0 010 4z" />
                              </svg>
                            </Button>
                          </DropdownMenuTrigger>
                          <DropdownMenuContent>
                            <DropdownMenuItem>
                              <EyeIcon className="h-4 w-4 mr-2" />
                              View Details
                            </DropdownMenuItem>
                            <DropdownMenuItem>
                              <ArrowDownTrayIcon className="h-4 w-4 mr-2" />
                              Download
                            </DropdownMenuItem>
                            <DropdownMenuSeparator />
                            <DropdownMenuItem 
                              onClick={() => handleDeleteDocument(doc)}
                              className="text-red-600 focus:text-red-600"
                            >
                              <TrashIcon className="h-4 w-4 mr-2" />
                              Delete
                            </DropdownMenuItem>
                          </DropdownMenuContent>
                        </DropdownMenu>
                      </div>
                    </CardHeader>
                    
                    <CardContent className="pt-0">
                      <div className="space-y-3">
                        <div className="flex items-center justify-between">
                          <Badge 
                            variant="secondary" 
                            className={cn("text-xs", statusConfig.bgColor, statusConfig.color)}
                          >
                            {statusConfig.label}
                          </Badge>
                          <span className="text-xs text-slate-500">
                            {doc.file_type || 'Unknown type'}
                          </span>
                        </div>
                        
                        <p className="text-sm text-slate-600 dark:text-slate-400">
                          {statusConfig.description}
                        </p>
                        
                        <div className="flex items-center gap-2 text-xs text-slate-500">
                          <CalendarIcon className="h-3 w-3" />
                          <span>
                            {doc.uploaded_at ? formatDate(doc.uploaded_at) : 'Unknown date'}
                          </span>
                        </div>
                        
                        {doc.processing_error && (
                          <div className="p-2 bg-red-50 dark:bg-red-900/20 rounded-lg">
                            <p className="text-xs text-red-600 dark:text-red-400">
                              {doc.processing_error}
                            </p>
                          </div>
                        )}
                      </div>
                    </CardContent>
                  </Card>
                </motion.div>
              )
            })}
          </div>
        )}
      </div>

      {/* Delete Confirmation Dialog */}
      <Dialog open={deleteDialogOpen} onOpenChange={setDeleteDialogOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Delete Document</DialogTitle>
            <DialogDescription>
              Are you sure you want to delete "{documentToDelete?.filename}"? This action cannot be undone.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button 
              variant="outline" 
              onClick={() => setDeleteDialogOpen(false)}
            >
              Cancel
            </Button>
            <Button 
              variant="destructive" 
              onClick={confirmDelete}
            >
              Delete
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  )
}