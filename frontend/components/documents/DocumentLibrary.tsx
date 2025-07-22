// frontend/components/documents/DocumentLibrary.tsx - An neue API angepasst
'use client'

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
import { Progress } from "@/components/ui/progress"
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
import { cn, formatFileSize, formatDate } from '@/lib/utils'
import ApiService from '@/services/api'
import toast from 'react-hot-toast'

// Types
interface Document {
  id: string
  filename: string
  file_type: string
  file_size: number
  uploaded_at?: string  // Backend liefert uploaded_at
  created_at?: string   // Fallback für ältere Versionen
  processing_status: 'pending' | 'processing' | 'completed' | 'failed'
  project_ids: string[]
  tags: string[]
  summary?: string
}

interface ExtendedDocument extends Document {
  uploading?: boolean
  uploadProgress?: number
}

interface UploadingFile {
  id: string
  file: File
  progress: number
  status: 'uploading' | 'processing' | 'completed' | 'error'
  error?: string
}

interface Project {
  id: string
  name: string
  description?: string
  document_count: number
}

export const DocumentLibrary: React.FC = () => {
  // State Management
  const [documents, setDocuments] = useState<ExtendedDocument[]>([])
  const [projects, setProjects] = useState<Project[]>([])
  const [selectedProjectId, setSelectedProjectId] = useState<string | null>(null)
  const [searchQuery, setSearchQuery] = useState('')
  const [sortBy, setSortBy] = useState<'name' | 'date' | 'size' | 'status'>('date')
  const [filterStatus, setFilterStatus] = useState<'all' | 'completed' | 'processing' | 'failed'>('all')
  const [uploadingFiles, setUploadingFiles] = useState<UploadingFile[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [selectedDocument, setSelectedDocument] = useState<ExtendedDocument | null>(null)
  const [connectionStatus, setConnectionStatus] = useState<'checking' | 'connected' | 'disconnected'>('checking')

  // Refs
  const fileInputRef = useRef<HTMLInputElement>(null)
  const dropZoneRef = useRef<HTMLDivElement>(null)

  // Load data on mount
  useEffect(() => {
    initializeLibrary()
  }, [])

  // Load documents when project selection changes
  useEffect(() => {
    if (selectedProjectId) {
      loadDocuments()
    }
  }, [selectedProjectId])

  const initializeLibrary = async () => {
    console.log('📚 Initializing document library...')
    setConnectionStatus('checking')
    
    try {
      // Test connection
      const healthCheck = await ApiService.healthCheck()
      
      if (healthCheck.status === 'healthy') {
        setConnectionStatus('connected')
        console.log('✅ Backend connection established')
        
        // Load projects
        await loadProjects()
        await loadDocuments()
        
        toast.success('Document library ready!', { duration: 2000 })
      } else {
        throw new Error('Backend unhealthy')
      }
    } catch (error) {
      setConnectionStatus('disconnected')
      console.error('💥 Library initialization failed:', error)
      toast.error('Failed to connect to backend. Please check the server.')
    }
  }

  const loadProjects = async () => {
    try {
      const response = await ApiService.getProjects()
      const projectsData = response.projects || []
      
      setProjects(projectsData)
      console.log(`📁 Loaded ${projectsData.length} projects`)
      
      // Auto-select first project if available
      if (projectsData.length > 0 && !selectedProjectId) {
        setSelectedProjectId(projectsData[0].id)
        console.log(`🎯 Auto-selected project: ${projectsData[0].name}`)
      }
    } catch (error) {
      console.error('Failed to load projects:', error)
      toast.error('Failed to load projects')
    }
  }

  const loadDocuments = async () => {
    setIsLoading(true)
    try {
      const response = await ApiService.getDocuments({
        project_id: selectedProjectId || undefined
      })
      
      const documentsData = response.documents || []
      setDocuments(documentsData)
      console.log(`📄 Loaded ${documentsData.length} documents`)
    } catch (error) {
      console.error('Failed to load documents:', error)
      toast.error('Failed to load documents')
    } finally {
      setIsLoading(false)
    }
  }

  // File upload handling
  const handleFileUpload = useCallback(async (files: FileList | File[]) => {
    if (!selectedProjectId) {
      toast.error('Please select a project first')
      return
    }

    const fileArray = Array.from(files)
    
    // Validate files first
    try {
      const validation = await ApiService.validateFiles(fileArray)
      
      if (validation.invalid_files > 0) {
        const errors = validation.results
          .filter(r => !r.valid)
          .map(r => `${r.filename}: ${r.errors.join(', ')}`)
        
        toast.error(`Invalid files:\n${errors.join('\n')}`)
        return
      }
    } catch (error) {
      console.warn('File validation failed, proceeding with upload:', error)
    }

    const newUploadingFiles: UploadingFile[] = fileArray.map(file => ({
      id: `upload-${Date.now()}-${Math.random()}`,
      file,
      progress: 0,
      status: 'uploading'
    }))

    setUploadingFiles(prev => [...prev, ...newUploadingFiles])

    // Upload files
    try {
      const result = await ApiService.uploadDocuments({
        files: fileArray,
        project_id: selectedProjectId
      })

      // Update upload status to completed
      setUploadingFiles(prev => prev.map(f => 
        newUploadingFiles.some(nf => nf.id === f.id)
          ? { ...f, status: 'completed', progress: 100 }
          : f
      ))

      toast.success(`Successfully uploaded ${fileArray.length} file(s)`)

      // Remove completed uploads after delay
      setTimeout(() => {
        setUploadingFiles(prev => prev.filter(f => 
          !newUploadingFiles.some(nf => nf.id === f.id)
        ))
      }, 2000)

      // Reload documents
      setTimeout(() => {
        loadDocuments()
      }, 1000)

    } catch (error) {
      console.error('Upload error:', error)
      
      // Mark uploads as failed
      setUploadingFiles(prev => prev.map(f => 
        newUploadingFiles.some(nf => nf.id === f.id)
          ? { ...f, status: 'error', error: error instanceof Error ? error.message : 'Upload failed' }
          : f
      ))

      toast.error('Upload failed. Please try again.')
    }
  }, [selectedProjectId])

  // Drag and drop handling
  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
  }, [])

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    
    const files = e.dataTransfer.files
    if (files.length > 0) {
      handleFileUpload(files)
    }
  }, [handleFileUpload])

  // File input handling
  const handleFileInputChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files
    if (files && files.length > 0) {
      handleFileUpload(files)
    }
    // Reset input
    if (fileInputRef.current) {
      fileInputRef.current.value = ''
    }
  }, [handleFileUpload])

  // Document deletion
  const deleteDocument = useCallback(async (documentId: string) => {
    try {
      await ApiService.deleteDocument(documentId)
      setDocuments(prev => prev.filter(doc => doc.id !== documentId))
      toast.success('Document deleted successfully')
    } catch (error) {
      console.error('Delete error:', error)
      toast.error('Failed to delete document')
    }
  }, [])

  // Filter and sort documents
  const filteredDocuments = documents
    .filter(doc => {
      // Search filter
      if (searchQuery && !doc.filename.toLowerCase().includes(searchQuery.toLowerCase())) {
        return false
      }
      
      // Status filter
      if (filterStatus !== 'all' && doc.processing_status !== filterStatus) {
        return false
      }
      
      return true
    })
    .sort((a, b) => {
      switch (sortBy) {
        case 'name':
          return a.filename.localeCompare(b.filename)
        case 'date':
          const dateA = a.uploaded_at || a.created_at || '1970-01-01'
          const dateB = b.uploaded_at || b.created_at || '1970-01-01'
          return new Date(dateB).getTime() - new Date(dateA).getTime()
        case 'size':
          return b.file_size - a.file_size
        case 'status':
          return a.processing_status.localeCompare(b.processing_status)
        default:
          return 0
      }
    })

  // Upload Progress Component
  const UploadProgress: React.FC<{ uploadingFile: UploadingFile }> = ({ uploadingFile }) => (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -10 }}
      className="bg-white p-3 rounded-lg border border-gray-200 shadow-sm"
    >
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <DocumentTextIcon className="w-4 h-4 text-blue-600" />
          <span className="text-sm font-medium text-gray-900 truncate max-w-xs">
            {uploadingFile.file.name}
          </span>
        </div>
        <div className="flex items-center gap-2">
          {uploadingFile.status === 'completed' && (
            <CheckCircleIcon className="w-4 h-4 text-green-600" />
          )}
          {uploadingFile.status === 'error' && (
            <ExclamationTriangleIcon className="w-4 h-4 text-red-600" />
          )}
          <span className="text-xs text-gray-500">
            {formatFileSize(uploadingFile.file.size)}
          </span>
        </div>
      </div>
      
      {uploadingFile.status === 'error' && uploadingFile.error && (
        <p className="text-xs text-red-600 mb-2">{uploadingFile.error}</p>
      )}
      
      <div className="flex items-center gap-2">
        <Progress 
          value={uploadingFile.progress} 
          className="flex-1 h-1.5"
        />
        <span className="text-xs text-gray-500 min-w-[3rem]">
          {uploadingFile.progress}%
        </span>
      </div>
      
      <div className="flex items-center justify-between mt-1">
        <span className="text-xs text-gray-500 capitalize">
          {uploadingFile.status === 'uploading' ? 'Uploading...' :
           uploadingFile.status === 'processing' ? 'Processing...' :
           uploadingFile.status === 'completed' ? 'Completed!' :
           'Failed'}
        </span>
      </div>
    </motion.div>
  )

  // Document Card Component
  const DocumentCard: React.FC<{ document: ExtendedDocument }> = ({ document }) => {
    const getStatusIcon = () => {
      switch (document.processing_status) {
        case 'completed':
          return <CheckCircleSolidIcon className="w-4 h-4 text-green-600" />
        case 'processing':
          return <ClockSolidIcon className="w-4 h-4 text-yellow-600" />
        case 'failed':
          return <ExclamationTriangleSolidIcon className="w-4 h-4 text-red-600" />
        default:
          return <ClockIcon className="w-4 h-4 text-gray-400" />
      }
    }

    const getStatusColor = () => {
      switch (document.processing_status) {
        case 'completed':
          return 'bg-green-100 text-green-800'
        case 'processing':
          return 'bg-yellow-100 text-yellow-800'
        case 'failed':
          return 'bg-red-100 text-red-800'
        default:
          return 'bg-gray-100 text-gray-800'
      }
    }

    return (
      <motion.div
        layout
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        exit={{ opacity: 0, y: -20 }}
        className="bg-white border border-gray-200 rounded-lg p-4 hover:shadow-md transition-shadow"
      >
        <div className="flex items-start justify-between mb-3">
          <div className="flex items-center gap-3 flex-1 min-w-0">
            <DocumentTextSolidIcon className="w-8 h-8 text-blue-600 flex-shrink-0" />
            <div className="min-w-0 flex-1">
              <h3 className="font-medium text-gray-900 truncate" title={document.filename}>
                {document.filename}
              </h3>
              <div className="flex items-center gap-2 mt-1">
                <span className="text-xs text-gray-500">
                  {formatFileSize(document.file_size)}
                </span>
                <span className="text-xs text-gray-300">•</span>
                <span className="text-xs text-gray-500">
                  {formatDate(document.uploaded_at || document.created_at || '1970-01-01')}
                </span>
              </div>
            </div>
          </div>
          
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="ghost" size="sm" className="h-8 w-8 p-0">
                <span className="sr-only">Open menu</span>
                <svg className="h-4 w-4" fill="currentColor" viewBox="0 0 20 20">
                  <path d="M10 6a2 2 0 110-4 2 2 0 010 4zM10 12a2 2 0 110-4 2 2 0 010 4zM10 18a2 2 0 110-4 2 2 0 010 4z" />
                </svg>
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end">
              <DropdownMenuItem onClick={() => setSelectedDocument(document)}>
                <EyeIcon className="w-4 h-4 mr-2" />
                View Details
              </DropdownMenuItem>
              <DropdownMenuSeparator />
              <DropdownMenuItem 
                onClick={() => deleteDocument(document.id)}
                className="text-red-600"
              >
                <TrashIcon className="w-4 h-4 mr-2" />
                Delete
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        </div>

        {/* Status Badge */}
        <div className="flex items-center justify-between">
          <Badge className={cn("flex items-center gap-1", getStatusColor())}>
            {getStatusIcon()}
            {document.processing_status}
          </Badge>
          
          {document.tags.length > 0 && (
            <div className="flex items-center gap-1">
              <TagIcon className="w-3 h-3 text-gray-400" />
              <span className="text-xs text-gray-500">
                {document.tags.length} tags
              </span>
            </div>
          )}
        </div>

        {/* Summary */}
        {document.summary && (
          <p className="text-sm text-gray-600 mt-2 line-clamp-2">
            {document.summary}
          </p>
        )}
      </motion.div>
    )
  }

  return (
    <div className="h-full flex flex-col bg-gray-50">
      {/* Hidden file input */}
      <input
        ref={fileInputRef}
        type="file"
        multiple
        accept=".pdf,.docx,.doc,.txt,.md"
        onChange={handleFileInputChange}
        className="hidden"
      />

      {/* Header */}
      <div className="bg-white border-b border-gray-200 p-6">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-gradient-to-br from-purple-600 to-blue-600 rounded-lg flex items-center justify-center">
              <DocumentTextIcon className="w-5 h-5 text-white" />
            </div>
            <div>
              <h1 className="text-xl font-semibold text-gray-900">Document Library</h1>
              <p className="text-sm text-gray-600">
                {documents.length} documents • {uploadingFiles.length} uploading
              </p>
            </div>
          </div>

          {/* Connection Status */}
          <div className={`flex items-center gap-1 px-3 py-1 rounded-full text-xs font-medium ${
            connectionStatus === 'connected' 
              ? 'bg-green-100 text-green-700' 
              : connectionStatus === 'disconnected'
              ? 'bg-red-100 text-red-700'
              : 'bg-yellow-100 text-yellow-700'
          }`}>
            <div className={`w-2 h-2 rounded-full ${
              connectionStatus === 'connected' 
                ? 'bg-green-500' 
                : connectionStatus === 'disconnected'
                ? 'bg-red-500'
                : 'bg-yellow-500'
            }`} />
            {connectionStatus === 'connected' ? 'Connected' : 
             connectionStatus === 'disconnected' ? 'Disconnected' : 'Checking...'}
          </div>
        </div>

        {/* Project Selection */}
        <div className="flex items-center gap-4 mb-4">
          <div className="flex items-center gap-2">
            <FolderIcon className="w-4 h-4 text-gray-600" />
            <select
              value={selectedProjectId || ''}
              onChange={(e) => setSelectedProjectId(e.target.value || null)}
              className="px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent bg-white text-sm"
            >
              <option value="">All Projects</option>
              {projects.map((project) => (
                <option key={project.id} value={project.id}>
                  {project.name} ({project.document_count} docs)
                </option>
              ))}
            </select>
          </div>

          <Button onClick={loadDocuments} variant="outline" size="sm">
            <ArrowPathIcon className="w-4 h-4 mr-2" />
            Refresh
          </Button>
        </div>

        {/* Search and Filters */}
        <div className="flex items-center gap-4">
          <div className="flex-1 max-w-md relative">
            <MagnifyingGlassIcon className="w-4 h-4 absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" />
            <Input
              placeholder="Search documents..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="pl-9"
            />
          </div>

          {/* Sort Dropdown */}
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="outline" size="sm">
                <FunnelIcon className="w-4 h-4 mr-2" />
                {sortBy === 'name' ? 'Name' : 
                 sortBy === 'date' ? 'Date' : 
                 sortBy === 'size' ? 'Size' : 'Status'}
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent>
              <DropdownMenuItem onClick={() => setSortBy('name')}>Name</DropdownMenuItem>
              <DropdownMenuItem onClick={() => setSortBy('date')}>Date</DropdownMenuItem>
              <DropdownMenuItem onClick={() => setSortBy('size')}>Size</DropdownMenuItem>
              <DropdownMenuItem onClick={() => setSortBy('status')}>Status</DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>

          {/* Filter Dropdown */}
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="outline" size="sm">
                <FunnelIcon className="w-4 h-4 mr-2" />
                {filterStatus === 'all' ? 'All Status' : filterStatus}
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent>
              <DropdownMenuItem onClick={() => setFilterStatus('all')}>All Status</DropdownMenuItem>
              <DropdownMenuItem onClick={() => setFilterStatus('completed')}>Completed</DropdownMenuItem>
              <DropdownMenuItem onClick={() => setFilterStatus('processing')}>Processing</DropdownMenuItem>
              <DropdownMenuItem onClick={() => setFilterStatus('failed')}>Failed</DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        </div>
      </div>

      {/* Upload Progress */}
      <AnimatePresence>
        {uploadingFiles.length > 0 && (
          <div className="p-4 bg-gray-50 border-b border-gray-200">
            <div className="space-y-3">
              {uploadingFiles.map(uploadingFile => (
                <UploadProgress key={uploadingFile.id} uploadingFile={uploadingFile} />
              ))}
            </div>
          </div>
        )}
      </AnimatePresence>

      {/* Main Content */}
      <div className="flex-1 overflow-y-auto p-6">
        {/* Drop Zone */}
        <div
          ref={dropZoneRef}
          onDragOver={handleDragOver}
          onDrop={handleDrop}
          className={cn(
            "border-2 border-dashed border-gray-300 rounded-lg p-8 text-center mb-6 transition-colors",
            "hover:border-gray-400 hover:bg-gray-50",
            !selectedProjectId && "opacity-50 pointer-events-none"
          )}
        >
          <CloudArrowUpIcon className="w-12 h-12 text-gray-400 mx-auto mb-4" />
          <h3 className="text-lg font-medium text-gray-900 mb-2">
            {selectedProjectId ? 'Drop files here or click to upload' : 'Select a project to upload documents'}
          </h3>
          <p className="text-gray-600 mb-4">
            Supports PDF, DOCX, TXT, MD files up to 50MB
          </p>
          {selectedProjectId && connectionStatus === 'connected' && (
            <Button 
              variant="outline" 
              onClick={() => fileInputRef.current?.click()}
            >
              <PlusIcon className="w-4 h-4 mr-2" />
              Choose Files
            </Button>
          )}
          {connectionStatus === 'disconnected' && (
            <p className="text-red-600 text-sm">Backend disconnected - please check connection</p>
          )}
        </div>

        {/* Documents Grid */}
        {isLoading ? (
          <div className="flex items-center justify-center py-12">
            <div className="text-center">
              <ArrowPathIcon className="w-8 h-8 text-gray-400 animate-spin mx-auto mb-4" />
              <p className="text-gray-600">Loading documents...</p>
            </div>
          </div>
        ) : filteredDocuments.length === 0 ? (
          <div className="text-center py-12">
            <DocumentTextIcon className="w-16 h-16 text-gray-300 mx-auto mb-4" />
            <h3 className="text-lg font-medium text-gray-900 mb-2">
              {searchQuery ? 'No documents found' : 'No documents yet'}
            </h3>
            <p className="text-gray-600 max-w-md mx-auto">
              {searchQuery 
                ? `No documents match "${searchQuery}". Try adjusting your search terms.`
                : selectedProjectId 
                  ? 'Upload your first document to get started with this project.'
                  : 'Upload documents to projects to see them here.'
              }
            </p>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            <AnimatePresence>
              {filteredDocuments.map((document) => (
                <DocumentCard key={document.id} document={document} />
              ))}
            </AnimatePresence>
          </div>
        )}
      </div>

      {/* Document Details Dialog */}
      <Dialog open={!!selectedDocument} onOpenChange={() => setSelectedDocument(null)}>
        <DialogContent className="max-w-2xl">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <DocumentTextIcon className="w-5 h-5" />
              Document Details
            </DialogTitle>
          </DialogHeader>
          
          {selectedDocument && (
            <div className="space-y-4">
              <div className="grid grid-cols-2 gap-4 text-sm">
                <div>
                  <span className="font-medium text-gray-600">Filename:</span>
                  <p className="text-gray-900">{selectedDocument.filename}</p>
                </div>
                <div>
                  <span className="font-medium text-gray-600">File Size:</span>
                  <p className="text-gray-900">{formatFileSize(selectedDocument.file_size)}</p>
                </div>
                <div>
                  <span className="font-medium text-gray-600">Type:</span>
                  <p className="text-gray-900">{selectedDocument.file_type}</p>
                </div>
                <div>
                  <span className="font-medium text-gray-600">Status:</span>
                  <Badge className={cn("capitalize", 
                    selectedDocument.processing_status === 'completed' ? 'bg-green-100 text-green-800' :
                    selectedDocument.processing_status === 'processing' ? 'bg-yellow-100 text-yellow-800' :
                    'bg-red-100 text-red-800'
                  )}>
                    {selectedDocument.processing_status}
                  </Badge>
                </div>
                <div className="col-span-2">
                  <span className="font-medium text-gray-600">Uploaded:</span>
                  <p className="text-gray-900">{formatDate(selectedDocument.uploaded_at || selectedDocument.created_at || '1970-01-01')}</p>
                </div>
              </div>

              {selectedDocument.tags.length > 0 && (
                <div>
                  <span className="font-medium text-gray-600 block mb-2">Tags:</span>
                  <div className="flex flex-wrap gap-2">
                    {selectedDocument.tags.map((tag, index) => (
                      <Badge key={index} variant="secondary">{tag}</Badge>
                    ))}
                  </div>
                </div>
              )}

              {selectedDocument.summary && (
                <div>
                  <span className="font-medium text-gray-600 block mb-2">Summary:</span>
                  <p className="text-gray-900 text-sm">{selectedDocument.summary}</p>
                </div>
              )}
            </div>
          )}

          <DialogFooter>
            <Button variant="outline" onClick={() => setSelectedDocument(null)}>
              Close
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  )
}

export default DocumentLibrary