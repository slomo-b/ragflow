// frontend/components/documents/DocumentLibrary.tsx
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
import { cn } from '@/lib/utils'
import { DocumentAPI, ProjectAPI, handleAPIError, withErrorHandling, Document, Project } from '@/lib/api'
import toast from 'react-hot-toast'

// Enhanced Document Interface
interface ExtendedDocument extends Document {
  uploading?: boolean
  uploadProgress?: number
  extracted_length?: number
}

interface UploadingFile {
  id: string
  file: File
  progress: number
  status: 'uploading' | 'processing' | 'completed' | 'error'
  error?: string
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
  const [showUploadDialog, setShowUploadDialog] = useState(false)

  // Refs
  const fileInputRef = useRef<HTMLInputElement>(null)
  const dropZoneRef = useRef<HTMLDivElement>(null)

  // Load data on mount
  useEffect(() => {
    loadProjects()
    loadDocuments()
  }, [])

  // Load documents when project selection changes
  useEffect(() => {
    loadDocuments()
  }, [selectedProjectId])

  const loadProjects = async () => {
    const result = await withErrorHandling(async () => {
      return await ProjectAPI.getProjects()
    })
    
    if (result) {
      setProjects(result)
    }
  }

  const loadDocuments = async () => {
    setIsLoading(true)
    const result = await withErrorHandling(async () => {
      return await DocumentAPI.getDocuments(selectedProjectId || undefined)
    })
    
    if (result) {
      setDocuments(result)
    }
    setIsLoading(false)
  }

  // File upload handling
  const handleFileUpload = useCallback(async (files: FileList | File[]) => {
    if (!selectedProjectId) {
      toast.error('Please select a project first')
      return
    }

    const fileArray = Array.from(files)
    const newUploadingFiles: UploadingFile[] = fileArray.map(file => ({
      id: `upload-${Date.now()}-${Math.random()}`,
      file,
      progress: 0,
      status: 'uploading'
    }))

    setUploadingFiles(prev => [...prev, ...newUploadingFiles])

    // Upload files one by one
    for (const uploadingFile of newUploadingFiles) {
      try {
        // Update status to uploading
        setUploadingFiles(prev => prev.map(f => 
          f.id === uploadingFile.id 
            ? { ...f, status: 'uploading' }
            : f
        ))

        const result = await DocumentAPI.uploadDocument(
          uploadingFile.file,
          selectedProjectId,
          (progress) => {
            setUploadingFiles(prev => prev.map(f => 
              f.id === uploadingFile.id 
                ? { ...f, progress }
                : f
            ))
          }
        )

        // Update status to processing
        setUploadingFiles(prev => prev.map(f => 
          f.id === uploadingFile.id 
            ? { ...f, status: 'processing', progress: 100 }
            : f
        ))

        // Wait a bit then mark as completed
        setTimeout(() => {
          setUploadingFiles(prev => prev.map(f => 
            f.id === uploadingFile.id 
              ? { ...f, status: 'completed' }
              : f
          ))

          // Remove from uploading files after delay
          setTimeout(() => {
            setUploadingFiles(prev => prev.filter(f => f.id !== uploadingFile.id))
          }, 2000)
        }, 1000)

        toast.success(`${uploadingFile.file.name} uploaded successfully`)

      } catch (error) {
        console.error('Upload error:', error)
        
        setUploadingFiles(prev => prev.map(f => 
          f.id === uploadingFile.id 
            ? { ...f, status: 'error', error: handleAPIError(error) }
            : f
        ))

        toast.error(`Failed to upload ${uploadingFile.file.name}`)
      }
    }

    // Reload documents
    setTimeout(() => {
      loadDocuments()
    }, 1000)
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
    const result = await withErrorHandling(async () => {
      await DocumentAPI.deleteDocument(documentId)
    })

    if (result !== null) {
      setDocuments(prev => prev.filter(doc => doc.id !== documentId))
      toast.success('Document deleted successfully')
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
          return new Date(b.created_at).getTime() - new Date(a.created_at).getTime()
        case 'size':
          return b.file_size - a.file_size
        case 'status':
          return a.processing_status.localeCompare(b.processing_status)
        default:
          return 0
      }
    })

  // Format file size
  const formatFileSize = (bytes: number): string => {
    if (bytes === 0) return '0 Bytes'
    const k = 1024
    const sizes = ['Bytes', 'KB', 'MB', 'GB']
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
  }

  // Get status icon and color
  const getStatusDisplay = (status: string) => {
    switch (status) {
      case 'completed':
        return {
          icon: <CheckCircleSolidIcon className="w-4 h-4" />,
          color: 'text-green-600',
          bgColor: 'bg-green-100',
          label: 'Completed'
        }
      case 'processing':
        return {
          icon: <ClockSolidIcon className="w-4 h-4" />,
          color: 'text-yellow-600',
          bgColor: 'bg-yellow-100',
          label: 'Processing'
        }
      case 'failed':
        return {
          icon: <ExclamationTriangleSolidIcon className="w-4 h-4" />,
          color: 'text-red-600',
          bgColor: 'bg-red-100',
          label: 'Failed'
        }
      default:
        return {
          icon: <ClockIcon className="w-4 h-4" />,
          color: 'text-gray-600',
          bgColor: 'bg-gray-100',
          label: 'Unknown'
        }
    }
  }

  // Document Card Component
  const DocumentCard: React.FC<{ document: ExtendedDocument; index: number }> = ({ document, index }) => {
    const statusDisplay = getStatusDisplay(document.processing_status)

    return (
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: index * 0.05 }}
        className="group"
      >
        <Card className="hover:shadow-md transition-all duration-200 border border-gray-200">
          <CardContent className="p-4">
            <div className="flex items-start justify-between">
              {/* Document Info */}
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2 mb-2">
                  <DocumentTextSolidIcon className="w-5 h-5 text-blue-500 flex-shrink-0" />
                  <h3 className="font-medium text-gray-900 truncate">
                    {document.filename}
                  </h3>
                </div>
                
                <div className="space-y-1 text-sm text-gray-600">
                  <div className="flex items-center gap-4">
                    <span>{formatFileSize(document.file_size)}</span>
                    <span>{document.file_type}</span>
                  </div>
                  
                  <div className="flex items-center gap-2">
                    <CalendarIcon className="w-3 h-3" />
                    <span>{new Date(document.created_at).toLocaleDateString()}</span>
                  </div>

                  {document.extracted_length && (
                    <div className="flex items-center gap-2">
                      <SparklesIcon className="w-3 h-3" />
                      <span>{document.extracted_length.toLocaleString()} characters extracted</span>
                    </div>
                  )}
                </div>
              </div>

              {/* Actions */}
              <div className="flex items-center gap-2 ml-4">
                {/* Status Badge */}
                <Badge 
                  variant="outline" 
                  className={cn(statusDisplay.bgColor, statusDisplay.color, "border-0")}
                >
                  {statusDisplay.icon}
                  <span className="ml-1">{statusDisplay.label}</span>
                </Badge>

                {/* Actions Menu */}
                <DropdownMenu>
                  <DropdownMenuTrigger asChild>
                    <Button variant="ghost" size="sm" className="opacity-0 group-hover:opacity-100 transition-opacity">
                      <ArrowDownTrayIcon className="w-4 h-4" />
                    </Button>
                  </DropdownMenuTrigger>
                  <DropdownMenuContent align="end">
                    <DropdownMenuItem onClick={() => setSelectedDocument(document)}>
                      <EyeIcon className="w-4 h-4 mr-2" />
                      View Details
                    </DropdownMenuItem>
                    <DropdownMenuSeparator />
                    <DropdownMenuItem 
                      className="text-red-600"
                      onClick={() => deleteDocument(document.id)}
                    >
                      <TrashIcon className="w-4 h-4 mr-2" />
                      Delete
                    </DropdownMenuItem>
                  </DropdownMenuContent>
                </DropdownMenu>
              </div>
            </div>
          </CardContent>
        </Card>
      </motion.div>
    )
  }

  // Upload Progress Component
  const UploadProgress: React.FC<{ uploadingFile: UploadingFile }> = ({ uploadingFile }) => {
    const getStatusColor = () => {
      switch (uploadingFile.status) {
        case 'uploading': return 'bg-blue-500'
        case 'processing': return 'bg-yellow-500'
        case 'completed': return 'bg-green-500'
        case 'error': return 'bg-red-500'
        default: return 'bg-gray-500'
      }
    }

    return (
      <motion.div
        initial={{ opacity: 0, height: 0 }}
        animate={{ opacity: 1, height: 'auto' }}
        exit={{ opacity: 0, height: 0 }}
        className="bg-white border border-gray-200 rounded-lg p-4 mb-4"
      >
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center gap-2">
            <DocumentTextIcon className="w-4 h-4 text-gray-500" />
            <span className="text-sm font-medium text-gray-900">
              {uploadingFile.file.name}
            </span>
          </div>
          
          <Badge variant="outline" className={cn("text-white border-0", getStatusColor())}>
            {uploadingFile.status === 'uploading' && 'Uploading'}
            {uploadingFile.status === 'processing' && 'Processing'}
            {uploadingFile.status === 'completed' && 'Completed'}
            {uploadingFile.status === 'error' && 'Error'}
          </Badge>
        </div>

        {uploadingFile.status !== 'error' && (
          <Progress 
            value={uploadingFile.progress} 
            className="h-2"
          />
        )}

        {uploadingFile.error && (
          <div className="mt-2 text-sm text-red-600">
            {uploadingFile.error}
          </div>
        )}
      </motion.div>
    )
  }

  return (
    <div className="h-full flex flex-col bg-gray-50">
      {/* Header */}
      <div className="bg-white border-b border-gray-200 p-6">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-gradient-to-br from-purple-500 to-pink-600 rounded-lg flex items-center justify-center">
              <FolderIcon className="w-5 h-5 text-white" />
            </div>
            <div>
              <h1 className="text-xl font-semibold text-gray-900">Document Library</h1>
              <p className="text-sm text-gray-600">
                {documents.length} documents
                {selectedProjectId && ` in ${projects.find(p => p.id === selectedProjectId)?.name || 'Selected Project'}`}
              </p>
            </div>
          </div>

          <div className="flex items-center gap-3">
            {/* Project Selector */}
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button variant="outline">
                  <FolderIcon className="w-4 h-4 mr-2" />
                  {selectedProjectId 
                    ? projects.find(p => p.id === selectedProjectId)?.name || 'Project'
                    : 'All Projects'
                  }
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent>
                <DropdownMenuItem onClick={() => setSelectedProjectId(null)}>
                  All Projects
                </DropdownMenuItem>
                <DropdownMenuSeparator />
                {projects.map(project => (
                  <DropdownMenuItem 
                    key={project.id}
                    onClick={() => setSelectedProjectId(project.id)}
                  >
                    {project.name}
                  </DropdownMenuItem>
                ))}
              </DropdownMenuContent>
            </DropdownMenu>

            {/* Upload Button */}
            <Button 
              onClick={() => fileInputRef.current?.click()}
              disabled={!selectedProjectId}
              className="bg-gradient-to-r from-purple-500 to-pink-600 hover:from-purple-600 hover:to-pink-700"
            >
              <CloudArrowUpIcon className="w-4 h-4 mr-2" />
              Upload Documents
            </Button>
          </div>
        </div>

        {/* Filters and Search */}
        <div className="flex items-center gap-4">
          {/* Search */}
          <div className="relative flex-1 max-w-md">
            <MagnifyingGlassIcon className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400" />
            <Input
              placeholder="Search documents..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="pl-10"
            />
          </div>

          {/* Sort */}
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="outline" size="sm">
                <FunnelIcon className="w-4 h-4 mr-2" />
                Sort by {sortBy}
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent>
              <DropdownMenuItem onClick={() => setSortBy('date')}>Date</DropdownMenuItem>
              <DropdownMenuItem onClick={() => setSortBy('name')}>Name</DropdownMenuItem>
              <DropdownMenuItem onClick={() => setSortBy('size')}>Size</DropdownMenuItem>
              <DropdownMenuItem onClick={() => setSortBy('status')}>Status</DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>

          {/* Filter */}
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="outline" size="sm">
                <TagIcon className="w-4 h-4 mr-2" />
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
            {uploadingFiles.map(uploadingFile => (
              <UploadProgress key={uploadingFile.id} uploadingFile={uploadingFile} />
            ))}
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
            Supports PDF, DOCX, TXT files up to 100MB
          </p>
          {selectedProjectId && (
            <Button 
              variant="outline" 
              onClick={() => fileInputRef.current?.click()}
            >
              <PlusIcon className="w-4 h-4 mr-2" />
              Choose Files
            </Button>
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
                ? 'Try adjusting your search query or filters'
                : 'Upload your first document to get started with AI-powered analysis'
              }
            </p>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            <AnimatePresence>
              {filteredDocuments.map((document, index) => (
                <DocumentCard key={document.id} document={document} index={index} />
              ))}
            </AnimatePresence>
          </div>
        )}
      </div>

      {/* Hidden File Input */}
      <input
        ref={fileInputRef}
        type="file"
        multiple
        accept=".pdf,.docx,.doc,.txt,.md"
        onChange={handleFileInputChange}
        className="hidden"
      />

      {/* Document Details Dialog */}
      <Dialog open={!!selectedDocument} onOpenChange={() => setSelectedDocument(null)}>
        <DialogContent className="max-w-2xl">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <DocumentTextIcon className="w-5 h-5" />
              {selectedDocument?.filename}
            </DialogTitle>
            <DialogDescription>
              Document details and processing information
            </DialogDescription>
          </DialogHeader>
          
          {selectedDocument && (
            <div className="space-y-4">
              <div className="grid grid-cols-2 gap-4 text-sm">
                <div>
                  <span className="font-medium text-gray-700">File Size:</span>
                  <span className="ml-2">{formatFileSize(selectedDocument.file_size)}</span>
                </div>
                <div>
                  <span className="font-medium text-gray-700">Type:</span>
                  <span className="ml-2">{selectedDocument.file_type}</span>
                </div>
                <div>
                  <span className="font-medium text-gray-700">Status:</span>
                  <span className="ml-2">{getStatusDisplay(selectedDocument.processing_status).label}</span>
                </div>
                <div>
                  <span className="font-medium text-gray-700">Upload Date:</span>
                  <span className="ml-2">{new Date(selectedDocument.created_at).toLocaleString()}</span>
                </div>
              </div>

              {selectedDocument.extracted_length && (
                <div className="bg-green-50 border border-green-200 rounded-lg p-4">
                  <div className="flex items-center gap-2 text-green-700 mb-2">
                    <CheckCircleIcon className="w-4 h-4" />
                    <span className="font-medium">Processing Complete</span>
                  </div>
                  <p className="text-sm text-green-600">
                    Successfully extracted {selectedDocument.extracted_length.toLocaleString()} characters 
                    and made available for AI chat.
                  </p>
                </div>
              )}

              {selectedDocument.project_ids.length > 0 && (
                <div>
                  <span className="font-medium text-gray-700 block mb-2">Projects:</span>
                  <div className="flex flex-wrap gap-2">
                    {selectedDocument.project_ids.map(projectId => {
                      const project = projects.find(p => p.id === projectId)
                      return (
                        <Badge key={projectId} variant="outline">
                          {project?.name || projectId}
                        </Badge>
                      )
                    })}
                  </div>
                </div>
              )}
            </div>
          )}

          <DialogFooter>
            <Button variant="outline" onClick={() => setSelectedDocument(null)}>
              Close
            </Button>
            {selectedDocument && (
              <Button 
                variant="destructive"
                onClick={() => {
                  deleteDocument(selectedDocument.id)
                  setSelectedDocument(null)
                }}
              >
                <TrashIcon className="w-4 h-4 mr-2" />
                Delete Document
              </Button>
            )}
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  )
}

export default DocumentLibrary