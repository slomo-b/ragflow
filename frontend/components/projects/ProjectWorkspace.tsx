// frontend/components/projects/ProjectWorkspace.tsx - An neue API angepasst
'use client'

import React, { useState, useCallback, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  PlusIcon,
  FolderIcon,
  DocumentTextIcon,
  ChatBubbleLeftRightIcon,
  EllipsisVerticalIcon,
  PencilIcon,
  TrashIcon,
  CalendarIcon,
  SparklesIcon,
  ArrowRightIcon,
  ExclamationTriangleIcon
} from '@heroicons/react/24/outline'
import {
  FolderIcon as FolderSolidIcon,
  CheckCircleIcon as CheckCircleSolidIcon
} from '@heroicons/react/24/solid'
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/Card"
import { Button } from "@/components/ui/Button"
import { Input } from "@/components/ui/Input"
import { Textarea } from "@/components/ui/textarea"
import { Badge } from "@/components/ui/Badge"
import { 
  DropdownMenu, 
  DropdownMenuContent, 
  DropdownMenuItem, 
  DropdownMenuTrigger,
  DropdownMenuSeparator
} from "@/components/ui/dropdown-menu"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import { Label } from "@/components/ui/label"
import { cn, formatDate } from '@/lib/utils'
import ApiService from '@/services/api'
import toast from 'react-hot-toast'

// Types
interface Project {
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

interface ProjectWithStats extends Project {
  document_count: number
  chat_count?: number
  last_activity?: string
}

interface ProjectFormData {
  name: string
  description: string
}

interface SystemInfo {
  app: {
    name: string
    version: string
    python_version: string
  }
  features: Record<string, boolean>
  stats: {
    projects: number
    documents: number
    chats: number
    rag_documents: number
  }
  settings: {
    chunk_size: number
    chunk_overlap: number
    top_k: number
  }
}

export const ProjectWorkspace: React.FC = () => {
  // State Management
  const [projects, setProjects] = useState<ProjectWithStats[]>([])
  const [systemInfo, setSystemInfo] = useState<SystemInfo | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [connectionStatus, setConnectionStatus] = useState<'checking' | 'connected' | 'disconnected'>('checking')
  const [showCreateDialog, setShowCreateDialog] = useState(false)
  const [showEditDialog, setShowEditDialog] = useState(false)
  const [showDeleteDialog, setShowDeleteDialog] = useState(false)
  const [selectedProject, setSelectedProject] = useState<ProjectWithStats | null>(null)
  const [formData, setFormData] = useState<ProjectFormData>({ name: '', description: '' })
  const [isSubmitting, setIsSubmitting] = useState(false)

  // Load data on mount
  useEffect(() => {
    initializeWorkspace()
  }, [])

  const initializeWorkspace = async () => {
    console.log('🚀 Initializing project workspace...')
    setConnectionStatus('checking')
    
    try {
      // Test connection first
      const healthCheck = await ApiService.healthCheck()
      
      if (healthCheck.status === 'healthy') {
        setConnectionStatus('connected')
        console.log('✅ Backend connection established')
        
        // Load data
        await Promise.all([
          loadProjects(),
          loadSystemInfo()
        ])
        
        toast.success('Project workspace ready!', { duration: 2000 })
      } else {
        throw new Error('Backend unhealthy')
      }
    } catch (error) {
      setConnectionStatus('disconnected')
      console.error('💥 Workspace initialization failed:', error)
      toast.error('Failed to connect to backend. Please check the server.')
    }
  }

  const loadProjects = async () => {
    setIsLoading(true)
    try {
      const response = await ApiService.getProjects()
      const projectsData = response.projects || []
      
      // Transform projects with stats
      const projectsWithStats: ProjectWithStats[] = projectsData.map(project => ({
        ...project,
        document_count: project.document_count || 0,
        chat_count: project.chat_count || 0,
        last_activity: project.updated_at || project.created_at
      }))
      
      setProjects(projectsWithStats)
      console.log(`📁 Loaded ${projectsData.length} projects`)
    } catch (error) {
      console.error('Failed to load projects:', error)
      toast.error('Failed to load projects')
    } finally {
      setIsLoading(false)
    }
  }

  const loadSystemInfo = async () => {
    try {
      const info = await ApiService.getSystemInfo()
      setSystemInfo(info)
      console.log('📊 System info loaded')
    } catch (error) {
      console.error('Failed to load system info:', error)
      // Don't show error toast for system info as it's not critical
    }
  }

  // Project creation
  const createProject = useCallback(async () => {
    if (!formData.name.trim()) {
      toast.error('Project name is required')
      return
    }

    setIsSubmitting(true)
    try {
      const newProject = await ApiService.createProject({
        name: formData.name.trim(),
        description: formData.description.trim()
      })

      // Add to projects list with stats
      const projectWithStats: ProjectWithStats = {
        ...newProject,
        document_count: 0,
        chat_count: 0,
        last_activity: newProject.created_at
      }

      setProjects(prev => [projectWithStats, ...prev])
      setShowCreateDialog(false)
      setFormData({ name: '', description: '' })
      toast.success('Project created successfully')
    } catch (error) {
      console.error('Create project error:', error)
      toast.error('Failed to create project')
    } finally {
      setIsSubmitting(false)
    }
  }, [formData])

  // Project update
  const updateProject = useCallback(async () => {
    if (!selectedProject || !formData.name.trim()) {
      toast.error('Project name is required')
      return
    }

    setIsSubmitting(true)
    try {
      const updatedProject = await ApiService.updateProject(selectedProject.id, {
        name: formData.name.trim(),
        description: formData.description.trim()
      })

      // Update in projects list
      setProjects(prev => prev.map(p => 
        p.id === selectedProject.id 
          ? { ...p, ...updatedProject, last_activity: updatedProject.updated_at || p.last_activity }
          : p
      ))

      setShowEditDialog(false)
      setSelectedProject(null)
      toast.success('Project updated successfully')
    } catch (error) {
      console.error('Update project error:', error)
      toast.error('Failed to update project')
    } finally {
      setIsSubmitting(false)
    }
  }, [selectedProject, formData])

  // Project deletion
  const deleteProject = useCallback(async () => {
    if (!selectedProject) return

    setIsSubmitting(true)
    try {
      await ApiService.deleteProject(selectedProject.id)
      
      setProjects(prev => prev.filter(p => p.id !== selectedProject.id))
      setShowDeleteDialog(false)
      setSelectedProject(null)
      toast.success('Project deleted successfully')
    } catch (error) {
      console.error('Delete project error:', error)
      toast.error('Failed to delete project')
    } finally {
      setIsSubmitting(false)
    }
  }, [selectedProject])

  // Dialog handlers
  const handleCreateClick = () => {
    setFormData({ name: '', description: '' })
    setShowCreateDialog(true)
  }

  const handleEditClick = (project: ProjectWithStats) => {
    setSelectedProject(project)
    setFormData({ name: project.name, description: project.description || '' })
    setShowEditDialog(true)
  }

  const handleDeleteClick = (project: ProjectWithStats) => {
    setSelectedProject(project)
    setShowDeleteDialog(true)
  }

  // System Stats Component
  const SystemStats: React.FC = () => {
    if (!systemInfo) return null

    const stats = [
      {
        label: 'Total Projects',
        value: systemInfo.stats.projects,
        icon: FolderSolidIcon,
        color: 'text-blue-600 bg-blue-100'
      },
      {
        label: 'Documents',
        value: systemInfo.stats.documents,
        icon: DocumentTextIcon,
        color: 'text-green-600 bg-green-100'
      },
      {
        label: 'RAG Documents',
        value: systemInfo.stats.rag_documents,
        icon: SparklesIcon,
        color: 'text-purple-600 bg-purple-100'
      },
      {
        label: 'Total Chats',
        value: systemInfo.stats.chats,
        icon: ChatBubbleLeftRightIcon,
        color: 'text-orange-600 bg-orange-100'
      }
    ]

    return (
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
        {stats.map((stat, index) => (
          <motion.div
            key={stat.label}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: index * 0.1 }}
            className="bg-white p-4 rounded-lg border border-gray-200 hover:shadow-md transition-shadow"
          >
            <div className="flex items-center gap-3">
              <div className={cn("w-10 h-10 rounded-lg flex items-center justify-center", stat.color)}>
                <stat.icon className="w-5 h-5" />
              </div>
              <div>
                <div className="text-2xl font-bold text-gray-900">{stat.value}</div>
                <div className="text-sm text-gray-600">{stat.label}</div>
              </div>
            </div>
          </motion.div>
        ))}
      </div>
    )
  }

  // Project Card Component
  const ProjectCard: React.FC<{ project: ProjectWithStats }> = ({ project }) => (
    <motion.div
      layout
      initial={{ opacity: 0, scale: 0.9 }}
      animate={{ opacity: 1, scale: 1 }}
      exit={{ opacity: 0, scale: 0.9 }}
      whileHover={{ y: -4 }}
      className="bg-white border border-gray-200 rounded-lg p-6 hover:shadow-lg transition-all duration-200"
    >
      <div className="flex items-start justify-between mb-4">
        <div className="flex items-center gap-3">
          <div className="w-12 h-12 bg-gradient-to-br from-blue-500 to-purple-600 rounded-lg flex items-center justify-center flex-shrink-0">
            <FolderSolidIcon className="w-6 h-6 text-white" />
          </div>
          <div className="min-w-0 flex-1">
            <h3 className="font-semibold text-gray-900 truncate" title={project.name}>
              {project.name}
            </h3>
            <p className="text-sm text-gray-600 line-clamp-2 mt-1">
              {project.description || 'No description provided'}
            </p>
          </div>
        </div>
        
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button variant="ghost" size="sm" className="h-8 w-8 p-0">
              <EllipsisVerticalIcon className="h-4 w-4" />
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end">
            <DropdownMenuItem onClick={() => handleEditClick(project)}>
              <PencilIcon className="w-4 h-4 mr-2" />
              Edit Project
            </DropdownMenuItem>
            <DropdownMenuSeparator />
            <DropdownMenuItem 
              onClick={() => handleDeleteClick(project)}
              className="text-red-600"
            >
              <TrashIcon className="w-4 h-4 mr-2" />
              Delete Project
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      </div>

      {/* Project Stats */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-1 text-sm text-gray-600">
            <DocumentTextIcon className="w-4 h-4" />
            <span>{project.document_count} docs</span>
          </div>
          <div className="flex items-center gap-1 text-sm text-gray-600">
            <ChatBubbleLeftRightIcon className="w-4 h-4" />
            <span>{project.chat_count || 0} chats</span>
          </div>
        </div>
        
        <Badge variant="secondary" className="text-xs">
          {project.status || 'active'}
        </Badge>
      </div>

      {/* Last Activity */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-1 text-xs text-gray-500">
          <CalendarIcon className="w-3 h-3" />
          <span>Updated {formatDate(project.last_activity || project.created_at)}</span>
        </div>
        
        <Button 
          variant="ghost" 
          size="sm"
          onClick={() => {
            // TODO: Navigate to project details or chat
            toast.info('Project navigation coming soon')
          }}
          className="text-blue-600 hover:text-blue-700 hover:bg-blue-50 px-3 py-1 h-7"
        >
          <ArrowRightIcon className="w-3 h-3 mr-1" />
          Open
        </Button>
      </div>
    </motion.div>
  )

  return (
    <div className="h-full flex flex-col bg-gray-50">
      {/* Header */}
      <div className="bg-white border-b border-gray-200 p-6">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-gradient-to-br from-green-600 to-blue-600 rounded-lg flex items-center justify-center">
              <FolderSolidIcon className="w-5 h-5 text-white" />
            </div>
            <div>
              <h1 className="text-xl font-semibold text-gray-900">Project Workspace</h1>
              <p className="text-sm text-gray-600">
                Manage your RAG projects and documents
              </p>
            </div>
          </div>

          <div className="flex items-center gap-3">
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

            <Button onClick={handleCreateClick} disabled={connectionStatus !== 'connected'}>
              <PlusIcon className="w-4 h-4 mr-2" />
              New Project
            </Button>
          </div>
        </div>

        {/* System Stats */}
        <SystemStats />
      </div>

      {/* Main Content */}
      <div className="flex-1 overflow-y-auto p-6">
        {isLoading ? (
          <div className="flex items-center justify-center py-12">
            <div className="text-center">
              <motion.div
                animate={{ rotate: 360 }}
                transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                className="w-8 h-8 border-2 border-blue-500 border-t-transparent rounded-full mx-auto mb-4"
              />
              <p className="text-gray-600">Loading projects...</p>
            </div>
          </div>
        ) : connectionStatus === 'disconnected' ? (
          <div className="text-center py-12">
            <ExclamationTriangleIcon className="w-16 h-16 text-red-300 mx-auto mb-4" />
            <h3 className="text-lg font-medium text-gray-900 mb-2">Backend Disconnected</h3>
            <p className="text-gray-600 max-w-md mx-auto mb-4">
              Unable to connect to the backend server. Please check if the server is running.
            </p>
            <Button onClick={initializeWorkspace} variant="outline">
              Retry Connection
            </Button>
          </div>
        ) : projects.length === 0 ? (
          <div className="text-center py-12">
            <FolderIcon className="w-16 h-16 text-gray-300 mx-auto mb-4" />
            <h3 className="text-lg font-medium text-gray-900 mb-2">No projects yet</h3>
            <p className="text-gray-600 max-w-md mx-auto mb-6">
              Create your first project to start organizing your documents and building RAG applications.
            </p>
            <Button onClick={handleCreateClick} size="lg">
              <PlusIcon className="w-5 h-5 mr-2" />
              Create Your First Project
            </Button>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            <AnimatePresence>
              {projects.map((project) => (
                <ProjectCard key={project.id} project={project} />
              ))}
            </AnimatePresence>
          </div>
        )}
      </div>

      {/* Create Project Dialog */}
      <Dialog open={showCreateDialog} onOpenChange={setShowCreateDialog}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <PlusIcon className="w-5 h-5" />
              Create New Project
            </DialogTitle>
            <DialogDescription>
              Create a new project to organize your documents and build RAG applications.
            </DialogDescription>
          </DialogHeader>
          
          <div className="space-y-4">
            <div>
              <Label htmlFor="create-name">Project Name *</Label>
              <Input
                id="create-name"
                value={formData.name}
                onChange={(e) => setFormData(prev => ({ ...prev, name: e.target.value }))}
                placeholder="Enter project name..."
                className="mt-1"
              />
            </div>
            
            <div>
              <Label htmlFor="create-description">Description</Label>
              <Textarea
                id="create-description"
                value={formData.description}
                onChange={(e) => setFormData(prev => ({ ...prev, description: e.target.value }))}
                placeholder="Describe your project..."
                className="mt-1"
                rows={3}
              />
            </div>
          </div>

          <DialogFooter>
            <Button 
              variant="outline" 
              onClick={() => setShowCreateDialog(false)}
              disabled={isSubmitting}
            >
              Cancel
            </Button>
            <Button 
              onClick={createProject}
              disabled={!formData.name.trim() || isSubmitting}
            >
              {isSubmitting ? (
                <motion.div
                  animate={{ rotate: 360 }}
                  transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                  className="w-4 h-4 border-2 border-white border-t-transparent rounded-full mr-2"
                />
              ) : (
                <PlusIcon className="w-4 h-4 mr-2" />
              )}
              Create Project
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Edit Project Dialog */}
      <Dialog open={showEditDialog} onOpenChange={setShowEditDialog}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <PencilIcon className="w-5 h-5" />
              Edit Project
            </DialogTitle>
            <DialogDescription>
              Update your project information.
            </DialogDescription>
          </DialogHeader>
          
          <div className="space-y-4">
            <div>
              <Label htmlFor="edit-name">Project Name *</Label>
              <Input
                id="edit-name"
                value={formData.name}
                onChange={(e) => setFormData(prev => ({ ...prev, name: e.target.value }))}
                placeholder="Enter project name..."
                className="mt-1"
              />
            </div>
            
            <div>
              <Label htmlFor="edit-description">Description</Label>
              <Textarea
                id="edit-description"
                value={formData.description}
                onChange={(e) => setFormData(prev => ({ ...prev, description: e.target.value }))}
                placeholder="Describe your project..."
                className="mt-1"
                rows={3}
              />
            </div>
          </div>

          <DialogFooter>
            <Button 
              variant="outline" 
              onClick={() => setShowEditDialog(false)}
              disabled={isSubmitting}
            >
              Cancel
            </Button>
            <Button 
              onClick={updateProject}
              disabled={!formData.name.trim() || isSubmitting}
            >
              {isSubmitting ? (
                <motion.div
                  animate={{ rotate: 360 }}
                  transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                  className="w-4 h-4 border-2 border-white border-t-transparent rounded-full mr-2"
                />
              ) : (
                <PencilIcon className="w-4 h-4 mr-2" />
              )}
              Update Project
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Delete Confirmation Dialog */}
      <Dialog open={showDeleteDialog} onOpenChange={setShowDeleteDialog}>
        <DialogContent className="sm:max-w-md">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <ExclamationTriangleIcon className="w-5 h-5 text-red-500" />
              Delete Project
            </DialogTitle>
            <DialogDescription>
              Are you sure you want to delete "{selectedProject?.name}"? This action cannot be undone 
              and will remove all associated documents and conversations.
            </DialogDescription>
          </DialogHeader>

          <DialogFooter>
            <Button 
              variant="outline" 
              onClick={() => setShowDeleteDialog(false)}
              disabled={isSubmitting}
            >
              Cancel
            </Button>
            <Button 
              variant="destructive"
              onClick={deleteProject}
              disabled={isSubmitting}
            >
              {isSubmitting ? (
                <motion.div
                  animate={{ rotate: 360 }}
                  transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                  className="w-4 h-4 border-2 border-white border-t-transparent rounded-full mr-2"
                />
              ) : (
                <TrashIcon className="w-4 h-4 mr-2" />
              )}
              Delete Project
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  )
}

export default ProjectWorkspace