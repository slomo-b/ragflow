// frontend/components/projects/ProjectWorkspace.tsx
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
import { cn } from '@/lib/utils'
import { ProjectAPI, DocumentAPI, handleAPIError, withErrorHandling, Project, SystemInfo, HealthAPI } from '@/lib/api'
import toast from 'react-hot-toast'

interface ProjectWithStats extends Project {
  document_count: number
  chat_count?: number
  last_activity?: string
}

interface ProjectFormData {
  name: string
  description: string
}

export const ProjectWorkspace: React.FC = () => {
  // State Management
  const [projects, setProjects] = useState<ProjectWithStats[]>([])
  const [systemInfo, setSystemInfo] = useState<SystemInfo | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [showCreateDialog, setShowCreateDialog] = useState(false)
  const [showEditDialog, setShowEditDialog] = useState(false)
  const [showDeleteDialog, setShowDeleteDialog] = useState(false)
  const [selectedProject, setSelectedProject] = useState<ProjectWithStats | null>(null)
  const [formData, setFormData] = useState<ProjectFormData>({ name: '', description: '' })
  const [isSubmitting, setIsSubmitting] = useState(false)

  // Load data on mount
  useEffect(() => {
    loadProjects()
    loadSystemInfo()
  }, [])

  const loadProjects = async () => {
    setIsLoading(true)
    const result = await withErrorHandling(async () => {
      return await ProjectAPI.getProjects()
    })
    
    if (result) {
      setProjects(result)
    }
    setIsLoading(false)
  }

  const loadSystemInfo = async () => {
    const result = await withErrorHandling(async () => {
      return await HealthAPI.getSystemInfo()
    }, false) // Don't show error toast for system info
    
    if (result) {
      setSystemInfo(result)
    }
  }

  // Project creation
  const createProject = useCallback(async () => {
    if (!formData.name.trim()) {
      toast.error('Project name is required')
      return
    }

    setIsSubmitting(true)
    const result = await withErrorHandling(async () => {
      return await ProjectAPI.createProject({
        name: formData.name.trim(),
        description: formData.description.trim()
      })
    })

    if (result) {
      setProjects(prev => [result, ...prev])
      setShowCreateDialog(false)
      setFormData({ name: '', description: '' })
      toast.success('Project created successfully')
    }
    setIsSubmitting(false)
  }, [formData])

  // Project deletion
  const deleteProject = useCallback(async () => {
    if (!selectedProject) return

    setIsSubmitting(true)
    const result = await withErrorHandling(async () => {
      await ProjectAPI.deleteProject(selectedProject.id)
    })

    if (result !== null) {
      setProjects(prev => prev.filter(p => p.id !== selectedProject.id))
      setShowDeleteDialog(false)
      setSelectedProject(null)
      toast.success('Project deleted successfully')
    }
    setIsSubmitting(false)
  }, [selectedProject])

  // Dialog handlers
  const handleCreateClick = () => {
    setFormData({ name: '', description: '' })
    setShowCreateDialog(true)
  }

  const handleEditClick = (project: ProjectWithStats) => {
    setSelectedProject(project)
    setFormData({ name: project.name, description: project.description })
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
          >
            <Card>
              <CardContent className="p-4">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm font-medium text-gray-600">{stat.label}</p>
                    <p className="text-2xl font-bold text-gray-900">{stat.value}</p>
                  </div>
                  <div className={cn('w-10 h-10 rounded-lg flex items-center justify-center', stat.color)}>
                    <stat.icon className="w-5 h-5" />
                  </div>
                </div>
              </CardContent>
            </Card>
          </motion.div>
        ))}
      </div>
    )
  }

  // Project Card Component
  const ProjectCard: React.FC<{ project: ProjectWithStats; index: number }> = ({ project, index }) => {
    return (
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: index * 0.1 }}
        className="group"
      >
        <Card className="hover:shadow-lg transition-all duration-200 border border-gray-200 hover:border-gray-300">
          <CardHeader className="pb-3">
            <div className="flex items-start justify-between">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 bg-gradient-to-br from-blue-500 to-purple-600 rounded-lg flex items-center justify-center">
                  <FolderSolidIcon className="w-5 h-5 text-white" />
                </div>
                <div className="min-w-0 flex-1">
                  <CardTitle className="text-lg font-semibold text-gray-900 truncate">
                    {project.name}
                  </CardTitle>
                  <p className="text-sm text-gray-600 mt-1">
                    Created {new Date(project.created_at).toLocaleDateString()}
                  </p>
                </div>
              </div>

              {/* Actions Menu */}
              <DropdownMenu>
                <DropdownMenuTrigger asChild>
                  <Button 
                    variant="ghost" 
                    size="sm" 
                    className="opacity-0 group-hover:opacity-100 transition-opacity"
                  >
                    <EllipsisVerticalIcon className="w-4 h-4" />
                  </Button>
                </DropdownMenuTrigger>
                <DropdownMenuContent align="end">
                  <DropdownMenuItem onClick={() => handleEditClick(project)}>
                    <PencilIcon className="w-4 h-4 mr-2" />
                    Edit Project
                  </DropdownMenuItem>
                  <DropdownMenuSeparator />
                  <DropdownMenuItem 
                    className="text-red-600"
                    onClick={() => handleDeleteClick(project)}
                  >
                    <TrashIcon className="w-4 h-4 mr-2" />
                    Delete Project
                  </DropdownMenuItem>
                </DropdownMenuContent>
              </DropdownMenu>
            </div>
          </CardHeader>

          <CardContent className="pt-0">
            {/* Description */}
            {project.description && (
              <p className="text-sm text-gray-600 mb-4 line-clamp-2">
                {project.description}
              </p>
            )}

            {/* Stats */}
            <div className="flex items-center gap-4 mb-4 text-sm text-gray-600">
              <div className="flex items-center gap-1">
                <DocumentTextIcon className="w-4 h-4" />
                <span>{project.document_count} documents</span>
              </div>
              {project.chat_count !== undefined && (
                <div className="flex items-center gap-1">
                  <ChatBubbleLeftRightIcon className="w-4 h-4" />
                  <span>{project.chat_count} chats</span>
                </div>
              )}
            </div>

            {/* Status Badge */}
            <div className="flex items-center justify-between">
              <Badge 
                variant={project.document_count > 0 ? "default" : "secondary"}
                className="text-xs"
              >
                {project.document_count > 0 ? (
                  <>
                    <CheckCircleSolidIcon className="w-3 h-3 mr-1" />
                    Active
                  </>
                ) : (
                  <>
                    <ExclamationTriangleIcon className="w-3 h-3 mr-1" />
                    Empty
                  </>
                )}
              </Badge>

              {/* Quick Actions */}
              <div className="flex items-center gap-2">
                <Button variant="ghost" size="sm" className="text-xs">
                  <ArrowRightIcon className="w-3 h-3 mr-1" />
                  Open
                </Button>
              </div>
            </div>
          </CardContent>
        </Card>
      </motion.div>
    )
  }

  return (
    <div className="h-full flex flex-col bg-gray-50">
      {/* Header */}
      <div className="bg-white border-b border-gray-200 p-6">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-gradient-to-br from-green-500 to-teal-600 rounded-lg flex items-center justify-center">
              <FolderIcon className="w-5 h-5 text-white" />
            </div>
            <div>
              <h1 className="text-xl font-semibold text-gray-900">Projects</h1>
              <p className="text-sm text-gray-600">
                Manage your document collections and AI workspaces
              </p>
            </div>
          </div>

          <Button 
            onClick={handleCreateClick}
            className="bg-gradient-to-r from-green-500 to-teal-600 hover:from-green-600 hover:to-teal-700"
          >
            <PlusIcon className="w-4 h-4 mr-2" />
            New Project
          </Button>
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
        ) : projects.length === 0 ? (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="text-center py-12"
          >
            <div className="w-16 h-16 bg-gradient-to-br from-green-500 to-teal-600 rounded-full flex items-center justify-center mx-auto mb-6">
              <FolderIcon className="w-8 h-8 text-white" />
            </div>
            <h3 className="text-lg font-medium text-gray-900 mb-2">No projects yet</h3>
            <p className="text-gray-600 max-w-md mx-auto mb-6">
              Create your first project to start organizing documents and building your AI-powered knowledge base.
            </p>
            <Button 
              onClick={handleCreateClick}
              className="bg-gradient-to-r from-green-500 to-teal-600 hover:from-green-600 hover:to-teal-700"
            >
              <PlusIcon className="w-4 h-4 mr-2" />
              Create First Project
            </Button>
          </motion.div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            <AnimatePresence>
              {projects.map((project, index) => (
                <ProjectCard key={project.id} project={project} index={index} />
              ))}
            </AnimatePresence>
          </div>
        )}
      </div>

      {/* Create Project Dialog */}
      <Dialog open={showCreateDialog} onOpenChange={setShowCreateDialog}>
        <DialogContent className="sm:max-w-md">
          <DialogHeader>
            <DialogTitle>Create New Project</DialogTitle>
            <DialogDescription>
              Create a new project to organize your documents and AI conversations.
            </DialogDescription>
          </DialogHeader>
          
          <div className="space-y-4">
            <div>
              <Label htmlFor="name">Project Name</Label>
              <Input
                id="name"
                value={formData.name}
                onChange={(e) => setFormData(prev => ({ ...prev, name: e.target.value }))}
                placeholder="Enter project name..."
                className="mt-1"
              />
            </div>
            
            <div>
              <Label htmlFor="description">Description (Optional)</Label>
              <Textarea
                id="description"
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
              className="bg-gradient-to-r from-green-500 to-teal-600 hover:from-green-600 hover:to-teal-700"
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
        <DialogContent className="sm:max-w-md">
          <DialogHeader>
            <DialogTitle>Edit Project</DialogTitle>
            <DialogDescription>
              Update your project information.
            </DialogDescription>
          </DialogHeader>
          
          <div className="space-y-4">
            <div>
              <Label htmlFor="edit-name">Project Name</Label>
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
              onClick={() => {
                // TODO: Implement project update
                toast.info('Project update functionality coming soon')
                setShowEditDialog(false)
              }}
              disabled={!formData.name.trim() || isSubmitting}
            >
              <PencilIcon className="w-4 h-4 mr-2" />
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