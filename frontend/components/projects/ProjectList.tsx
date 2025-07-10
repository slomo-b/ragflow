// === components/projects/ProjectList.tsx ===
'use client'

import React from 'react'
import { motion } from 'framer-motion'
import { FolderIcon } from '@heroicons/react/24/outline'
import { Button } from "@/components/ui/button"
import { useStore } from '@/stores/useStore'
import { cn } from '@/lib/utils'

export function ProjectList() {
  const { projects, currentProject, setCurrentProject } = useStore()

  if (projects.length === 0) {
    return (
      <div className="text-center py-8 text-gray-500 dark:text-gray-400">
        <FolderIcon className="h-8 w-8 mx-auto mb-2 opacity-50" />
        <p className="text-sm">No projects yet</p>
      </div>
    )
  }

  return (
    <div className="space-y-2">
      {projects.map((project, index) => (
        <motion.div
          key={project.id}
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.3, delay: index * 0.1 }}
        >
          <Button
            variant="ghost"
            onClick={() => setCurrentProject(project)}
            className={cn(
              "w-full justify-start text-left p-3 h-auto",
              currentProject?.id === project.id && "bg-gray-100 dark:bg-gray-800"
            )}
          >
            <FolderIcon className="h-4 w-4 mr-2 flex-shrink-0" />
            <div className="truncate">
              <div className="font-medium text-sm truncate">{project.name}</div>
              <div className="text-xs text-gray-500 dark:text-gray-400">
                {new Intl.DateTimeFormat('en-US', {
                  month: 'short',
                  day: 'numeric',
                }).format(project.createdAt)}
              </div>
            </div>
          </Button>
        </motion.div>
      ))}
    </div>
  )
}