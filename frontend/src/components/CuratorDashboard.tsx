import { useState } from 'react';
import { Upload, Tag, TrendingUp, Clock, FileCheck, AlertCircle, BarChart3 } from 'lucide-react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { Progress } from './ui/progress';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';
import { ScrollArea } from './ui/scroll-area';
import type { ArchiveItem } from '../App';
import { AddArchiveModal } from './AddArchiveModal';
import { ArchiveDetailModal } from './ArchiveDetailModal';
import { ArchiveCard } from './ArchiveCard';
import { toast } from 'sonner';
import { deleteArchive } from '../services/api';

type CuratorDashboardProps = {
  archives: ArchiveItem[];
  setArchives: (archives: ArchiveItem[]) => void;
};

export function CuratorDashboard({ archives, setArchives }: CuratorDashboardProps) {
  const [isAddModalOpen, setIsAddModalOpen] = useState(false);
  const [detailOpen, setDetailOpen] = useState(false);
  const [selectedItem, setSelectedItem] = useState<ArchiveItem | null>(null);
  const [isEditMode, setIsEditMode] = useState(false);

  // Calculate statistics
  const stats = {
    totalItems: 12847,
    thisMonth: 234,
    pendingReview: 18,
    storageUsed: 68,
  };

  const recentUploads = archives.slice(0, 5);
  
  const popularTags = [
    { name: 'batik', count: 1247 },
    { name: 'heritage', count: 982 },
    { name: 'traditional', count: 856 },
    { name: 'architecture', count: 743 },
    { name: 'crafts', count: 621 },
    { name: 'costume', count: 512 },
    { name: 'oral history', count: 389 },
    { name: 'colonial', count: 267 },
  ];

  const recentActivity = [
    { action: 'Upload', item: 'Traditional Batik Pattern', user: 'Ahmad Ibrahim', time: '2 hours ago' },
    { action: 'Edit', item: 'Heritage Building Documentation', user: 'Siti Nurhaliza', time: '5 hours ago' },
    { action: 'Delete', item: 'Duplicate Image File', user: 'Lee Wei Ming', time: '1 day ago' },
    { action: 'Upload', item: 'Oral History Interview', user: 'Ahmad Ibrahim', time: '1 day ago' },
    { action: 'Tag Update', item: 'Wayang Kulit Performance', user: 'Siti Nurhaliza', time: '2 days ago' },
  ];

  const handleBulkTagging = () => {
    toast.success('Bulk tagging interface coming soon');
  };

  const handleExport = () => {
    toast.success('Export started - you will receive an email when ready');
  };

  const handleDeleteArchive = async (id: string) => {
    try {
      await deleteArchive(id);
      setArchives(archives.filter((a) => a.id !== id));
      toast.success('Archive item deleted successfully');
    } catch (error) {
      console.error('Error deleting archive:', error);
      toast.error('Failed to delete archive item');
    }
  };

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-stone-800">Curator Dashboard</h1>
          <p className="text-stone-500">Manage your digital heritage collection</p>
        </div>
        <Button onClick={() => setIsAddModalOpen(true)} className="bg-forest hover:bg-forest">
          <Upload className="w-4 h-4 mr-2" />
          Upload New Item
        </Button>
      </div>

      {/* Statistics Cards */}
      <div className="flex gap-4 w-full">
        <Card className="flex-1">
          <CardHeader className="pb-3">
            <CardTitle className="text-sm flex items-center justify-between">
              Total Archives
              <BarChart3 className="w-4 h-4 text-stone-500" />
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl text-stone-800">{stats.totalItems.toLocaleString()}</div>
            <p className="text-xs text-stone-500 mt-1">Across all categories</p>
          </CardContent>
        </Card>

        <Card className="flex-1">
          <CardHeader className="pb-3">
            <CardTitle className="text-sm flex items-center justify-between">
              This Month
              <TrendingUp className="w-4 h-4 text-green-500" />
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl text-stone-800">+{stats.thisMonth}</div>
            <p className="text-xs text-stone-500 mt-1">
              <span className="text-green-600">↑ 12%</span> from last month
            </p>
          </CardContent>
        </Card>

        <Card className="flex-1">
          <CardHeader className="pb-3">
            <CardTitle className="text-sm flex items-center justify-between">
              Storage Used
              <FileCheck className="w-4 h-4 text-blue-500" />
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl text-stone-800">{stats.storageUsed}%</div>
            <Progress value={stats.storageUsed} className="mt-2 h-1" />
          </CardContent>
        </Card>
      </div>

      {/* Main Content */}
      <Tabs defaultValue="recent" className="space-y-4 w-full">
        <TabsList className="w-full flex justify-start gap-3 overflow-x-auto">
          <TabsTrigger value="recent" className="flex-1 min-w-[150px]">Recent Uploads</TabsTrigger>
          <TabsTrigger value="tags" className="flex-1 min-w-[150px]">Tag Management</TabsTrigger>
          <TabsTrigger value="activity" className="flex-1 min-w-[150px]">Activity Log</TabsTrigger>
          <TabsTrigger value="tools" className="flex-1 min-w-[150px]">Curator Tools</TabsTrigger>
        </TabsList>


        {/* Recent Uploads */}
        <TabsContent value="recent" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Clock className="w-5 h-5" />
                Recently Added Items
              </CardTitle>
              <CardDescription>Latest additions to the archive (last 30 days)</CardDescription>
            </CardHeader>
            <CardContent>
              <ScrollArea className="h-[500px]">
                <div className="space-y-3">
                  {recentUploads.map((item) => (
                    <ArchiveCard
                      key={item.id}
                      item={item}
                      onView={(it) => { setSelectedItem(it); setIsEditMode(false); setDetailOpen(true); }}
                      onEdit={(it) => { setSelectedItem(it); setIsEditMode(true); setDetailOpen(true); }}
                      onDelete={handleDeleteArchive}
                      viewMode="list"
                    />
                  ))}
                </div>
              </ScrollArea>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Tag Management */}
        <TabsContent value="tags" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Tag className="w-5 h-5" />
                Tag Management
              </CardTitle>
              <CardDescription>Manage tags and metadata across your collection</CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div>
                <h3 className="text-stone-800 mb-3">Most Used Tags</h3>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                  {popularTags.map((tag) => (
                    <div
                      key={tag.name}
                      className="border border-stone-200 rounded-lg p-3 hover:border-forest cursor-pointer transition-colors"
                    >
                      <div className="flex items-center justify-between mb-1">
                        <Badge variant="outline">{tag.name}</Badge>
                        <Button variant="ghost" size="sm" className="h-6 w-6 p-0">
                          ⋮
                        </Button>
                      </div>
                      <p className="text-xs text-stone-500">{tag.count.toLocaleString()} items</p>
                    </div>
                  ))}
                </div>
              </div>

              <div className="flex gap-2">
                <Button variant="outline" className="flex-1">
                  <Tag className="w-4 h-4 mr-2" />
                  Create New Tag
                </Button>
                <Button variant="outline" className="flex-1" onClick={handleBulkTagging}>
                  <Tag className="w-4 h-4 mr-2" />
                  Bulk Tagging
                </Button>
                <Button variant="outline" className="flex-1">
                  <Tag className="w-4 h-4 mr-2" />
                  Merge Tags
                </Button>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Activity Log */}
        <TabsContent value="activity" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Clock className="w-5 h-5" />
                Activity Log
              </CardTitle>
              <CardDescription>Recent actions by curators</CardDescription>
            </CardHeader>
            <CardContent>
              <ScrollArea className="h-[500px]">
                <div className="space-y-3">
                  {recentActivity.map((activity, idx) => (
                    <div key={idx} className="flex items-start gap-3 pb-3 border-b border-stone-100 last:border-0">
                      <div className={`w-8 h-8 rounded-full flex items-center justify-center shrink-0 ${
                        activity.action === 'Upload' ? 'bg-green-100 text-green-600' :
                        activity.action === 'Edit' ? 'bg-blue-100 text-blue-600' :
                        activity.action === 'Delete' ? 'bg-red-100 text-red-600' :
                        'bg-forest text-white'
                      }`}>
                        {activity.action === 'Upload' && <Upload className="w-4 h-4" />}
                        {activity.action === 'Edit' && <FileCheck className="w-4 h-4" />}
                        {activity.action === 'Delete' && <AlertCircle className="w-4 h-4" />}
                        {activity.action === 'Tag Update' && <Tag className="w-4 h-4" />}
                      </div>
                      <div className="flex-1">
                        <p className="text-sm text-stone-800">
                          <span className="font-medium">{activity.user}</span> {activity.action.toLowerCase()}d{' '}
                          <span className="font-medium">{activity.item}</span>
                        </p>
                        <p className="text-xs text-stone-500">{activity.time}</p>
                      </div>
                    </div>
                  ))}
                </div>
              </ScrollArea>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Curator Tools */}
        <TabsContent value="tools" className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <Card className="cursor-pointer hover:shadow-lg transition-shadow" onClick={handleBulkTagging}>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Tag className="w-5 h-5 text-forest" />
                  Bulk Operations
                </CardTitle>
                <CardDescription>Edit multiple items at once</CardDescription>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-stone-600">
                  Apply tags, update metadata, or move items to different collections in bulk.
                </p>
              </CardContent>
            </Card>

            <Card className="cursor-pointer hover:shadow-lg transition-shadow" onClick={handleExport}>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <FileCheck className="w-5 h-5 text-blue-600" />
                  Export Data
                </CardTitle>
                <CardDescription>Export collection data</CardDescription>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-stone-600">
                  Generate CSV, JSON, or XML exports of your collection metadata.
                </p>
              </CardContent>
            </Card>

            <Card className="cursor-pointer hover:shadow-lg transition-shadow">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <TrendingUp className="w-5 h-5 text-green-600" />
                  Analytics
                </CardTitle>
                <CardDescription>Collection insights</CardDescription>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-stone-600">
                  View detailed statistics and trends about your collection usage and growth.
                </p>
              </CardContent>
            </Card>

            <Card className="cursor-pointer hover:shadow-lg transition-shadow">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <AlertCircle className="w-5 h-5 text-red-600" />
                  Quality Check
                </CardTitle>
                <CardDescription>Data validation</CardDescription>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-stone-600">
                  Identify incomplete metadata, broken links, or duplicate items.
                </p>
              </CardContent>
            </Card>
          </div>
        </TabsContent>
      </Tabs>

      <AddArchiveModal
        isOpen={isAddModalOpen}
        onClose={() => setIsAddModalOpen(false)}
        onAdd={(newItem) => setArchives([newItem, ...archives])}
      />
      <ArchiveDetailModal
        isOpen={detailOpen}
        item={selectedItem}
        onClose={() => setDetailOpen(false)}
        initialEditMode={isEditMode}
        onSave={(updated) => {
          setArchives(archives.map((a) => a.id === updated.id ? updated : a));
          setSelectedItem(updated);
        }}
      />
    </div>
  );
}
