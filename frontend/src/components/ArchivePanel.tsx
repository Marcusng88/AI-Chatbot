import { useState } from 'react';
import { Plus, Search, Grid3x3, List, SortAsc } from 'lucide-react';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select';
import type { ArchiveItem } from '../App';
import { ArchiveCard } from './ArchiveCard';
import { AddArchiveModal } from './AddArchiveModal';
import { ArchiveDetailModal } from './ArchiveDetailModal';
import { ScrollArea } from './ui/scroll-area';
import { deleteArchive } from '../services/api';
import { toast } from 'sonner';

type ArchivePanelProps = {
  archives: ArchiveItem[];
  setArchives: (archives: ArchiveItem[]) => void;
  mode?: 'preview' | 'full';
};

export function ArchivePanel({ archives, setArchives, mode = 'full' }: ArchivePanelProps) {
  const [searchTerm, setSearchTerm] = useState('');
  const [sortBy, setSortBy] = useState('date-desc');
  const [viewMode, setViewMode] = useState<'grid' | 'list'>('grid');
  const [isAddModalOpen, setIsAddModalOpen] = useState(false);
  const [detailOpen, setDetailOpen] = useState(false);
  const [selectedItem, setSelectedItem] = useState<ArchiveItem | null>(null);
  const [isEditMode, setIsEditMode] = useState(false);

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

  const filteredArchives = archives
    .filter((item) => {
      const search = searchTerm.toLowerCase();
      return (
        item.title.toLowerCase().includes(search) ||
        item.description.toLowerCase().includes(search) ||
        item.tags.some((tag) => tag.toLowerCase().includes(search))
      );
    })
    .sort((a, b) => {
      switch (sortBy) {
        case 'date-desc':
          return new Date(b.date).getTime() - new Date(a.date).getTime();
        case 'date-asc':
          return new Date(a.date).getTime() - new Date(b.date).getTime();
        case 'title-asc':
          return a.title.localeCompare(b.title);
        case 'title-desc':
          return b.title.localeCompare(a.title);
        default:
          return 0;
      }
    });

  return (
    <div className="bg-white rounded-xl shadow-lg border border-stone-200 flex flex-col h-full">
      {/* Header */}
      <div className="p-4 border-b border-stone-200">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h2 className="text-stone-800">Archive Collection</h2>
            <p className="text-sm text-stone-500">
              {filteredArchives.length} item{filteredArchives.length !== 1 ? 's' : ''}
            </p>
          </div>
          
          {mode === 'full' && (
            <div className="flex items-center gap-2">
              <div className="flex border border-stone-200 rounded-lg overflow-hidden">
                <Button
                  variant={viewMode === 'grid' ? 'default' : 'ghost'}
                  size="sm"
                  onClick={() => setViewMode('grid')}
                  className={viewMode === 'grid' ? 'bg-forest hover:bg-forest' : ''}
                >
                  <Grid3x3 className="w-4 h-4" />
                </Button>
                <Button
                  variant={viewMode === 'list' ? 'default' : 'ghost'}
                  size="sm"
                  onClick={() => setViewMode('list')}
                  className={viewMode === 'list' ? 'bg-forest hover:bg-forest' : ''}
                >
                  <List className="w-4 h-4" />
                </Button>
              </div>
              
              <Button
                onClick={() => setIsAddModalOpen(true)}
                className="bg-forest hover:bg-forest"
              >
                <Plus className="w-4 h-4 mr-2" />
                Add Item
              </Button>
            </div>
          )}
        </div>

        {/* Search & Sort */}
        <div className="flex gap-2">
          <div className="flex-1 relative">
            <Search className="w-4 h-4 absolute left-3 top-1/2 -translate-y-1/2 text-stone-400" />
            <Input
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              placeholder="Search archives..."
              className="pl-9"
            />
          </div>
          
          <Select value={sortBy} onValueChange={setSortBy}>
            <SelectTrigger className="w-40">
              <SortAsc className="w-4 h-4 mr-2" />
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="date-desc">Newest First</SelectItem>
              <SelectItem value="date-asc">Oldest First</SelectItem>
              <SelectItem value="title-asc">Title A-Z</SelectItem>
              <SelectItem value="title-desc">Title Z-A</SelectItem>
            </SelectContent>
          </Select>
        </div>
      </div>

      {/* Archive List */}
      <ScrollArea className="flex-1 p-4">
        {filteredArchives.length === 0 ? (
          <div className="flex items-center justify-center h-full text-stone-400">
            <div className="text-center">
              <p>No archives found</p>
              {mode === 'full' && (
                <Button
                  variant="link"
                  onClick={() => setIsAddModalOpen(true)}
                  className="text-forest mt-2"
                >
                  Add your first item
                </Button>
              )}
            </div>
          </div>
        ) : (
          <div
            className={
              viewMode === 'grid'
                ? 'grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4'
                : 'space-y-3'
            }
          >
            {filteredArchives.map((item) => (
              <ArchiveCard
                key={item.id}
                item={item}
                onView={(it) => { setSelectedItem(it); setIsEditMode(false); setDetailOpen(true); }}
                onEdit={(it) => { setSelectedItem(it); setIsEditMode(true); setDetailOpen(true); }}
                onDelete={handleDeleteArchive}
                viewMode={viewMode}
              />
            ))}
          </div>
        )}
      </ScrollArea>

      {/* Add Archive Modal */}
      <AddArchiveModal
        isOpen={isAddModalOpen}
        onClose={() => setIsAddModalOpen(false)}
        onAdd={(newItem) => setArchives([...archives, newItem])}
      />

      {/* Archive Detail Modal */}
      <ArchiveDetailModal
        isOpen={detailOpen}
        item={selectedItem}
        onClose={() => setDetailOpen(false)}
        initialEditMode={isEditMode}
        onSave={(updated: ArchiveItem) => {
          setArchives(archives.map((a) => a.id === updated.id ? updated : a));
          setSelectedItem(updated);
        }}
      />
    </div>
  );
}
