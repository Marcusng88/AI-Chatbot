import { Image as ImageIcon, Video, FileText, Music, Eye, Trash2, Edit, Calendar, Tag } from 'lucide-react';
import type { ArchiveItem } from '../App';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { ImageWithFallback } from './figma/ImageWithFallback';
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from './ui/alert-dialog';
import { toast } from 'sonner';

type ArchiveCardProps = {
  item: ArchiveItem;
  onDelete?: (id: string) => void;
  onView?: (item: ArchiveItem) => void;
  onEdit?: (item: ArchiveItem) => void;
  compact?: boolean;
  viewMode?: 'grid' | 'list';
};

export function ArchiveCard({ item, onDelete, onView, onEdit, compact = false, viewMode = 'grid' }: ArchiveCardProps) {
  const getTypeIcon = (type: string) => {
    switch (type) {
      case 'image':
        return ImageIcon;
      case 'video':
        return Video;
      case 'audio':
        return Music;
      case 'document':
        return FileText;
      default:
        return FileText;
    }
  };

  const getTypeColor = (type: string) => {
    switch (type) {
      case 'image':
        return 'bg-blue-100 text-blue-700 border-blue-200';
      case 'video':
        return 'bg-purple-100 text-purple-700 border-purple-200';
      case 'audio':
        return 'bg-green-100 text-green-700 border-green-200';
      case 'document':
        return 'bg-forest text-white border-forest';
      default:
        return 'bg-stone-100 text-stone-700 border-stone-200';
    }
  };

  const Icon = getTypeIcon(item.type);

  const handleDelete = () => {
    if (onDelete) {
      onDelete(item.id);
      toast.success('Archive item deleted');
    }
  };

  if (compact) {
    return (
      <div
        className="bg-white border border-stone-200 rounded-lg p-3 flex gap-3 hover:shadow-md hover:border-forest transition-all cursor-pointer"
        onClick={() => onView && onView(item)}
      >
        {item.thumbnail ? (
          <ImageWithFallback
            src={item.thumbnail}
            alt={item.title}
            className="w-16 h-16 rounded object-cover shrink-0"
          />
        ) : (
          <div className={`w-16 h-16 rounded flex items-center justify-center shrink-0 ${getTypeColor(item.type)}`}>
            <Icon className="w-6 h-6" />
          </div>
        )}
        <div className="flex-1 min-w-0">
          <div className="flex items-start justify-between gap-2 mb-1">
            <h4 className="text-sm text-stone-800 truncate flex-1">{item.title}</h4>
            <Badge className={`${getTypeColor(item.type)} text-xs shrink-0`}>{item.type}</Badge>
          </div>
          <p className="text-xs text-stone-500 line-clamp-2 mb-2">{item.description}</p>
          <div className="flex gap-1 flex-wrap">
            {item.tags.slice(0, 3).map((tag) => (
              <Badge key={tag} variant="outline" className="text-xs px-1.5 py-0">
                {tag}
              </Badge>
            ))}
            {item.tags.length > 3 && (
              <Badge variant="outline" className="text-xs px-1.5 py-0">
                +{item.tags.length - 3}
              </Badge>
            )}
          </div>
        </div>
      </div>
    );
  }

  if (viewMode === 'list') {
    return (
      <div className="bg-white border border-stone-200 rounded-lg p-4 flex gap-4 hover:shadow-md transition-shadow">
        {item.thumbnail && (
          <ImageWithFallback
            src={item.thumbnail}
            alt={item.title}
            className="w-20 h-20 object-cover rounded shrink-0"
          />
        )}
        {!item.thumbnail && (
          <div className={`w-20 h-20 rounded flex items-center justify-center shrink-0 ${getTypeColor(item.type)}`}>
            <Icon className="w-8 h-8" />
          </div>
        )}
        
        <div className="flex-1 min-w-0">
          <div className="flex items-start justify-between mb-2">
            <div className="flex-1">
              <h3 className="text-stone-800 mb-1">{item.title}</h3>
              <p className="text-sm text-stone-500">{item.description}</p>
            </div>
            <Badge className={getTypeColor(item.type)}>{item.type}</Badge>
          </div>
          
          <div className="flex items-center gap-4 text-xs text-stone-500 mb-2">
            <div className="flex items-center gap-1">
              <Calendar className="w-3 h-3" />
              {new Date(item.date).toLocaleDateString('en-MY')}
            </div>
            <div className="flex items-center gap-1">
              <Tag className="w-3 h-3" />
              {item.tags.length} tag{item.tags.length !== 1 ? 's' : ''}
            </div>
          </div>
          
          <div className="flex gap-1 flex-wrap">
            {item.tags.map((tag) => (
              <Badge key={tag} variant="outline" className="text-xs">
                {tag}
              </Badge>
            ))}
          </div>
        </div>
        
        {onDelete && (
          <div className="flex flex-col gap-2 shrink-0">
            <Button variant="outline" size="sm" onClick={() => onView && onView(item)}>
              <Eye className="w-3 h-3 mr-1" />
              View
            </Button>
            <Button variant="outline" size="sm" onClick={() => onEdit && onEdit(item)}>
              <Edit className="w-3 h-3 mr-1" />
              Edit
            </Button>
            <AlertDialog>
              <AlertDialogTrigger asChild>
                <Button variant="outline" size="sm" className="text-red-600 hover:text-red-700">
                  <Trash2 className="w-3 h-3 mr-1" />
                  Delete
                </Button>
              </AlertDialogTrigger>
              <AlertDialogContent>
                <AlertDialogHeader>
                  <AlertDialogTitle>Delete Archive Item?</AlertDialogTitle>
                  <AlertDialogDescription>
                    This will permanently delete "{item.title}" from the archive. This action cannot be undone.
                  </AlertDialogDescription>
                </AlertDialogHeader>
                <AlertDialogFooter>
                  <AlertDialogCancel>Cancel</AlertDialogCancel>
                  <AlertDialogAction onClick={handleDelete} className="bg-red-600 hover:bg-red-700">
                    Delete
                  </AlertDialogAction>
                </AlertDialogFooter>
              </AlertDialogContent>
            </AlertDialog>
          </div>
        )}
      </div>
    );
  }

  return (
    <div className="bg-white border border-stone-200 rounded-lg overflow-hidden hover:shadow-lg transition-shadow">
      {/* Thumbnail */}
      {item.thumbnail ? (
        <ImageWithFallback
          src={item.thumbnail}
          alt={item.title}
          className="w-full h-40 object-cover"
        />
      ) : (
        <div className={`w-full h-40 flex items-center justify-center ${getTypeColor(item.type)}`}>
          <Icon className="w-12 h-12" />
        </div>
      )}

      {/* Content */}
      <div className="p-4">
        <div className="flex items-start justify-between mb-2">
          <h3 className="text-stone-800 flex-1">{item.title}</h3>
          <Badge className={getTypeColor(item.type)}>{item.type}</Badge>
        </div>
        
        <p className="text-sm text-stone-500 mb-3 line-clamp-2">{item.description}</p>
        
        <div className="flex items-center gap-2 text-xs text-stone-500 mb-3">
          <Calendar className="w-3 h-3" />
          {new Date(item.date).toLocaleDateString('en-MY')}
        </div>
        
        <div className="flex gap-1 flex-wrap mb-4">
          {item.tags.map((tag) => (
            <Badge key={tag} variant="outline" className="text-xs">
              {tag}
            </Badge>
          ))}
        </div>

        {/* Actions */}
        {onDelete && (
          <div className="flex gap-2">
            <Button variant="outline" size="sm" className="flex-1" onClick={() => onView && onView(item)}>
              <Eye className="w-3 h-3 mr-1" />
              View
            </Button>
            <Button variant="outline" size="sm" className="flex-1" onClick={() => onEdit && onEdit(item)}>
              <Edit className="w-3 h-3 mr-1" />
              Edit
            </Button>
            <AlertDialog>
              <AlertDialogTrigger asChild>
                <Button variant="outline" size="sm" className="text-red-600 hover:text-red-700">
                  <Trash2 className="w-3 h-3" />
                </Button>
              </AlertDialogTrigger>
              <AlertDialogContent>
                <AlertDialogHeader>
                  <AlertDialogTitle>Delete Archive Item?</AlertDialogTitle>
                  <AlertDialogDescription>
                    This will permanently delete "{item.title}" from the archive. This action cannot be undone.
                  </AlertDialogDescription>
                </AlertDialogHeader>
                <AlertDialogFooter>
                  <AlertDialogCancel>Cancel</AlertDialogCancel>
                  <AlertDialogAction onClick={handleDelete} className="bg-red-600 hover:bg-red-700">
                    Delete
                  </AlertDialogAction>
                </AlertDialogFooter>
              </AlertDialogContent>
            </AlertDialog>
          </div>
        )}
      </div>
    </div>
  );
}