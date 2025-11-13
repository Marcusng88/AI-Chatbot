import { Input } from './ui/input';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select';
import { Label } from './ui/label';

type ChatFiltersProps = {
  filters: {
    dateFrom: string;
    dateTo: string;
    mediaType: string;
    keywords: string;
  };
  setFilters: (filters: any) => void;
};

export function ChatFilters({ filters, setFilters }: ChatFiltersProps) {
  return (
    <div className="p-4 bg-stone-50 border-b border-stone-200 grid grid-cols-2 gap-4">
      <div>
        <Label className="text-xs text-stone-600 mb-1">Date From</Label>
        <Input
          type="date"
          value={filters.dateFrom}
          onChange={(e) => setFilters({ ...filters, dateFrom: e.target.value })}
          className="text-sm"
        />
      </div>
      
      <div>
        <Label className="text-xs text-stone-600 mb-1">Date To</Label>
        <Input
          type="date"
          value={filters.dateTo}
          onChange={(e) => setFilters({ ...filters, dateTo: e.target.value })}
          className="text-sm"
        />
      </div>
      
      <div>
        <Label className="text-xs text-stone-600 mb-1">Media Type</Label>
        <Select value={filters.mediaType} onValueChange={(value) => setFilters({ ...filters, mediaType: value })}>
          <SelectTrigger className="text-sm">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="all">All Types</SelectItem>
            <SelectItem value="image">Images</SelectItem>
            <SelectItem value="video">Videos</SelectItem>
            <SelectItem value="audio">Audio</SelectItem>
            <SelectItem value="document">Documents</SelectItem>
          </SelectContent>
        </Select>
      </div>
      
      <div>
        <Label className="text-xs text-stone-600 mb-1">Keywords (comma separated)</Label>
        <Input
          value={filters.keywords}
          onChange={(e) => setFilters({ ...filters, keywords: e.target.value })}
          placeholder="batik, heritage, craft"
          className="text-sm"
        />
      </div>
    </div>
  );
}
