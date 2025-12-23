import { useState } from 'react';
import { FileText, Download } from 'lucide-react';
import { Button } from './ui/button';
import { ScrollArea } from './ui/scroll-area';
import type { ArchiveItem } from '../App';
import { ArchiveCard } from './ArchiveCard';
import { Badge } from './ui/badge';
import { ArchiveDetailModal } from './ArchiveDetailModal';

type SearchResultsPanelProps = {
  results: ArchiveItem[];
  query: string;
};

export function SearchResultsPanel({ results, query }: SearchResultsPanelProps) {
  const [detailOpen, setDetailOpen] = useState(false);
  const [selectedItem, setSelectedItem] = useState<ArchiveItem | null>(null);

  if (!query && results.length === 0) {
    return (
      <div className="bg-white rounded-xl shadow-lg border border-stone-200 flex flex-col h-full">
        <div className="p-6 flex-1 flex items-center justify-center">
          <div className="text-center text-stone-400">
            <FileText className="w-16 h-16 mx-auto mb-4 opacity-50" />
            <h3 className="text-stone-600 mb-2">Start Your Search</h3>
            <p className="text-sm">
              Use the chatbot to search through thousands of heritage items
            </p>
            <div className="mt-6 space-y-2">
              <p className="text-xs text-stone-500">Try searching for:</p>
              <div className="flex flex-wrap gap-2 justify-center">
                <Badge variant="outline" className="cursor-pointer hover:bg-stone-100">
                  batik patterns
                </Badge>
                <Badge variant="outline" className="cursor-pointer hover:bg-stone-100">
                  colonial buildings
                </Badge>
                <Badge variant="outline" className="cursor-pointer hover:bg-stone-100">
                  traditional crafts
                </Badge>
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-xl shadow-lg border border-stone-200 flex flex-col h-full">
      {/* Header */}
      <div className="p-4 border-b border-stone-200">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-stone-800">Search Results</h2>
            <p className="text-sm text-stone-500">
              {results.length > 0 
                ? `Found ${results.length} item${results.length !== 1 ? 's' : ''}`
                : 'No results found'}
            </p>
          </div>
          {results.length > 0 && (
            <Button variant="outline" size="sm">
              <Download className="w-4 h-4 mr-2" />
              Export
            </Button>
          )}
        </div>
      </div>

      {/* Results */}
      <ScrollArea className="flex-1 p-4">
        {results.length > 0 ? (
          <div className="space-y-3">
            {results.map((item) => (
              <ArchiveCard
                key={item.id}
                item={item}
                compact
                onView={(archiveItem) => {
                  setSelectedItem(archiveItem);
                  setDetailOpen(true);
                }}
              />
            ))}
          </div>
        ) : (
          <div className="flex items-center justify-center h-full text-stone-400">
            <div className="text-center">
              <p>No items found matching "{query}"</p>
              <p className="text-sm mt-2">Try using different keywords or filters</p>
            </div>
          </div>
        )}
      </ScrollArea>

      <ArchiveDetailModal
        isOpen={detailOpen}
        item={selectedItem}
        onClose={() => {
          setDetailOpen(false);
          setSelectedItem(null);
        }}
      />
    </div>
  );
}
