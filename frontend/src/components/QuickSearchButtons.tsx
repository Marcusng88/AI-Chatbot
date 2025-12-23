import { Button } from './ui/button';
import { Sparkles } from 'lucide-react';

type QuickSearchButtonsProps = {
  onSearch: (query: string) => void;
};

export function QuickSearchButtons({ onSearch }: QuickSearchButtonsProps) {
  const quickSearches = [
    'batik patterns',
    'traditional architecture',
    'colonial buildings',
    'oral history interviews',
    'traditional crafts',
    'wayang kulit',
  ];

  return (
    <div className="p-4 bg-stone-50 border-t border-stone-200">
      <div className="flex items-center gap-2 mb-2">
        <Sparkles className="w-4 h-4 text-forest" />
        <p className="text-xs text-stone-600">Quick Searches:</p>
      </div>
      <div className="flex flex-wrap gap-2">
        {quickSearches.map((query) => (
          <Button
            key={query}
            variant="outline"
            size="sm"
            onClick={() => onSearch(query)}
            className="text-xs"
          >
            {query}
          </Button>
        ))}
      </div>
    </div>
  );
}
