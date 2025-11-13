import { useState, useRef, useEffect } from 'react';
import { Send, Paperclip, Filter } from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { ScrollArea } from './ui/scroll-area';
import type { ChatMessage as ChatMessageType, ArchiveItem } from '../App';
import { ChatMessage } from './ChatMessage';
import { ChatFilters } from './ChatFilters';
import { QuickSearchButtons } from './QuickSearchButtons';
import { TypingIndicator } from './TypingIndicator';
import { toast } from 'sonner';
import { aiSearchArchives, type ArchiveResponse } from '../services/api';

type ChatPanelProps = {
  messages: ChatMessageType[];
  setMessages: React.Dispatch<React.SetStateAction<ChatMessageType[]>>;
  archives: ArchiveItem[];
};

type FilterState = {
  dateFrom: string;
  dateTo: string;
  mediaType: string;
  keywords: string;
};

// Archives prop kept for backward compatibility but using AI search API instead
export function ChatPanel({ messages, setMessages }: Omit<ChatPanelProps, 'archives'> & { archives?: ArchiveItem[] }) {
  const [input, setInput] = useState('');
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const [showFilters, setShowFilters] = useState(false);
  const [isTyping, setIsTyping] = useState(false);
  const [filters, setFilters] = useState<FilterState>({
    dateFrom: '',
    dateTo: '',
    mediaType: 'all',
    keywords: '',
  });
  const scrollRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages, isTyping]);

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      const files = Array.from(e.target.files);
      setSelectedFiles((prev) => [...prev, ...files]);
      toast.success(`${files.length} file(s) attached`);
    }
  };

  const handleSend = async () => {
    if (!input.trim() && selectedFiles.length === 0) return;

    // Create user message
    const userMessage: ChatMessageType = {
      id: Date.now().toString(),
      role: 'user',
      content: input || 'Uploaded files for analysis',
      timestamp: new Date(),
      files: selectedFiles.map((file) => ({
        name: file.name,
        url: URL.createObjectURL(file),
        type: file.type,
      })),
    };

    setMessages([...messages, userMessage]);
    setIsTyping(true);

    try {
      // Use AI agent for search
      const result = await aiSearchArchives(input);
      
      // Convert API response to ArchiveItem format
      const searchResults: ArchiveItem[] = result.archives.map((archive: ArchiveResponse) => ({
        id: archive.id,
        title: archive.title,
        description: archive.description || '',
        type: archive.media_types[0] || 'image',
        date: archive.dates?.[0] || archive.created_at,
        tags: archive.tags || [],
        fileUrl: archive.file_uris?.[0] || archive.storage_paths?.[0] || 'https://via.placeholder.com/400',
        thumbnail: archive.file_uris?.[0] || archive.storage_paths?.[0] || 'https://via.placeholder.com/400',
        file_uris: archive.file_uris || archive.storage_paths || [],
        summary: archive.summary,
      }));

      const aiMessage: ChatMessageType = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: result.message,
        timestamp: new Date(),
        archiveResults: searchResults.length > 0 ? searchResults : undefined,
      };
      
      setMessages((prev: ChatMessageType[]) => [...prev, aiMessage]);
      
      if (searchResults.length > 0) {
        toast.success(`Found ${searchResults.length} matching archive(s)`);
      } else {
        toast.info('No matching archives found. Try different keywords.');
      }
    } catch (error) {
      const errorMessage: ChatMessageType = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: `Sorry, I encountered an error while searching: ${error instanceof Error ? error.message : 'Unknown error'}. Please try again.`,
        timestamp: new Date(),
      };
      setMessages((prev: ChatMessageType[]) => [...prev, errorMessage]);
      toast.error('Search failed. Please try again.');
    } finally {
      setIsTyping(false);
    }

    setInput('');
    setSelectedFiles([]);
  };

  return (
    <div className="bg-white rounded-xl shadow-lg border border-stone-200 flex flex-col h-full">
      {/* Header */}
      <div className="p-4 border-b border-stone-200 flex items-center justify-between">
        <div>
          <h2 className="text-stone-800">AI Heritage Search</h2>
          <p className="text-sm text-stone-500">Ask questions about the collection</p>
        </div>
        <Button
          variant={showFilters ? 'default' : 'outline'}
          size="sm"
          onClick={() => setShowFilters(!showFilters)}
          className={showFilters ? 'bg-forest hover:bg-forest' : ''}
        >
          <Filter className="w-4 h-4 mr-2" />
          Filters
        </Button>
      </div>

      {/* Filters */}
      {showFilters && (
        <ChatFilters filters={filters} setFilters={setFilters} />
      )}

      {/* Messages */}
      <div className="flex-1 overflow-hidden">
        <ScrollArea className="h-full p-4">
          <div className="space-y-4">
            <AnimatePresence>
              {messages.map((message) => (
                <ChatMessage key={message.id} message={message} />
              ))}
              {isTyping && (
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                >
                  <TypingIndicator />
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </ScrollArea>
      </div>

      {/* Input Area */}
      <div className="p-4 border-t border-stone-200">
        {selectedFiles.length > 0 && (
          <div className="mb-2 flex flex-wrap gap-2">
            {selectedFiles.map((file, idx) => (
              <div
                key={idx}
                className="bg-stone-100 px-3 py-1 rounded-full text-xs text-stone-700 flex items-center gap-2"
              >
                {file.name}
                <button
                  onClick={() => setSelectedFiles(selectedFiles.filter((_, i) => i !== idx))}
                  className="text-stone-500 hover:text-stone-700"
                >
                  ×
                </button>
              </div>
            ))}
          </div>
        )}
        
        <div className="flex gap-2">
          <input
            type="file"
            ref={fileInputRef}
            onChange={handleFileSelect}
            className="hidden"
            multiple
            accept="image/*,video/*,audio/*,.pdf,.doc,.docx"
          />
          
          <Button
            variant="outline"
            size="icon"
            onClick={() => fileInputRef.current?.click()}
            className="shrink-0"
          >
            <Paperclip className="w-4 h-4" />
          </Button>
          
          <Input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleSend()}
            placeholder="Search for heritage items..."
            className="flex-1"
          />
          
          <Button
            onClick={handleSend}
            disabled={!input.trim() && selectedFiles.length === 0}
            className="shrink-0 bg-forest hover:bg-forest"
          >
            <Send className="w-4 h-4" />
          </Button>
        </div>
      </div>

      {/* Quick Search Buttons - show only when no messages yet */}
      {messages.length === 1 && (
        <QuickSearchButtons onSearch={async (query) => {
          // Create user message
          const userMessage: ChatMessageType = {
            id: Date.now().toString(),
            role: 'user',
            content: query,
            timestamp: new Date(),
          };

          setMessages([...messages, userMessage]);
          setIsTyping(true);

          try {
            // Use AI agent for search
            const result = await aiSearchArchives(query);
            
            // Convert API response to ArchiveItem format
            const searchResults: ArchiveItem[] = result.archives.map((archive: ArchiveResponse) => ({
              id: archive.id,
              title: archive.title,
              description: archive.description || '',
              type: archive.media_types[0] || 'image',
              date: archive.dates?.[0] || archive.created_at,
              tags: archive.tags || [],
              fileUrl: archive.file_uris?.[0] || archive.storage_paths?.[0] || 'https://via.placeholder.com/400',
              thumbnail: archive.file_uris?.[0] || archive.storage_paths?.[0] || 'https://via.placeholder.com/400',
              file_uris: archive.file_uris || archive.storage_paths || [],
              summary: archive.summary,
            }));

            const aiMessage: ChatMessageType = {
              id: (Date.now() + 1).toString(),
              role: 'assistant',
              content: result.message,
              timestamp: new Date(),
              archiveResults: searchResults.length > 0 ? searchResults : undefined,
            };
            
            setMessages((prev: ChatMessageType[]) => [...prev, aiMessage]);
          } catch (error) {
            const errorMessage: ChatMessageType = {
              id: (Date.now() + 1).toString(),
              role: 'assistant',
              content: `Sorry, I encountered an error while searching: ${error instanceof Error ? error.message : 'Unknown error'}. Please try again.`,
              timestamp: new Date(),
            };
            setMessages((prev: ChatMessageType[]) => [...prev, errorMessage]);
            toast.error('Search failed. Please try again.');
          } finally {
            setIsTyping(false);
          }
        }} />
      )}
    </div>
  );
}