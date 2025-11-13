import { motion } from 'motion/react';
import { Bot, User, Image as ImageIcon, Video, FileText, Music } from 'lucide-react';
import type { ChatMessage as ChatMessageType } from '../App';
import { ArchiveCard } from './ArchiveCard';
import { ArchiveDetailModal } from './ArchiveDetailModal';
import { useState } from 'react';

type ChatMessageProps = {
  message: ChatMessageType;
};

export function ChatMessage({ message }: ChatMessageProps) {
  const isUser = message.role === 'user';
  const [detailOpen, setDetailOpen] = useState(false);
  const [selectedItem, setSelectedItem] = useState<any | null>(null);

  const getFileIcon = (type: string) => {
    if (type.startsWith('image/')) return ImageIcon;
    if (type.startsWith('video/')) return Video;
    if (type.startsWith('audio/')) return Music;
    return FileText;
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
      className={`flex gap-3 ${isUser ? 'flex-row-reverse' : ''}`}
    >
      {/* Avatar */}
      <div
        className={`w-8 h-8 rounded-full flex items-center justify-center shrink-0 ${
          isUser ? 'bg-forest' : 'bg-stone-600'
        }`}
      >
        {isUser ? (
          <User className="w-4 h-4 text-white" />
        ) : (
          <Bot className="w-4 h-4 text-white" />
        )}
      </div>

      {/* Message Content */}
      <div className={`flex-1 ${isUser ? 'flex flex-col items-end' : ''}`}>
        <div
          className={`inline-block px-4 py-3 rounded-lg ${
            isUser
              ? 'bg-forest text-white max-w-[80%]'
              : 'bg-stone-100 text-stone-800 max-w-full'
          }`}
        >
          <p className="text-sm whitespace-pre-wrap">{message.content}</p>
          
          {/* Attached Files */}
          {message.files && message.files.length > 0 && (
            <div className="mt-3 space-y-2">
              {message.files.map((file, idx) => {
                const Icon = getFileIcon(file.type);
                return (
                  <div
                    key={idx}
                    className={`flex items-center gap-2 p-2 rounded ${
                      isUser ? 'bg-amber-700' : 'bg-white'
                    }`}
                  >
                    <Icon className="w-4 h-4" />
                    <span className="text-xs truncate">{file.name}</span>
                  </div>
                );
              })}
            </div>
          )}
        </div>

        {/* Archive Results */}
        {message.archiveResults && message.archiveResults.length > 0 && (
          <div className="mt-3 space-y-2 w-full">
            {message.archiveResults.map((item) => (
              <ArchiveCard key={item.id} item={item} compact onView={(it) => { setSelectedItem(it); setDetailOpen(true); }} />
            ))}
          </div>
        )}

        <p className="text-xs text-stone-400 mt-1">
          {message.timestamp.toLocaleTimeString('en-MY', {
            hour: '2-digit',
            minute: '2-digit',
          })}
        </p>
      </div>
      <ArchiveDetailModal
        isOpen={detailOpen}
        item={selectedItem}
        onClose={() => setDetailOpen(false)}
      />
    </motion.div>
  );
}