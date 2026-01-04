import { User } from 'lucide-react';
import { Avatar, AvatarFallback } from './ui/avatar';

export function TopBar() {
  return (
    <header className="h-16 bg-white border-b border-stone-200 flex items-center justify-between px-6 shadow-sm">
      <div>
        <h1 className="text-stone-800">Badan Warisan Digital Archive</h1>
        <p className="text-xs text-stone-500">Interactive Heritage Collection Management</p>
      </div>

      <div className="flex items-center gap-4">
        <div className="flex items-center gap-3">
          <div className="text-right">
            <p className="text-sm text-stone-800">Curator Admin</p>
            <p className="text-xs text-stone-500">Museum Manager</p>
          </div>
          <Avatar>
            <AvatarFallback className="bg-forest text-white">
              <User className="w-5 h-5" />
            </AvatarFallback>
          </Avatar>
        </div>
      </div>
    </header>
  );
}
