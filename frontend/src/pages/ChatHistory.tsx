import React from 'react';

interface ChatHistoryItem {
  messages: any[];
  summary: string;
}

interface ChatHistoryProps {
  history: ChatHistoryItem[];
  onNewChat: () => void;
  collapsed: boolean;
  onToggleCollapse: () => void;
}

export default function ChatHistory({ history, onNewChat, collapsed, onToggleCollapse }: ChatHistoryProps) {
  return (
    <div className="chat-history">
      <div className="sidebar-header">
        {collapsed ? (
          <img src="/AppgallopSM.png" alt="AppGallop" className="sidebar-logo-small" onClick={onToggleCollapse} />
        ) : (
          <div className="sidebar-logo-section">
            <img src="/AppgallopLG.png" alt="AppGallop" className="sidebar-logo-large" />
            <button className="collapse-btn" onClick={onToggleCollapse}>
              ←
            </button>
          </div>
        )}
      </div>
      {!collapsed && (
        <>
          <button className="new-chat" onClick={onNewChat}>+ New Chat</button>
          <h4>History</h4>
          {history.length === 0 ? (
            <p className="no-history">No chats yet</p>
          ) : (
            <ul>
              {history.map((item, index) => (
                <li key={index}>{item.summary}</li>
              ))}
            </ul>
          )}
        </>
      )}
    </div>
  );
}
