// import React, { useState } from 'react';

// const ChatbotWidget = ({ onSendMessage }) => {
//   const [input, setInput] = useState('');

//   const handleSend = () => {
//     if (input.trim()) {
//       onSendMessage(input);
//       setInput('');
//     }
//   };

//   return (
//     <div className="chatbot-widget">
//       <input
//         type="text"
//         value={input}
//         onChange={(e) => setInput(e.target.value)}
//         placeholder="Type your message..."
//       />
//       <button onClick={handleSend}>Send</button>
//     </div>
//   );
// };

// export default ChatbotWidget;