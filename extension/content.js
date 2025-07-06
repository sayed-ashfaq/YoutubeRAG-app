// content.js
console.log("YouTube Summarizer Extension Loaded");

// Create a container for the chatbot
const chatContainer = document.createElement("div");
chatContainer.id = "ashai-chatbox";
chatContainer.innerHTML = `
  <div id="ashai-header">AshAI 🤖</div>
  <div id="ashai-body">
    <p>Hi! I can summarize this video for you 👇</p>
    <input id="ashai-input" type="text" placeholder="Ask me anything..."/>
    <button id="ashai-send">Send</button>
    <div id="ashai-response"></div>
  </div>
`;
document.body.appendChild(chatContainer);
