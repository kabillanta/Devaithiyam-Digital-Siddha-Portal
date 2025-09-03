document.addEventListener("DOMContentLoaded", () => {
    const chatBox = document.getElementById("chat-box");
    const userInput = document.getElementById("user-input");
    const sendBtn = document.getElementById("send-btn");

    let chatHistory = [];

    const sendMessage = async () => {
        const question = userInput.value.trim();
        if (!question) return;

        console.log("Sending question:", question);
        appendMessage(question, "user-message", true);
        userInput.value = "";
        const thinkingIndicator = appendMessage("Thinking...", "bot-message thinking", false);

        try {
            console.log("Making API request to http://127.0.0.1:8000/chat");
            const response = await fetch("http://127.0.0.1:8000/chat", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    question: question,
                    chat_history: chatHistory,
                }),
            });

            console.log("Response status:", response.status);
            console.log("Response ok:", response.ok);

            if (!response.ok) {
                const errorText = await response.text();
                console.error("Response error:", errorText);
                throw new Error(`HTTP ${response.status}: ${errorText}`);
            }

            const data = await response.json();
            console.log("Received data:", data);
            
            if (!data.answer) {
                throw new Error("No answer received from server");
            }
            
            chatBox.removeChild(thinkingIndicator);
            
            // Check if marked is available
            if (typeof marked !== 'undefined' && marked.parse) {
                console.log("Using marked.parse for markdown");
                appendMessage(marked.parse(data.answer), "bot-message", false);
            } else {
                console.log("Marked not available, using plain text");
                appendMessage(data.answer, "bot-message", false);
            }

            chatHistory.push([question, data.answer]);
            console.log("Updated chat history:", chatHistory);

        } catch (error) {
            console.error("Complete error object:", error);
            console.error("Error message:", error.message);
            chatBox.removeChild(thinkingIndicator);
            appendMessage("Sorry, something went wrong. Please try again. Error: " + error.message, "bot-message error", false);
        }
    };

    const appendMessage = (text, className, isUser) => {
        console.log("Appending message:", { text: text.substring(0, 100) + "...", className, isUser });
        
        const messageDiv = document.createElement("div");
        messageDiv.className = `chat-message ${className}`;
        
        let iconHtml = isUser ? '<i class="fas fa-user user-icon"></i>' : '<i class="fas fa-leaf bot-icon"></i>';
        if (className.includes('thinking')) iconHtml = '';

        // For user messages, wrap text in a <p> tag. For bot messages, insert the parsed HTML.
        const contentHtml = isUser ? `<p>${text}</p>` : text;

        messageDiv.innerHTML = `${iconHtml}<div class="message-content">${contentHtml}</div>`;
        chatBox.appendChild(messageDiv);
        chatBox.scrollTop = chatBox.scrollHeight;
        
        console.log("Message appended successfully");
        return messageDiv;
    };

    sendBtn.addEventListener("click", sendMessage);
    userInput.addEventListener("keypress", (event) => {
        if (event.key === "Enter") sendMessage();
    });
});