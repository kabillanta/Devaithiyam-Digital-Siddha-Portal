document.addEventListener("DOMContentLoaded", () => {
    const chatBox = document.getElementById("chat-box");
    const userInput = document.getElementById("user-input");
    const sendBtn = document.getElementById("send-btn");

    let chatHistory = [];

    const sendMessage = async () => {
        const question = userInput.value.trim();
        if (!question) return;

        appendMessage(question, "user-message", true);
        userInput.value = "";
        const thinkingIndicator = appendMessage("Thinking...", "bot-message thinking", false);

        try {
            const response = await fetch("http://127.0.0.1:8080/chat", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    question: question,
                    chat_history: chatHistory,
                }),
            });

            if (!response.ok) throw new Error("Network response was not ok.");

            const data = await response.json();
            
            chatBox.removeChild(thinkingIndicator);
            // UPDATED: Use marked.parse() on the bot's answer
            appendMessage(marked.parse(data.answer), "bot-message", false);

            chatHistory.push([question, data.answer]);

        } catch (error) {
            console.error("Error:", error);
            chatBox.removeChild(thinkingIndicator);
            appendMessage("Sorry, something went wrong. Please try again.", "bot-message error", false);
        }
    };

    const appendMessage = (text, className, isUser) => {
        const messageDiv = document.createElement("div");
        messageDiv.className = `chat-message ${className}`;
        
        let iconHtml = isUser ? '<i class="fas fa-user user-icon"></i>' : '<i class="fas fa-leaf bot-icon"></i>';
        if (className.includes('thinking')) iconHtml = '';

        // For user messages, wrap text in a <p> tag. For bot messages, insert the parsed HTML.
        const contentHtml = isUser ? `<p>${text}</p>` : text;

        messageDiv.innerHTML = `${iconHtml}<div class="message-content">${contentHtml}</div>`;
        chatBox.appendChild(messageDiv);
        chatBox.scrollTop = chatBox.scrollHeight;
        return messageDiv;
    };

    sendBtn.addEventListener("click", sendMessage);
    userInput.addEventListener("keypress", (event) => {
        if (event.key === "Enter") sendMessage();
    });
});