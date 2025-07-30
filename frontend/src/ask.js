import React, { useState } from "react";
import { askQuestion } from "./services/api";

function Ask() {
  const [fileId, setFileId] = useState("");
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");

  const handleAsk = async () => {
    const res = await askQuestion(fileId, question);
    setAnswer(res.answer);
  };

  return (
    <div>
      <h2>Ask a Question</h2>
      <input
        placeholder="File name"
        onChange={(e) => setFileId(e.target.value)}
      />
      <br />
      <input
        placeholder="Your question"
        onChange={(e) => setQuestion(e.target.value)}
      />
      <br />
      <button onClick={handleAsk}>Ask</button>
      <p><strong>Answer:</strong> {answer}</p>
    </div>
  );
}

export default Ask;
